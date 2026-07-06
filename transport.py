"""
Analog kernel-based event-driven neutron transport.

This module handles:
    - SimpleNeutron class
    - box boundary handling
    - physical event scheduling by time
    - real collision processing
    - MF4/MF6 energy-direction updates
    - secondary neutron creation
    - residual product logging
"""

import heapq
import numpy as np
import pandas as pd

from nuclear_data import (
    neutron_speed_from_energy,
    build_material_reaction_list_for_energy,
    build_material_reaction_list_for_energy_cached,
    total_macroscopic_xs,
    sample_free_path,
    count_outgoing_neutrons,
    mt_name,
    residual_product_for_mt,
)

from transport_kernel import (
    get_kernel_reaction_rows,
    sample_reaction_from_kernel_cdf,
    sample_neutron_multiplicity_from_kernel,
    sample_residual_product_from_kernel,
    sample_elastic_mf4_update_from_kernel,
    sample_mf6_outgoing_neutron_energy_from_kernel,
)

from mf6 import sample_mf6_outgoing_neutron_energy
from mf4 import sample_elastic_mf4_update, rotate_direction_2d


def random_unit_vector_2d():
    """Return a random 2D unit vector."""
    angle = 2.0 * np.pi * np.random.random()
    return np.array([np.cos(angle), np.sin(angle)])


def inside_box(x, y, box_size_m):
    """Return True if position is inside square box centered at origin."""
    half = box_size_m / 2.0
    return (-half <= x <= half) and (-half <= y <= half)


class SimpleNeutron:
    """A very simple neutron object with history logging."""

    def __init__(
        self,
        energy_eV,
        x=0.0,
        y=0.0,
        direction=None,
        t=0.0,
        box_size_m=None,
        idx=0,
        generation=0,
        parent_idx=None,
    ):
        if direction is None:
            direction = [1.0, 0.0]

        self.energy_eV = float(energy_eV)
        self.x = float(x)
        self.y = float(y)
        self.t = float(t)

        self.direction = np.array(direction, dtype=float)
        mag = np.linalg.norm(self.direction)
        if mag == 0.0:
            raise ValueError("Direction vector cannot be zero.")
        self.direction = self.direction / mag

        self.box_size_m = box_size_m
        self.idx = int(idx)
        self.generation = int(generation)
        self.parent_idx = parent_idx
        self.alive = True
        self.history = []
        self.record(event="start")

    def speed(self):
        return neutron_speed_from_energy(self.energy_eV)

    def set_energy(self, energy_eV):
        self.energy_eV = max(float(energy_eV), 0.0)
        if self.energy_eV <= 0.0:
            self.alive = False

    def set_direction(self, direction):
        direction = np.array(direction, dtype=float)
        mag = np.linalg.norm(direction)
        if mag == 0.0:
            raise ValueError("Direction vector cannot be zero.")
        self.direction = direction / mag

    def move(self, distance_m):
        distance_m = float(distance_m)
        self.x = self.x + self.direction[0] * distance_m
        self.y = self.y + self.direction[1] * distance_m

        v = self.speed()
        if v > 0.0:
            self.t = self.t + distance_m / v
        else:
            self.t = float("inf")

        if self.box_size_m is not None:
            if not inside_box(self.x, self.y, self.box_size_m):
                self.alive = False
                return "escaped"
        return "inside"

    def record(self, event, mt=None, reaction_name=None, distance_m=None):
        row = {
            "neutron_id": self.idx,
            "generation": self.generation,
            "parent_id": self.parent_idx,
            "t": self.t,
            "x": self.x,
            "y": self.y,
            "energy_eV": self.energy_eV,
            "dir_x": self.direction[0],
            "dir_y": self.direction[1],
            "event": event,
            "mt": mt,
            "reaction_name": reaction_name,
            "distance_m": distance_m,
        }
        self.history.append(row)

    def print_state(self):
        print("neutron id =", self.idx)
        print("generation =", self.generation)
        print("parent id =", self.parent_idx)
        print("x =", self.x, "m")
        print("y =", self.y, "m")
        print("t =", self.t, "s")
        print("energy =", self.energy_eV, "eV")
        print("speed =", self.speed(), "m/s")
        print("direction =", self.direction)
        print("alive =", self.alive)


def mf6_outgoing_neutron_samples(chosen_reaction, data, mt, incoming_energy_eV, n_out):
    """Sample outgoing neutron energies and keep source labels.

    Preferred path: precomputed MF6 kernel attached to chosen_reaction.
    Fallback path: old runtime MF6 sampler.
    """
    samples = []
    for _ in range(int(n_out)):
        mf6_row = sample_mf6_outgoing_neutron_energy_from_kernel(
            chosen_reaction=chosen_reaction,
            incoming_energy_eV=incoming_energy_eV,
        )

        if mf6_row is None:
            mf6_row = sample_mf6_outgoing_neutron_energy(data, mt, incoming_energy_eV)

        if mf6_row is None:
            samples.append({"energy_eV": float(incoming_energy_eV), "source": "unchanged-no-MF6"})
        else:
            samples.append({
                "energy_eV": float(mf6_row["energy_eV"]),
                "source": mf6_row["source"],
                "law": mf6_row.get("law"),
                "product_index": mf6_row.get("product_index"),
                "yield_at_E": mf6_row.get("yield_at_E"),
                "mu_cm": mf6_row.get("mu_cm"),
                "mu_lab": mf6_row.get("mu_lab"),
                "theta_lab_rad": mf6_row.get("theta_lab_rad"),
                "theta_lab_deg": mf6_row.get("theta_lab_deg"),
                "angle_source": mf6_row.get("angle_source"),
                "angle_frame": mf6_row.get("angle_frame"),
                "mf6_product_random_number": mf6_row.get("mf6_product_random_number"),
                "mf6_sample_random_number": mf6_row.get("mf6_sample_random_number"),
                "mf6_product_probability": mf6_row.get("mf6_product_probability"),
                "mf6_product_cdf": mf6_row.get("mf6_product_cdf"),
                "mf6_product_weight_source": mf6_row.get("mf6_product_weight_source"),
            })
    return samples


def outgoing_neutron_count(chosen_reaction, mt):
    """Sample integer outgoing-neutron count from kernel metadata or MT fallback."""
    multiplicity = sample_neutron_multiplicity_from_kernel(chosen_reaction)
    if multiplicity is not None:
        return multiplicity

    n_out = int(count_outgoing_neutrons(mt))
    return {
        "n_out": n_out,
        "n_out_expected": float(n_out),
        "n_out_integer_rule": "fixed-integer" if n_out > 0 else "zero",
        "n_out_source": "runtime-MT-emitted-particle-fallback",
        "n_out_random_number": None,
        "n_out_mt_count": n_out,
        "n_out_mf6_total_yield": None,
        "n_out_mf6_product_count": None,
    }


def direction_from_outgoing_sample(incoming_direction, sample):
    """Return a 2D outgoing direction and angle diagnostics for one sample."""
    theta_lab_rad = sample.get("theta_lab_rad")

    if theta_lab_rad is not None and np.isfinite(float(theta_lab_rad)):
        azimuth_random = float(np.random.random())
        sign = -1.0 if azimuth_random < 0.5 else 1.0
        direction = rotate_direction_2d(incoming_direction, sign * float(theta_lab_rad))
        return {
            "direction": direction,
            "mu_cm": sample.get("mu_cm"),
            "mu_lab": sample.get("mu_lab"),
            "theta_lab_deg": sample.get("theta_lab_deg"),
            "angle_source": sample.get("angle_source"),
            "angle_frame": sample.get("angle_frame"),
            "angle_azimuth_random_number": azimuth_random,
        }

    return {
        "direction": random_unit_vector_2d(),
        "mu_cm": sample.get("mu_cm"),
        "mu_lab": sample.get("mu_lab"),
        "theta_lab_deg": sample.get("theta_lab_deg"),
        "angle_source": "random-2D-no-MF6-angle",
        "angle_frame": "lab",
        "angle_azimuth_random_number": None,
    }


def apply_basic_reaction_effect(neutron, chosen_reaction, data):
    """Apply reaction effect after a collision."""
    mt = int(chosen_reaction["mt"])
    multiplicity_info = outgoing_neutron_count(chosen_reaction, mt)
    n_out = int(multiplicity_info["n_out"])

    incoming_energy = neutron.energy_eV
    incoming_direction = neutron.direction.copy()
    outgoing_energy = None
    secondary_neutrons = []
    energy_update_source = None

    mu_cm = None
    mu_lab = None
    theta_lab_deg = None
    angle_source = None
    angle_frame = None
    angle_azimuth_random_number = None

    if n_out <= 0:
        neutron.alive = False
        energy_update_source = "absorbed-no-outgoing-neutron"
        event = "absorbed"

    else:
        if mt == 2:
            elastic_row = sample_elastic_mf4_update_from_kernel(
                chosen_reaction=chosen_reaction,
                incoming_energy_eV=incoming_energy,
            )

            if elastic_row is None:
                elastic_row = sample_elastic_mf4_update(
                    data=data,
                    incoming_energy_eV=incoming_energy,
                )
 
            if elastic_row is not None:
                outgoing_energy = elastic_row["energy_eV"]
                energy_update_source = elastic_row["energy_update_source"]
                mu_cm = elastic_row["mu_cm"]
                mu_lab = elastic_row["mu_lab"]
                theta_lab_deg = elastic_row["theta_lab_deg"]
                angle_source = elastic_row["angle_source"]
                angle_frame = elastic_row["angle_frame"]

                neutron.set_energy(outgoing_energy)
                angle_azimuth_random_number = float(np.random.random())
                sign = -1.0 if angle_azimuth_random_number < 0.5 else 1.0
                theta = sign * elastic_row["theta_lab_rad"]
                neutron.set_direction(rotate_direction_2d(neutron.direction, theta))
            else:
                outgoing_energy = incoming_energy
                energy_update_source = "elastic-unchanged-no-MF4"
                neutron.set_direction(random_unit_vector_2d())

            event = "scatter-like"

        else:
            outgoing_samples = mf6_outgoing_neutron_samples(
                chosen_reaction=chosen_reaction,
                data=data,
                mt=mt,
                incoming_energy_eV=incoming_energy,
                n_out=n_out,
            )
            first_sample = outgoing_samples[0]
            outgoing_energy = first_sample["energy_eV"]
            energy_update_source = first_sample["source"]
            direction_row = direction_from_outgoing_sample(incoming_direction, first_sample)

            neutron.set_energy(outgoing_energy)
            neutron.set_direction(direction_row["direction"])

            mu_cm = direction_row["mu_cm"]
            mu_lab = direction_row["mu_lab"]
            theta_lab_deg = direction_row["theta_lab_deg"]
            angle_source = direction_row["angle_source"]
            angle_frame = direction_row["angle_frame"]
            angle_azimuth_random_number = direction_row["angle_azimuth_random_number"]

            for sample in outgoing_samples[1:]:
                secondary_direction_row = direction_from_outgoing_sample(incoming_direction, sample)
                secondary_neutrons.append({
                    "energy_eV": sample["energy_eV"],
                    "direction": secondary_direction_row["direction"],
                    "energy_update_source": sample.get("source"),
                    "mu_cm": secondary_direction_row["mu_cm"],
                    "mu_lab": secondary_direction_row["mu_lab"],
                    "theta_lab_deg": secondary_direction_row["theta_lab_deg"],
                    "angle_source": secondary_direction_row["angle_source"],
                    "angle_frame": secondary_direction_row["angle_frame"],
                    "angle_azimuth_random_number": secondary_direction_row["angle_azimuth_random_number"],
                })

            if n_out == 1:
                event = "scatter-like"
            else:
                event = "neutron multiplication"

    return (
        event,
        n_out,
        incoming_energy,
        outgoing_energy,
        energy_update_source,
        secondary_neutrons,
        mu_cm,
        mu_lab,
        theta_lab_deg,
        angle_source,
        angle_frame,
        angle_azimuth_random_number,
        multiplicity_info,
    )


def make_secondary_neutrons(parent_neutron, secondary_specs, next_idx):
    """Create secondary neutron objects from extra outgoing neutron samples."""
    secondaries = []
    for j, spec in enumerate(secondary_specs):
        if isinstance(spec, dict):
            energy_eV = spec["energy_eV"]
            direction = spec.get("direction")
        else:
            energy_eV = spec
            direction = None
        if direction is None:
            direction = random_unit_vector_2d()

        secondary = SimpleNeutron(
            energy_eV=energy_eV,
            x=parent_neutron.x,
            y=parent_neutron.y,
            direction=direction,
            t=parent_neutron.t,
            box_size_m=parent_neutron.box_size_m,
            idx=next_idx + j,
            generation=parent_neutron.generation + 1,
            parent_idx=parent_neutron.idx,
        )
        secondary.record(event="created secondary", distance_m=0.0)
        if isinstance(spec, dict):
            row = secondary.history[-1]
            row["energy_update_source"] = spec.get("energy_update_source")
            row["mu_cm"] = spec.get("mu_cm")
            row["mu_lab"] = spec.get("mu_lab")
            row["theta_lab_deg"] = spec.get("theta_lab_deg")
            row["angle_source"] = spec.get("angle_source")
            row["angle_frame"] = spec.get("angle_frame")
            row["angle_azimuth_random_number"] = spec.get("angle_azimuth_random_number")
        secondaries.append(secondary)
    return secondaries


def distance_to_box_boundary(neutron):
    """Distance from neutron to square box boundary along its direction."""
    if neutron.box_size_m is None:
        return float("inf")

    x = neutron.x
    y = neutron.y
    dx = neutron.direction[0]
    dy = neutron.direction[1]
    half = neutron.box_size_m / 2.0

    distances = []
    if dx > 0.0:
        distances.append((half - x) / dx)
    elif dx < 0.0:
        distances.append((-half - x) / dx)

    if dy > 0.0:
        distances.append((half - y) / dy)
    elif dy < 0.0:
        distances.append((-half - y) / dy)

    forward = [d for d in distances if d >= 0.0]
    if len(forward) == 0:
        return 0.0
    return min(forward)


def schedule_next_event_material(neutron, material, kernel=None):
    """Schedule the next event for one neutron in a material mixture.

    If kernel is provided, use the precomputed material reaction kernel.
    If kernel is not provided, fall back to the older cached reactio-list method.
    """
    if not neutron.alive:
        return None

    if kernel is None:
        open_reactions = build_material_reaction_list_for_energy_cached(
            material=material, 
            energy_eV=neutron.energy_eV, 
            relative_width=0.01
        )
        Sigma_total = total_macroscopic_xs(open_reactions)

    else:
        open_reactions, Sigma_total = get_kernel_reaction_rows(
            kernel=kernel,
            energy_eV=neutron.energy_eV,
        )
        
        
    if Sigma_total <= 0.0:
        return {
            "event_type": "no open reactions",
            "neutron_id": neutron.idx,
            "event_time": neutron.t,
            "distance_m": 0.0,
            "Sigma_total_1_per_m": Sigma_total,
            "mean_free_path_m": float("inf"),
            "open_reactions": open_reactions,
        }

    collision_distance = sample_free_path(Sigma_total)
    boundary_distance = distance_to_box_boundary(neutron)
    v = neutron.speed()

    if v <= 0.0:
        return {
            "event_type": "no speed",
            "neutron_id": neutron.idx,
            "event_time": neutron.t,
            "distance_m": 0.0,
            "Sigma_total_1_per_m": Sigma_total,
            "mean_free_path_m": 1.0 / Sigma_total,
            "open_reactions": open_reactions,
        }

    if boundary_distance <= collision_distance:
        distance_m = boundary_distance
        event_type = "escaped"
    else:
        distance_m = collision_distance
        event_type = "collision"

    return {
        "event_type": event_type,
        "neutron_id": neutron.idx,
        "event_time": neutron.t + distance_m / v,
        "distance_m": distance_m,
        "Sigma_total_1_per_m": Sigma_total,
        "mean_free_path_m": 1.0 / Sigma_total,
        "open_reactions": open_reactions,
    }


def process_event_material(event, neutrons_by_id, material, next_idx):
    """Process one event from the event queue using a material mixture."""
    new_secondaries = []
    neutron_id = event["neutron_id"]

    if neutron_id not in neutrons_by_id:
        return new_secondaries, next_idx

    neutron = neutrons_by_id[neutron_id]
    if not neutron.alive:
        return new_secondaries, next_idx

    event_type = event["event_type"]
    distance_m = event["distance_m"]

    if event_type == "no open reactions":
        neutron.alive = False
        neutron.record(event="no open reactions", distance_m=distance_m)
        neutron.history[-1]["Sigma_total_1_per_m"] = event["Sigma_total_1_per_m"]
        neutron.history[-1]["mean_free_path_m"] = event["mean_free_path_m"]
        return new_secondaries, next_idx

    if event_type == "no speed":
        neutron.alive = False
        neutron.record(event="no speed", distance_m=distance_m)
        neutron.history[-1]["Sigma_total_1_per_m"] = event["Sigma_total_1_per_m"]
        neutron.history[-1]["mean_free_path_m"] = event["mean_free_path_m"]
        return new_secondaries, next_idx

    neutron.move(distance_m)

    if event_type == "escaped":
        neutron.alive = False
        neutron.record(event="escaped", distance_m=distance_m)
        neutron.history[-1]["Sigma_total_1_per_m"] = event["Sigma_total_1_per_m"]
        neutron.history[-1]["mean_free_path_m"] = event["mean_free_path_m"]
        return new_secondaries, next_idx

    if event_type == "collision":
        open_reactions = event["open_reactions"]
        chosen_reaction = sample_reaction_from_kernel_cdf(open_reactions)
        if chosen_reaction is None:
            neutron.alive = False
            neutron.record(event="no chosen reaction", distance_m=distance_m)
            return new_secondaries, next_idx

        target_isotope = chosen_reaction["target_isotope"]
        target_data = chosen_reaction["target_data"]
        collision_energy_eV = neutron.energy_eV
        product_info = sample_residual_product_from_kernel(chosen_reaction)
        if product_info is None:
            product_info = residual_product_for_mt(
                target_data,
                chosen_reaction["mt"],
                energy_eV=collision_energy_eV,
            )

        (
            collision_event,
            n_out,
            incoming_energy,
            outgoing_energy,
            energy_update_source,
            secondary_neutrons,
            mu_cm,
            mu_lab,
            theta_lab_deg,
            angle_source,
            angle_frame,
            angle_azimuth_random_number,
            multiplicity_info,
        ) = apply_basic_reaction_effect(neutron, chosen_reaction, target_data)

        neutron.record(
            event=collision_event,
            mt=chosen_reaction["mt"],
            reaction_name=mt_name(chosen_reaction["mt"]),
            distance_m=distance_m,
        )

        row = neutron.history[-1]
        row["target_isotope"] = target_isotope
        row["residual_product"] = product_info["product"]
        row["residual_product_Z"] = product_info["product_Z"]
        row["residual_product_A"] = product_info["product_A"]
        row["product_note"] = product_info["note"]
        row["product_state"] = product_info.get("product_state")
        row["product_state_source"] = product_info.get("product_state_source")
        row["product_branch_probability"] = product_info.get("product_branch_probability")
        row["product_branch_total_xs"] = product_info.get("product_branch_total_xs")
        row["residual_product_sampling_source"] = product_info.get("residual_product_sampling_source")
        row["residual_product_random_number"] = product_info.get("residual_product_random_number")
        row["residual_product_branch_cdf"] = product_info.get("residual_product_branch_cdf")
        row["n_out"] = n_out
        row["alive_after_collision"] = neutron.alive
        row["Sigma_total_1_per_m"] = event["Sigma_total_1_per_m"]
        row["mean_free_path_m"] = event["mean_free_path_m"]
        row["chosen_probability"] = chosen_reaction.get("probability")
        row["chosen_cdf"] = chosen_reaction.get("cdf")
        row["reaction_random_number"] = chosen_reaction.get("reaction_random_number")
        row["reaction_sampling_source"] = chosen_reaction.get("reaction_sampling_source")
        row["kernel_bin_index"] = chosen_reaction.get("kernel_bin_index")
        row["kernel_energy_eV"] = chosen_reaction.get("kernel_energy_eV")
        row["kernel_E_low_eV"] = chosen_reaction.get("kernel_E_low_eV")
        row["kernel_E_high_eV"] = chosen_reaction.get("kernel_E_high_eV")
        row["incoming_energy_eV"] = incoming_energy
        row["outgoing_energy_eV"] = outgoing_energy
        row["energy_update_source"] = energy_update_source
        row["mu_cm"] = mu_cm
        row["mu_lab"] = mu_lab
        row["theta_lab_deg"] = theta_lab_deg
        row["angle_source"] = angle_source
        row["angle_frame"] = angle_frame
        row["angle_azimuth_random_number"] = angle_azimuth_random_number
        row["n_out_expected"] = multiplicity_info.get("n_out_expected")
        row["n_out_integer_rule"] = multiplicity_info.get("n_out_integer_rule")
        row["n_out_source"] = multiplicity_info.get("n_out_source")
        row["n_out_random_number"] = multiplicity_info.get("n_out_random_number")
        row["n_out_mt_count"] = multiplicity_info.get("n_out_mt_count")
        row["n_out_mf6_total_yield"] = multiplicity_info.get("n_out_mf6_total_yield")
        row["n_out_mf6_product_count"] = multiplicity_info.get("n_out_mf6_product_count")
        row["elastic_has_mf4_kernel"] = chosen_reaction.get("elastic_has_mf4_kernel", False)
        row["mf6_has_kernel"] = chosen_reaction.get("mf6_has_kernel", False)
        row["mf6_product_count"] = chosen_reaction.get("mf6_product_count")
        row["mf6_total_neutron_product_count"] = chosen_reaction.get("mf6_total_neutron_product_count")
        row["mf6_missing_product_count"] = chosen_reaction.get("mf6_missing_product_count")
        row["num_secondaries_created"] = len(secondary_neutrons)

        if len(secondary_neutrons) > 0:
            new_secondaries = make_secondary_neutrons(neutron, secondary_neutrons, next_idx)
            next_idx += len(new_secondaries)

        return new_secondaries, next_idx

    return new_secondaries, next_idx


def run_neutron_population_event_driven_material(
    starting_neutron,
    material,
    max_events=1000,
    max_neutrons=1000,
    kernel=None,
):
    """Run event-driven neutron transport in a material mixture."""
    neutrons_by_id = {starting_neutron.idx: starting_neutron}
    next_idx = starting_neutron.idx + 1
    event_queue = []
    event_counter = 0

    first_event = schedule_next_event_material(starting_neutron, material, kernel=kernel)
    
    if first_event is not None:
        heapq.heappush(event_queue, (first_event["event_time"], event_counter, first_event))
        event_counter += 1

    events_processed = 0

    while len(event_queue) > 0:
        if events_processed >= max_events:
            break
        if len(neutrons_by_id) >= max_neutrons:
            break

        event_time, _, event = heapq.heappop(event_queue)
        neutron_id = event["neutron_id"]
        if neutron_id not in neutrons_by_id:
            continue

        neutron = neutrons_by_id[neutron_id]
        if not neutron.alive:
            continue

        new_secondaries, next_idx = process_event_material(event, neutrons_by_id, material, next_idx)
        events_processed += 1

        for secondary in new_secondaries:
            neutrons_by_id[secondary.idx] = secondary
            secondary_event = schedule_next_event_material(secondary, material, kernel=kernel)
            if secondary_event is not None:
                heapq.heappush(event_queue, (secondary_event["event_time"], event_counter, secondary_event))
                event_counter += 1

        if neutron.alive:
            next_event = schedule_next_event_material(neutron, material, kernel=kernel)
            if next_event is not None:
                heapq.heappush(event_queue, (next_event["event_time"], event_counter, next_event))
                event_counter += 1

    return list(neutrons_by_id.values())


def neutron_history_dataframe(neutron):
    return pd.DataFrame(neutron.history)


def all_neutron_histories_dataframe(neutrons):
    rows = []
    for neutron in neutrons:
        for row in neutron.history:
            rows.append(row)
    return pd.DataFrame(rows)


def secondary_creation_diagnostics(hist_df):
    """Diagnose whether secondary neutrons were created correctly."""
    print("Total history rows:", len(hist_df))
    if "neutron_id" not in hist_df.columns:
        print("No neutron_id column found.")
        return
    print("Total neutron IDs:", hist_df["neutron_id"].nunique())
    print("\nEvent counts:")
    print(hist_df["event"].value_counts(dropna=False))

    multiplication_rows = hist_df[hist_df["event"] == "neutron multiplication"]
    secondary_rows = hist_df[hist_df["event"] == "created secondary"]
    print("\nNumber of multiplication events:", len(multiplication_rows))
    print("Number of created secondary rows:", len(secondary_rows))

    expected_total = 0
    for _, row in multiplication_rows.iterrows():
        expected_total += max(int(row["n_out"]) - 1, 0)
    print("Expected secondaries:", expected_total)
    print("Actual created secondary rows:", len(secondary_rows))


def check_event_time_ordering(hist_df):
    """Check whether physical events are monotonic in time."""
    physical_events = [
        "scatter-like",
        "neutron multiplication",
        "absorbed",
        "escaped",
        "no open reactions",
        "no speed",
    ]
    event_rows = hist_df[hist_df["event"].isin(physical_events)].copy().sort_values("t")
    if len(event_rows) == 0:
        print("No physical event rows found.")
        return
    times = event_rows["t"].values
    print("Events are monotonic in time?", np.all(np.diff(times) >= 0.0))
    return event_rows
