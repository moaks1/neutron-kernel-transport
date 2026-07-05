import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from nuclear_data import (
    material_from_composition_file,
    build_material_reaction_list_for_energy,
    total_macroscopic_xs,
    mt_name,
    clear_material_reaction_cache,
    material_reaction_cache_summary,
)
from transport import (
    SimpleNeutron,
    random_unit_vector_2d,
    run_neutron_population_event_driven_material,
    all_neutron_histories_dataframe,
    neutron_history_dataframe,
    secondary_creation_diagnostics,
    check_event_time_ordering,
)
from decay_activity import (
    activation_product_dataframe,
    add_decay_status_to_activation_products,
    direct_activity_curves_from_history,
    evolve_decay_chains_from_history,
)


def set_random_seed(random_seed):
    """Seed NumPy's global random generator used by the transport code."""
    if random_seed is not None:
        np.random.seed(int(random_seed))


def random_unit_vector(rng):
    """Return a random 2D unit vector using a NumPy Generator."""
    direction = rng.normal(0.0, 1.0, 2)
    mag = np.linalg.norm(direction)
    if mag == 0.0:
        return np.array([1.0, 0.0])
    return direction / mag


def resolve_worker_count(n_workers, n_tasks):
    """Normalize notebook worker settings to a safe process count."""
    n_tasks = int(n_tasks)
    if n_tasks <= 1 or n_workers is None:
        return 1

    if isinstance(n_workers, str):
        text = n_workers.strip().lower()
        if text in {"auto", "all"}:
            requested = os.cpu_count() or 1
        else:
            requested = int(text)
    else:
        requested = int(n_workers)

    if requested <= 1:
        return 1
    return min(requested, n_tasks)


def make_source_seeds(n_source_neutrons, random_seed):
    """Return independent uint32 seeds for source histories in parallel mode."""
    if random_seed is None:
        seed_sequence = np.random.SeedSequence()
    else:
        seed_sequence = np.random.SeedSequence(int(random_seed))

    child_sequences = seed_sequence.spawn(int(n_source_neutrons))
    return [int(seq.generate_state(1, dtype=np.uint32)[0]) for seq in child_sequences]


def kernel_payload_for_workers(kernel):
    """Drop lazy flat-kernel views before sending a kernel to worker processes."""
    if isinstance(kernel, dict) and kernel.get("flat_storage", False) and "bins" in kernel:
        payload = dict(kernel)
        payload.pop("bins", None)
        return payload
    return kernel


_WORKER_MATERIAL = None
_WORKER_KERNEL = None
_WORKER_SETTINGS = None


def many_source_worker_init(material, kernel, settings):
    """Attach shared transport inputs once per worker process."""
    global _WORKER_MATERIAL, _WORKER_KERNEL, _WORKER_SETTINGS
    _WORKER_MATERIAL = material
    _WORKER_KERNEL = kernel
    _WORKER_SETTINGS = settings


def run_one_source_worker(task):
    """Run one source-neutron history inside a worker process."""
    source_id, source_seed, starting_idx = task
    settings = _WORKER_SETTINGS

    if source_seed is not None:
        np.random.seed(int(source_seed))

    starting_neutron = SimpleNeutron(
        energy_eV=settings["initial_energy_eV"],
        x=settings["start_x"],
        y=settings["start_y"],
        direction=random_unit_vector_2d(),
        box_size_m=settings["box_size_m"],
        idx=int(starting_idx),
        generation=0,
        parent_idx=None,
    )

    source_neutrons = run_neutron_population_event_driven_material(
        starting_neutron=starting_neutron,
        material=_WORKER_MATERIAL,
        max_events=settings["max_events_per_source"],
        max_neutrons=settings["max_neutrons_per_source"],
        kernel=_WORKER_KERNEL,
    )

    for neutron in source_neutrons:
        for row in neutron.history:
            row["source_id"] = int(source_id)

    return source_neutrons


def load_material_summary(material_file, xs_dir):
    """Load a material and return both the material dictionary and display table."""
    material = material_from_composition_file(
        material_file=material_file,
        xs_dir=xs_dir,
    )

    print("Material:", material["name"])
    print("Density:", material["density_g_cm3"], "g/cm^3")
    print("Loaded isotope count:", len(material["isotopes"]))
    print("Missing XS files:", material.get("missing_xs_files", []))

    rows = []
    for isotope, info in material["isotopes"].items():
        rows.append({
            "isotope": isotope,
            "element": info.get("element"),
            "number_density_m3": info["number_density_m3"],
            "mass_fraction": info.get("mass_fraction", np.nan),
        })

    material_df = pd.DataFrame(rows)
    if len(material_df) > 0:
        material_df = material_df.sort_values("number_density_m3", ascending=False)

    return material, material_df


def reaction_preview_dataframe(material, energy_eV, max_rows=40):
    """Build and display a concise reaction-probability preview."""
    open_reactions = build_material_reaction_list_for_energy(
        material=material,
        energy_eV=energy_eV,
    )

    reaction_preview = []
    for row in open_reactions[:max_rows]:
        reaction_preview.append({
            "target_isotope": row["target_isotope"],
            "mt": row["mt"],
            "reaction": mt_name(row["mt"]),
            "sigma_barns": row["sigma_barns"],
            "Sigma_1_per_m": row["macro_xs_1_per_m"],
            "probability": row["probability"],
        })

    reaction_preview_df = pd.DataFrame(reaction_preview)
    Sigma_total = total_macroscopic_xs(open_reactions)

    print("Total Sigma at initial energy =", Sigma_total, "1/m")
    if Sigma_total > 0:
        print("Mean free path =", 1.0 / Sigma_total, "m")
        print("Mean free path =", 100.0 / Sigma_total, "cm")

    return reaction_preview_df, open_reactions, Sigma_total


def run_single_source_transport(
    material,
    initial_energy_eV,
    box_size_m,
    start_x=0.0,
    start_y=0.0,
    start_direction=None,
    max_events=500,
    max_neutrons=50,
    random_seed=None,
    reset_cache=True,
    kernel=None,
):
    """Run one source neutron and return neutrons, history DataFrame, and execution time."""
    set_random_seed(random_seed)

    if reset_cache and "_reaction_cache" in material:
        clear_material_reaction_cache(material)

    if start_direction is None:
        start_direction = random_unit_vector_2d()

    start_time = time.perf_counter()

    starting_neutron = SimpleNeutron(
        energy_eV=initial_energy_eV,
        x=start_x,
        y=start_y,
        direction=start_direction,
        box_size_m=box_size_m,
        idx=0,
        generation=0,
        parent_idx=None,
    )

    neutrons = run_neutron_population_event_driven_material(
        starting_neutron=starting_neutron,
        material=material,
        max_events=max_events,
        max_neutrons=max_neutrons,
        kernel=kernel,
    )

    hist_df = all_neutron_histories_dataframe(neutrons)
    execution_time = time.perf_counter() - start_time

    print("Total neutron objects:", len(neutrons))
    print("Alive at end:", sum(neutron.alive for neutron in neutrons))
    print(f"MC transport took {execution_time:.6f} seconds to complete.")

    return neutrons, hist_df, execution_time


def run_many_source_neutrons(
    material,
    n_source_neutrons,
    initial_energy_eV,
    box_size_m,
    start_x=0.0,
    start_y=0.0,
    max_events_per_source=500,
    max_neutrons_per_source=500,
    kernel=None,
    n_workers=1,
    source_seeds=None,
):
    """Run many independent source-neutron histories."""
    n_source = int(n_source_neutrons)
    workers = resolve_worker_count(n_workers, n_source)
    all_neutrons = []

    if workers <= 1:
        next_starting_idx = 0

        for source_id in range(n_source):
            starting_neutron = SimpleNeutron(
                energy_eV=initial_energy_eV,
                x=start_x,
                y=start_y,
                direction=random_unit_vector_2d(),
                box_size_m=box_size_m,
                idx=next_starting_idx,
                generation=0,
                parent_idx=None,
            )

            source_neutrons = run_neutron_population_event_driven_material(
                starting_neutron=starting_neutron,
                material=material,
                max_events=max_events_per_source,
                max_neutrons=max_neutrons_per_source,
                kernel=kernel,
            )

            for neutron in source_neutrons:
                for row in neutron.history:
                    row["source_id"] = source_id

            all_neutrons.extend(source_neutrons)
            next_starting_idx = max(neutron.idx for neutron in all_neutrons) + 1

        return all_neutrons

    if source_seeds is None:
        source_seeds = make_source_seeds(n_source, random_seed=None)

    id_stride = max(int(max_neutrons_per_source), 1)
    tasks = [
        (source_id, source_seeds[source_id], source_id * id_stride)
        for source_id in range(n_source)
    ]
    settings = {
        "initial_energy_eV": float(initial_energy_eV),
        "box_size_m": float(box_size_m),
        "start_x": float(start_x),
        "start_y": float(start_y),
        "max_events_per_source": int(max_events_per_source),
        "max_neutrons_per_source": int(max_neutrons_per_source),
    }
    chunksize = max(1, n_source // max(workers * 8, 1))
    kernel_payload = kernel_payload_for_workers(kernel)

    print(f"Running {n_source} source neutrons with {workers} worker processes.")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=many_source_worker_init,
        initargs=(material, kernel_payload, settings),
    ) as executor:
        for source_neutrons in executor.map(run_one_source_worker, tasks, chunksize=chunksize):
            all_neutrons.extend(source_neutrons)

    return all_neutrons


def run_many_source_transport(
    material,
    n_source_neutrons,
    initial_energy_eV,
    box_size_m,
    start_x=0.0,
    start_y=0.0,
    max_events_per_source=500,
    max_neutrons_per_source=500,
    random_seed=None,
    reset_cache=True,
    kernel=None,
    n_workers=1,
):
    """Run many source neutrons and return neutrons, history DataFrame, and execution time."""
    set_random_seed(random_seed)

    if reset_cache and "_reaction_cache" in material:
        clear_material_reaction_cache(material)

    n_source = int(n_source_neutrons)
    workers = resolve_worker_count(n_workers, n_source)
    source_seeds = make_source_seeds(n_source, random_seed) if workers > 1 else None

    start_time = time.perf_counter()

    neutrons = run_many_source_neutrons(
        material=material,
        n_source_neutrons=n_source_neutrons,
        initial_energy_eV=initial_energy_eV,
        box_size_m=box_size_m,
        start_x=start_x,
        start_y=start_y,
        max_events_per_source=max_events_per_source,
        max_neutrons_per_source=max_neutrons_per_source,
        kernel=kernel,
        n_workers=workers,
        source_seeds=source_seeds,
    )

    hist_df = all_neutron_histories_dataframe(neutrons)
    execution_time = time.perf_counter() - start_time

    print("Source neutrons launched:", n_source_neutrons)
    print("Worker processes:", workers)
    print("Total neutron objects including secondaries:", len(neutrons))
    print("Alive at end:", sum(neutron.alive for neutron in neutrons))
    if "source_id" in hist_df.columns:
        print("Source histories represented:", hist_df["source_id"].nunique())
    print(f"MC transport took {execution_time:.6f} seconds to complete.")

    return neutrons, hist_df, execution_time


def show_transport_diagnostics(hist_df, material=None, include_source_summary=False):
    """Display event, isotope, MT, energy-update, secondary, and time-order diagnostics."""
    print("Event counts:")
    display(hist_df["event"].value_counts(dropna=False))

    print("Target isotope counts:")
    if "target_isotope" in hist_df.columns:
        display(hist_df["target_isotope"].value_counts(dropna=False))

    print("MT counts:")
    if "mt" in hist_df.columns:
        display(hist_df["mt"].value_counts(dropna=False))

    print("Energy update sources:")
    if "energy_update_source" in hist_df.columns:
        display(hist_df["energy_update_source"].value_counts(dropna=False))

    if include_source_summary and "source_id" in hist_df.columns:
        print("Events per source neutron:")
        events_per_source = hist_df.groupby("source_id").size().reset_index(name="history_rows")
        display(events_per_source.describe())
        display(events_per_source.head(20))

    if material is not None:
        print("Cache summary:")
        material_reaction_cache_summary(material)

    secondary_creation_diagnostics(hist_df)

    time_ordered_events = check_event_time_ordering(hist_df)
    if time_ordered_events is not None:
        display(time_ordered_events.head(20))

    return time_ordered_events


def plot_neutron_trajectories(
    neutrons,
    box_size_m,
    max_paths=None,
    save_name="neutron_traj.png",
    show_labels=True,
):
    """Plot neutron trajectories in the x-y plane and save in the current folder."""
    if max_paths is None:
        selected_neutrons = list(neutrons)
    else:
        selected_neutrons = list(neutrons)[: int(max_paths)]

    fig, ax = plt.subplots(figsize=(7, 7))

    for neutron in selected_neutrons:
        df = neutron_history_dataframe(neutron)
        if len(df) == 0:
            continue

        label = f"n{neutron.idx}, gen {neutron.generation}" if show_labels else None
        ax.plot(
            df["x"],
            df["y"],
            marker="o",
            linewidth=1.0 if max_paths is None else 0.8,
            markersize=3 if max_paths is None else 2,
            alpha=1.0 if max_paths is None else 0.7,
            label=label,
        )

    half = box_size_m / 2.0
    box_x = [-half, half, half, -half, -half]
    box_y = [-half, -half, half, half, -half]
    ax.plot(box_x, box_y, linestyle="--", label="box")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if max_paths is None:
        ax.set_title("Event-driven neutron population trajectories")
    else:
        ax.set_title(f"Neutron trajectories, first {min(max_paths, len(neutrons))} neutron objects")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()

    if save_name is not None:
        fig.savefig(save_name, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax


def make_auto_activity_time_grid(
    hist_df,
    decay_dir,
    points_per_decade=80,
    min_linear_points=400,
    half_lives_to_show=8.0,
    min_activity_half_life_s=1.0e-12,
):
    """Build an activity time grid from production times and radioactive half-lives."""
    activation_df = activation_product_dataframe(hist_df)

    if len(activation_df) == 0:
        print("No activation products found. Using a simple 0 to 1 s grid.")
        return np.linspace(0.0, 1.0, 200)

    activation_df = add_decay_status_to_activation_products(activation_df, decay_dir)

    production_times = pd.to_numeric(activation_df["t"], errors="coerce").dropna().values.astype(float)
    production_times = production_times[np.isfinite(production_times)]

    if len(production_times) == 0:
        production_times = np.array([0.0])

    t_first = float(np.min(production_times))
    t_last = float(np.max(production_times))
    transport_span = max(t_last - t_first, t_last, 1.0e-12)

    radioactive_df = activation_df[activation_df["is_radioactive"] == True].copy()
    half_lives = pd.to_numeric(radioactive_df["half_life_s"], errors="coerce").dropna().values.astype(float)
    half_lives = half_lives[np.isfinite(half_lives) & (half_lives > 0.0)]
    if min_activity_half_life_s is not None and min_activity_half_life_s > 0.0:
        half_lives = half_lives[half_lives >= float(min_activity_half_life_s)]

    if len(half_lives) == 0:
        t_end = max(10.0 * transport_span, 1.0)
        shortest_half_life = None
        longest_half_life = None
    else:
        shortest_half_life = float(np.min(half_lives))
        longest_half_life = float(np.max(half_lives))
        t_end = max(
            t_last + half_lives_to_show * longest_half_life,
            t_last + 20.0 * shortest_half_life,
            10.0 * transport_span,
            1.0e-6,
        )

    early_end = max(10.0 * transport_span, 1.0e-6)
    if shortest_half_life is not None:
        early_end = max(early_end, t_last + 0.05 * shortest_half_life)
    early_end = min(early_end, t_end)

    linear_grid = np.linspace(0.0, early_end, int(min_linear_points))

    positive_times = production_times[production_times > 0.0]
    if len(positive_times) > 0:
        log_start = max(float(np.min(positive_times)), 1.0e-15)
    else:
        log_start = 1.0e-15

    log_start = min(log_start, max(early_end * 1.0e-6, 1.0e-15))

    if t_end > log_start:
        decades = np.log10(t_end) - np.log10(log_start)
        n_log = max(200, int(points_per_decade * max(decades, 1.0)))
        log_grid = np.logspace(np.log10(log_start), np.log10(t_end), n_log)
    else:
        log_grid = np.array([])

    time_grid_s = np.unique(np.concatenate([
        np.array([0.0]),
        production_times,
        linear_grid,
        log_grid,
    ]))
    time_grid_s = time_grid_s[np.isfinite(time_grid_s)]
    time_grid_s.sort()

    print("Auto activity time grid")
    print("  activation events:", len(activation_df))
    print("  radioactive activation events:", len(radioactive_df))
    print("  first production time [s]:", t_first)
    print("  last production time  [s]:", t_last)
    if shortest_half_life is not None:
        print("  shortest radioactive half-life [s]:", shortest_half_life)
        print("  longest radioactive half-life  [s]:", longest_half_life)
    print("  final grid time [s]:", float(time_grid_s[-1]))
    print("  grid points:", len(time_grid_s))

    return time_grid_s


def compute_activity_outputs(
    hist_df,
    decay_dir,
    points_per_decade=80,
    min_linear_points=400,
    half_lives_to_show=8.0,
    max_chain_generations=20,
    min_activity_half_life_s=1.0e-12,
):
    """Compute direct-product and daughter-chain activity outputs."""
    time_grid_s = make_auto_activity_time_grid(
        hist_df=hist_df,
        decay_dir=decay_dir,
        points_per_decade=points_per_decade,
        min_linear_points=min_linear_points,
        half_lives_to_show=half_lives_to_show,
        min_activity_half_life_s=min_activity_half_life_s,
    )

    direct_activity_df, activation_df_with_decay = direct_activity_curves_from_history(
        hist_df=hist_df,
        decay_dir=decay_dir,
        time_grid_s=time_grid_s,
        min_report_half_life_s=min_activity_half_life_s,
    )

    chain_activity_df, production_events = evolve_decay_chains_from_history(
        hist_df=hist_df,
        decay_dir=decay_dir,
        time_grid_s=time_grid_s,
        max_chain_generations=max_chain_generations,
        min_report_half_life_s=min_activity_half_life_s,
    )

    print("Production events:", len(production_events))
    display(direct_activity_df.head())
    display(chain_activity_df.head())

    return time_grid_s, direct_activity_df, chain_activity_df, production_events, activation_df_with_decay


def activity_plot_rows(activity_df, y_col):
    """Return only finite, positive activity rows so plotting starts when activity exists."""
    plot_df = activity_df[["time_s", y_col]].copy()
    plot_df["time_s"] = pd.to_numeric(plot_df["time_s"], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[(plot_df["time_s"] > 0.0) & (plot_df[y_col] > 0.0)]
    return plot_df


def apply_auto_activity_axes(ax, x_values, y_values):
    """Let the data choose log/linear axes and add extra headroom above the curve."""
    x_values = np.array(x_values, dtype=float)
    y_values = np.array(y_values, dtype=float)
    x_values = x_values[np.isfinite(x_values) & (x_values > 0.0)]
    y_values = y_values[np.isfinite(y_values) & (y_values > 0.0)]

    use_log_x = len(x_values) > 0 and np.max(x_values) / np.min(x_values) > 1.0e3
    use_log_y = len(y_values) > 0 and np.max(y_values) / np.min(y_values) > 1.0e3

    if use_log_x:
        ax.set_xscale("log")
    if use_log_y:
        ax.set_yscale("log")

    ax.autoscale(enable=True, axis="both", tight=False)
    ax.margins(x=0.02)

    if len(y_values) == 0:
        return

    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    current_bottom, _ = ax.get_ylim()

    if use_log_y:
        if y_min > 0.0 and y_max > 0.0:
            log_span = max(np.log10(y_max) - np.log10(y_min), 1.0)
            top_padding_decades = max(0.20, 0.06 * log_span)
            padded_top = y_max * 10.0**top_padding_decades
            ax.set_ylim(current_bottom, padded_top)
    else:
        y_span = y_max - y_min
        if y_span <= 0.0:
            y_span = abs(y_max) if y_max != 0.0 else 1.0
        padded_top = y_max + 0.15 * y_span
        ax.set_ylim(current_bottom, padded_top)


def plot_direct_vs_chain_activity_auto(direct_df, chain_df):
    """Plot direct and chain activity without a forced x-axis window."""
    fig, ax = plt.subplots(figsize=(7, 5))

    all_x = []
    all_y = []
    plotted = False

    for df, label in [
        (direct_df, "direct products only"),
        (chain_df, "including daughters"),
    ]:
        plot_df = activity_plot_rows(df, "total_Bq")
        if len(plot_df) == 0:
            continue
        ax.plot(plot_df["time_s"], plot_df["total_Bq"], label=label)
        all_x.extend(plot_df["time_s"].values)
        all_y.extend(plot_df["total_Bq"].values)
        plotted = True

    if not plotted:
        print("No positive activity values to plot.")
        return None, None

    apply_auto_activity_axes(ax, all_x, all_y)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("total activity [Bq]")
    ax.set_title("Direct vs daughter-chain activity")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    plt.show()
    return fig, ax


def plot_total_activity_auto(activity_df, y_col="total_Ci", ylabel="total activity [Ci]", save_name=None):
    """Plot total activity without manually setting x-limits or a fixed time window."""
    plot_df = activity_plot_rows(activity_df, y_col)

    if len(plot_df) == 0:
        print("No positive activity values to plot.")
        return None, None

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(plot_df["time_s"], plot_df[y_col])
    apply_auto_activity_axes(ax, plot_df["time_s"].values, plot_df[y_col].values)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title("Total activity")
    ax.grid(True, alpha=0.3, which="both")

    if save_name is not None:
        fig.savefig(save_name, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax


def plot_isotope_activities_auto(activity_df, unit="Bq", save_name=None):
    """Plot isotope activities with automatic axis scaling."""
    suffix = "_" + unit
    activity_cols = [col for col in activity_df.columns if col.endswith(suffix) and col != "total" + suffix]

    if len(activity_cols) == 0:
        print("No isotope activity columns found.")
        return None, None

    fig, ax = plt.subplots(figsize=(7, 5))
    all_x = []
    all_y = []
    plotted = False

    for col in activity_cols:
        plot_df = activity_plot_rows(activity_df, col)
        if len(plot_df) == 0:
            continue
        isotope = col.replace(suffix, "")
        ax.plot(plot_df["time_s"], plot_df[col], label=isotope)
        all_x.extend(plot_df["time_s"].values)
        all_y.extend(plot_df[col].values)
        plotted = True

    if not plotted:
        print("No positive isotope activity values to plot.")
        return None, None

    apply_auto_activity_axes(ax, all_x, all_y)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("activity [" + unit + "]")
    ax.set_title("Activity by isotope")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    if save_name is not None:
        fig.savefig(save_name, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax
