"""Activation inventory, decay status, direct activity, and daughter-chain activity."""

import numpy as np
import pandas as pd

from nuclear_data import (
    BQ_TO_CI,
    find_decay_file,
    isotope_label_to_decay_filename_candidates,
)

_SCIPY_EXPM = None
_SCIPY_EXPM_IMPORT_TRIED = False
DEFAULT_MIN_ACTIVITY_HALF_LIFE_S = 1.0e-12


def residual_product_dataframe(hist_df):
    """Extract residual product records from the neutron history DataFrame."""
    if "residual_product" not in hist_df.columns:
        return pd.DataFrame()
    product_rows = hist_df[hist_df["residual_product"].notna()].copy()
    cols = [
        "neutron_id", "generation", "parent_id", "t", "x", "y",
        "target_isotope", "mt", "reaction_name", "residual_product",
        "residual_product_Z", "residual_product_A", "product_note",
        "incoming_energy_eV", "outgoing_energy_eV", "energy_update_source",
    ]
    existing_cols = [col for col in cols if col in product_rows.columns]
    return product_rows[existing_cols]


def summarize_residual_products(hist_df):
    """Print/count residual products created during transport."""
    products_df = residual_product_dataframe(hist_df)
    if len(products_df) == 0:
        print("No residual products recorded.")
        return products_df
    print("Residual product counts:")
    print(products_df["residual_product"].value_counts())
    print("\nProducts by MT:")
    print(products_df.groupby(["mt", "reaction_name", "residual_product"]).size())
    return products_df


def activation_product_dataframe(hist_df):
    """Extract actual activation products, excluding elastic/inelastic same-isotope rows."""
    if "residual_product" not in hist_df.columns:
        return pd.DataFrame()
    product_rows = hist_df[hist_df["residual_product"].notna()].copy()
    if "product_note" in product_rows.columns:
        product_rows = product_rows[product_rows["product_note"] == "residual product"].copy()
    cols = [
        "neutron_id", "generation", "parent_id", "t", "x", "y",
        "target_isotope", "mt", "reaction_name", "residual_product",
        "residual_product_Z", "residual_product_A", "incoming_energy_eV",
        "outgoing_energy_eV", "energy_update_source",
    ]
    existing_cols = [col for col in cols if col in product_rows.columns]
    return product_rows[existing_cols]


def read_decay_summary(decay_dir, isotope_label):
    """Read important decay information for one isotope."""
    path, filename = find_decay_file(decay_dir, isotope_label)

    if path is None:
        candidates = isotope_label_to_decay_filename_candidates(isotope_label)
        return {
            "decay_filename": candidates[0] if len(candidates) > 0 else None,
            "has_decay_file": False,
            "half_life_s": None,
            "is_stable": None,
            "is_radioactive": None,
            "decay_daughters": [],
            "branching_ratios": [],
            "qvalues_eV": [],
            "decay_energy_eV": None,
            "decay_note": "no decay file found",
        }

    decay_data = np.load(path, allow_pickle=True)
    half_life_s = float(decay_data["half_life_s"])

    if np.isinf(half_life_s):
        is_stable = True
        is_radioactive = False
        decay_note = "stable isotope"
    else:
        is_stable = False
        is_radioactive = True
        decay_note = "radioactive isotope"

    daughters = [str(x) for x in decay_data["modes_daughter"]] if "modes_daughter" in decay_data.files else []
    branching_ratios = [float(x) for x in decay_data["modes_br"]] if "modes_br" in decay_data.files else []
    qvalues_eV = [float(x) for x in decay_data["modes_qvalue_eV"]] if "modes_qvalue_eV" in decay_data.files else []
    decay_energy_eV = float(decay_data["decay_energy_eV"]) if "decay_energy_eV" in decay_data.files else None

    return {
        "decay_filename": filename,
        "has_decay_file": True,
        "half_life_s": half_life_s,
        "is_stable": is_stable,
        "is_radioactive": is_radioactive,
        "decay_daughters": daughters,
        "branching_ratios": branching_ratios,
        "qvalues_eV": qvalues_eV,
        "decay_energy_eV": decay_energy_eV,
        "decay_note": decay_note,
    }


def add_decay_status_to_activation_products(activation_df, decay_dir):
    """Add explicit decay status information to activation product rows."""
    activation_df = activation_df.copy()

    rows = []
    for _, row in activation_df.iterrows():
        isotope = row["residual_product"]
        summary = read_decay_summary(decay_dir, isotope)
        rows.append(summary)

    if len(rows) == 0:
        return activation_df

    for key in rows[0].keys():
        activation_df[key] = [row[key] for row in rows]

    return activation_df


def summarize_activation_with_decay_status(hist_df, decay_dir):
    """Summarize activation products and decay info."""
    activation_df = activation_product_dataframe(hist_df)
    if len(activation_df) == 0:
        print("No activation products recorded.")
        return pd.DataFrame()

    activation_df = add_decay_status_to_activation_products(activation_df, decay_dir)

    summary = (
        activation_df
        .groupby([
            "residual_product", "decay_filename", "has_decay_file", "is_stable",
            "is_radioactive", "half_life_s", "decay_note",
        ], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    daughter_texts = []
    branch_texts = []
    qvalue_texts = []
    decay_energy_texts = []
    for _, row in summary.iterrows():
        isotope = row["residual_product"]
        s = read_decay_summary(decay_dir, isotope)
        daughter_texts.append(str(s["decay_daughters"]))
        branch_texts.append(str(s["branching_ratios"]))
        qvalue_texts.append(str(s["qvalues_eV"]))
        decay_energy_texts.append(s["decay_energy_eV"])

    summary["decay_daughters"] = daughter_texts
    summary["branching_ratios"] = branch_texts
    summary["qvalues_eV"] = qvalue_texts
    summary["decay_energy_eV"] = decay_energy_texts

    return summary


def activation_inventory_summary(hist_df):
    """Count how many times each residual isotope was produced."""
    activation_df = activation_product_dataframe(hist_df)
    if len(activation_df) == 0:
        print("No activation products recorded.")
        return pd.DataFrame()
    return (
        activation_df.groupby("residual_product")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def activation_inventory_by_reaction(hist_df):
    """Count products by reaction channel."""
    activation_df = activation_product_dataframe(hist_df)
    if len(activation_df) == 0:
        print("No activation products recorded.")
        return pd.DataFrame()
    return (
        activation_df.groupby(["mt", "reaction_name", "residual_product"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def isotope_production_times(hist_df, isotope):
    """Return production times for one residual isotope."""
    activation_df = activation_product_dataframe(hist_df)
    rows = activation_df[activation_df["residual_product"] == isotope].copy()
    if len(rows) == 0:
        return np.array([])
    return rows["t"].values


def activation_source_dictionary(hist_df):
    """Convert activation product records into isotope -> production time list."""
    activation_df = activation_product_dataframe(hist_df)
    source = {}
    for _, row in activation_df.iterrows():
        isotope = row["residual_product"]
        t = float(row["t"])
        source.setdefault(isotope, []).append(t)
    return source


def activity_from_production_times(times_created, half_life_s, time_grid_s):
    """Compute direct activity from nuclei produced at specific times."""
    times_created = np.array(times_created, dtype=float)
    time_grid_s = np.array(time_grid_s, dtype=float)
    activity_Bq = np.zeros(len(time_grid_s))

    if len(times_created) == 0:
        return activity_Bq
    if half_life_s is None or np.isinf(half_life_s) or half_life_s <= 0.0:
        return activity_Bq

    decay_lambda = np.log(2.0) / half_life_s
    for t_created in times_created:
        age = time_grid_s - t_created
        mask = age >= 0.0
        activity_Bq[mask] += decay_lambda * np.exp(-decay_lambda * age[mask])
    return activity_Bq


def decay_constant_from_half_life(half_life_s):
    """Convert half-life in seconds to decay constant in 1/s."""
    if half_life_s is None:
        return 0.0
    try:
        half_life_s = float(half_life_s)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(half_life_s) or half_life_s <= 0.0:
        return 0.0
    return float(np.log(2.0) / half_life_s)


def direct_activity_curves_from_history(
    hist_df,
    decay_dir,
    time_grid_s,
    min_report_half_life_s=DEFAULT_MIN_ACTIVITY_HALF_LIFE_S,
):
    """Compute direct activity curves from activation products, excluding daughters."""
    activation_df = activation_product_dataframe(hist_df)
    activity_df = pd.DataFrame({"time_s": time_grid_s})
    total_Bq = np.zeros(len(time_grid_s))

    if len(activation_df) == 0:
        print("No activation products recorded.")
        activity_df["total_Bq"] = total_Bq
        activity_df["total_Ci"] = total_Bq * BQ_TO_CI
        return activity_df, activation_df

    activation_df = add_decay_status_to_activation_products(activation_df, decay_dir)
    radioactive_df = activation_df[activation_df["is_radioactive"] == True].copy()
    if min_report_half_life_s is not None and min_report_half_life_s > 0.0:
        half_lives = pd.to_numeric(radioactive_df["half_life_s"], errors="coerce")
        radioactive_df = radioactive_df[half_lives >= float(min_report_half_life_s)].copy()

    if len(radioactive_df) == 0:
        print("No radioactive activation products recorded.")
        activity_df["total_Bq"] = total_Bq
        activity_df["total_Ci"] = total_Bq * BQ_TO_CI
        return activity_df, activation_df

    for isotope in sorted(radioactive_df["residual_product"].unique()):
        rows = radioactive_df[radioactive_df["residual_product"] == isotope]
        times_created = rows["t"].values
        half_life_s = float(rows["half_life_s"].iloc[0])
        activity_Bq = activity_from_production_times(times_created, half_life_s, time_grid_s)
        activity_df[isotope + "_Bq"] = activity_Bq
        total_Bq += activity_Bq

    activity_df["total_Bq"] = total_Bq
    activity_df["total_Ci"] = total_Bq * BQ_TO_CI
    return activity_df, activation_df


def get_decay_info(decay_dir, isotope_label):
    """Return simple decay information for one isotope."""
    summary = read_decay_summary(decay_dir, isotope_label)
    return {
        "isotope": isotope_label,
        "has_decay_file": summary["has_decay_file"],
        "half_life_s": summary["half_life_s"],
        "is_stable": summary["is_stable"],
        "is_radioactive": summary["is_radioactive"],
        "daughters": summary["decay_daughters"],
        "branching_ratios": summary["branching_ratios"],
        "decay_note": summary["decay_note"],
    }


def _valid_daughter_label(daughter):
    daughter = str(daughter).strip()
    if daughter == "":
        return None
    if daughter.lower() in {"none", "nan"}:
        return None
    return daughter


def _daughter_branch_pairs(info):
    daughters = list(info.get("daughters", []))
    branching_ratios = list(info.get("branching_ratios", []))

    if len(daughters) == 1 and len(branching_ratios) == 0:
        branching_ratios = [1.0]

    pairs = []
    for daughter, branching_ratio in zip(daughters, branching_ratios):
        daughter = _valid_daughter_label(daughter)
        if daughter is None:
            continue
        try:
            branching_ratio = float(branching_ratio)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(branching_ratio) or branching_ratio <= 0.0:
            continue
        pairs.append((daughter, branching_ratio))

    branch_sum = sum(branching_ratio for _, branching_ratio in pairs)
    if branch_sum > 1.0 and branch_sum <= 1.0 + 1.0e-6:
        pairs = [(daughter, branching_ratio / branch_sum) for daughter, branching_ratio in pairs]

    return pairs


def _is_collapsed_half_life(half_life_s, min_report_half_life_s):
    if min_report_half_life_s is None or min_report_half_life_s <= 0.0:
        return False
    try:
        half_life_s = float(half_life_s)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(half_life_s) or half_life_s <= 0.0:
        return False
    return half_life_s < float(min_report_half_life_s)


def resolve_observable_decay_products(
    isotope,
    amount,
    decay_dir,
    decay_info_cache,
    min_report_half_life_s=DEFAULT_MIN_ACTIVITY_HALF_LIFE_S,
    max_collapse_generations=20,
):
    """Collapse very short-lived nuclides into longer-lived decay products."""
    isotope = str(isotope)
    amount = float(amount)
    max_collapse_generations = max(0, int(max_collapse_generations))

    def info_for(label):
        if label not in decay_info_cache:
            decay_info_cache[label] = get_decay_info(decay_dir, label)
        return decay_info_cache[label]

    products = []
    stack = [(isotope, amount, 0)]
    while len(stack) > 0:
        current, current_amount, depth = stack.pop()
        info = info_for(current)
        if (
            not _is_collapsed_half_life(info["half_life_s"], min_report_half_life_s)
            or depth >= max_collapse_generations
        ):
            products.append((current, current_amount))
            continue

        pairs = _daughter_branch_pairs(info)
        if len(pairs) == 0:
            continue

        for daughter, branching_ratio in reversed(pairs):
            stack.append((daughter, current_amount * float(branching_ratio), depth + 1))

    merged = {}
    for product, product_amount in products:
        merged[product] = merged.get(product, 0.0) + product_amount
    return sorted(merged.items())


def collect_decay_network(
    initial_isotopes,
    decay_dir,
    max_chain_generations=20,
    min_report_half_life_s=DEFAULT_MIN_ACTIVITY_HALF_LIFE_S,
):
    """Collect all observable isotopes reachable through decay daughters."""
    max_chain_generations = max(0, int(max_chain_generations))
    initial_isotopes = sorted(set(str(isotope) for isotope in initial_isotopes))

    isotope_order = []
    decay_info_cache = {}
    queue = []
    shallowest_depth = {}

    for isotope in initial_isotopes:
        for product, _ in resolve_observable_decay_products(
            isotope=isotope,
            amount=1.0,
            decay_dir=decay_dir,
            decay_info_cache=decay_info_cache,
            min_report_half_life_s=min_report_half_life_s,
            max_collapse_generations=max_chain_generations,
        ):
            queue.append((product, 0))

    def info_for(isotope):
        if isotope not in decay_info_cache:
            decay_info_cache[isotope] = get_decay_info(decay_dir, isotope)
        return decay_info_cache[isotope]

    while len(queue) > 0:
        isotope, depth = queue.pop(0)
        previous_depth = shallowest_depth.get(isotope)
        if previous_depth is not None and previous_depth <= depth:
            continue

        shallowest_depth[isotope] = depth
        if isotope not in isotope_order:
            isotope_order.append(isotope)

        info = info_for(isotope)
        decay_lambda = decay_constant_from_half_life(info["half_life_s"])
        if decay_lambda <= 0.0 or depth >= max_chain_generations:
            continue

        for daughter, branching_ratio in _daughter_branch_pairs(info):
            for product, product_amount in resolve_observable_decay_products(
                isotope=daughter,
                amount=branching_ratio,
                decay_dir=decay_dir,
                decay_info_cache=decay_info_cache,
                min_report_half_life_s=min_report_half_life_s,
                max_collapse_generations=max_chain_generations - depth - 1,
            ):
                if product_amount > 0.0:
                    queue.append((product, depth + 1))

    return sorted(isotope_order), decay_info_cache


def build_decay_matrix(
    isotopes,
    decay_dir,
    decay_info_cache=None,
    min_report_half_life_s=DEFAULT_MIN_ACTIVITY_HALF_LIFE_S,
    max_collapse_generations=20,
):
    """Build dN/dt = A N for an observable decay network."""
    if decay_info_cache is None:
        decay_info_cache = {}

    isotope_index = {isotope: i for i, isotope in enumerate(isotopes)}
    matrix = np.zeros((len(isotopes), len(isotopes)), dtype=float)

    def info_for(isotope):
        if isotope not in decay_info_cache:
            decay_info_cache[isotope] = get_decay_info(decay_dir, isotope)
        return decay_info_cache[isotope]

    for parent in isotopes:
        parent_i = isotope_index[parent]
        info = info_for(parent)
        decay_lambda = decay_constant_from_half_life(info["half_life_s"])
        if decay_lambda <= 0.0:
            continue

        matrix[parent_i, parent_i] -= decay_lambda
        for daughter, branching_ratio in _daughter_branch_pairs(info):
            products = resolve_observable_decay_products(
                isotope=daughter,
                amount=branching_ratio,
                decay_dir=decay_dir,
                decay_info_cache=decay_info_cache,
                min_report_half_life_s=min_report_half_life_s,
                max_collapse_generations=max_collapse_generations,
            )
            for product, product_amount in products:
                if product not in isotope_index:
                    continue
                daughter_i = isotope_index[product]
                matrix[daughter_i, parent_i] += decay_lambda * product_amount

    return matrix, decay_info_cache


def _get_scipy_expm():
    global _SCIPY_EXPM
    global _SCIPY_EXPM_IMPORT_TRIED

    if not _SCIPY_EXPM_IMPORT_TRIED:
        _SCIPY_EXPM_IMPORT_TRIED = True
        try:
            from scipy.linalg import expm
            _SCIPY_EXPM = expm
        except ImportError:
            _SCIPY_EXPM = None

    return _SCIPY_EXPM


def _matrix_exponential_pade13(matrix):
    """Fallback matrix exponential using scaling and squaring with Pade(13)."""
    n = matrix.shape[0]
    identity = np.eye(n)
    matrix_norm = np.linalg.norm(matrix, 1)
    if matrix_norm == 0.0:
        return identity

    theta_13 = 5.371920351148152
    if matrix_norm > theta_13:
        scale_power = int(np.ceil(np.log2(matrix_norm / theta_13)))
    else:
        scale_power = 0

    matrix = matrix / (2.0 ** scale_power)

    b = [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ]

    matrix_2 = matrix @ matrix
    matrix_4 = matrix_2 @ matrix_2
    matrix_6 = matrix_4 @ matrix_2

    u = matrix @ (
        matrix_6 @ (b[13] * matrix_6 + b[11] * matrix_4 + b[9] * matrix_2)
        + b[7] * matrix_6
        + b[5] * matrix_4
        + b[3] * matrix_2
        + b[1] * identity
    )
    v = (
        matrix_6 @ (b[12] * matrix_6 + b[10] * matrix_4 + b[8] * matrix_2)
        + b[6] * matrix_6
        + b[4] * matrix_4
        + b[2] * matrix_2
        + b[0] * identity
    )

    p = v + u
    q = v - u
    result = np.linalg.solve(q, p)

    for _ in range(scale_power):
        result = result @ result

    return result


def matrix_exponential(matrix, dt):
    """Return exp(matrix * dt), using SciPy when available."""
    dt = float(dt)
    n = matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    if dt == 0.0:
        return np.eye(n)

    scaled_matrix = matrix * dt
    scipy_expm = _get_scipy_expm()
    if scipy_expm is not None:
        return scipy_expm(scaled_matrix)
    return _matrix_exponential_pade13(scaled_matrix)


def _clean_inventory_roundoff(inventory):
    if len(inventory) == 0:
        return inventory
    scale = max(1.0, float(np.max(np.abs(inventory))))
    tolerance = 1.0e-12 * scale
    small_negative = (inventory < 0.0) & (inventory > -tolerance)
    inventory[small_negative] = 0.0
    tiny = np.abs(inventory) < 1.0e-30
    inventory[tiny] = 0.0
    return inventory


def _activity_row(t, isotopes, inventory, decay_info_cache):
    row = {"time_s": float(t)}
    total_Bq = 0.0

    for isotope, amount in zip(isotopes, inventory):
        amount = float(amount)
        decay_lambda = decay_constant_from_half_life(decay_info_cache[isotope]["half_life_s"])
        activity_Bq = decay_lambda * amount if decay_lambda > 0.0 else 0.0
        row[isotope + "_N"] = amount
        row[isotope + "_Bq"] = activity_Bq
        total_Bq += activity_Bq

    row["total_Bq"] = total_Bq
    row["total_Ci"] = total_Bq * BQ_TO_CI
    return row


def production_events_from_history(hist_df, decay_dir):
    """Convert activation product rows into production events."""
    activation_df = activation_product_dataframe(hist_df)
    if len(activation_df) == 0:
        return []
    activation_df = add_decay_status_to_activation_products(activation_df, decay_dir)

    events = []
    for _, row in activation_df.iterrows():
        events.append({"time_s": float(row["t"]), "isotope": row["residual_product"], "amount": 1.0})
    events.sort(key=lambda row: row["time_s"])
    return events


def evolve_decay_chains_from_history(
    hist_df,
    decay_dir,
    time_grid_s,
    max_chain_generations=20,
    min_report_half_life_s=DEFAULT_MIN_ACTIVITY_HALF_LIFE_S,
):
    """Evolve activation products and daughter chains on a time grid.

    The decay network is solved as dN/dt = A N with exact matrix-exponential
    transitions between production/output times. Production events are treated
    as impulse sources added at their event time.
    """
    input_time_grid = np.array(time_grid_s, dtype=float)
    input_time_grid = input_time_grid[np.isfinite(input_time_grid)]
    production_events = []

    for event in production_events_from_history(hist_df, decay_dir):
        try:
            time_s = float(event["time_s"])
            amount = float(event.get("amount", 1.0))
        except (TypeError, ValueError):
            continue
        isotope = str(event.get("isotope", "")).strip()
        if isotope == "" or not np.isfinite(time_s) or not np.isfinite(amount):
            continue
        if amount == 0.0:
            continue
        production_events.append({
            "time_s": time_s,
            "isotope": isotope,
            "amount": amount,
        })

    production_events.sort(key=lambda row: row["time_s"])

    if len(production_events) == 0:
        print("No activation products to evolve.")
        decay_df = pd.DataFrame({
            "time_s": input_time_grid,
            "total_Bq": np.zeros(len(input_time_grid)),
            "total_Ci": np.zeros(len(input_time_grid)),
        })
        return decay_df, production_events

    decay_info_cache = {}
    observable_events = []
    for event in production_events:
        for isotope, amount in resolve_observable_decay_products(
            isotope=event["isotope"],
            amount=event["amount"],
            decay_dir=decay_dir,
            decay_info_cache=decay_info_cache,
            min_report_half_life_s=min_report_half_life_s,
            max_collapse_generations=max_chain_generations,
        ):
            if amount == 0.0:
                continue
            observable_events.append({
                "time_s": event["time_s"],
                "isotope": isotope,
                "amount": amount,
            })

    production_events = sorted(observable_events, key=lambda row: row["time_s"])
    if len(production_events) == 0:
        print("No activation products remain after short-lived decay collapse.")
        decay_df = pd.DataFrame({
            "time_s": input_time_grid,
            "total_Bq": np.zeros(len(input_time_grid)),
            "total_Ci": np.zeros(len(input_time_grid)),
        })
        return decay_df, production_events

    production_isotopes = [event["isotope"] for event in production_events]
    isotopes, decay_info_cache = collect_decay_network(
        initial_isotopes=production_isotopes,
        decay_dir=decay_dir,
        max_chain_generations=max_chain_generations,
        min_report_half_life_s=min_report_half_life_s,
    )
    decay_matrix, decay_info_cache = build_decay_matrix(
        isotopes=isotopes,
        decay_dir=decay_dir,
        decay_info_cache=decay_info_cache,
        min_report_half_life_s=min_report_half_life_s,
        max_collapse_generations=max_chain_generations,
    )
    isotope_index = {isotope: i for i, isotope in enumerate(isotopes)}

    production_times = np.array([event["time_s"] for event in production_events], dtype=float)
    internal_time_grid = np.unique(np.concatenate([input_time_grid, production_times]))
    internal_time_grid = internal_time_grid[np.isfinite(internal_time_grid)]
    internal_time_grid.sort()

    output_times = set(float(t) for t in np.unique(input_time_grid))
    inventory = np.zeros(len(isotopes), dtype=float)
    transition_cache = {}
    output_rows = []
    production_index = 0
    last_time = None

    for t in internal_time_grid:
        t = float(t)
        if last_time is not None:
            dt = t - last_time
            if dt > 0.0:
                if dt not in transition_cache:
                    transition_cache[dt] = matrix_exponential(decay_matrix, dt)
                inventory = transition_cache[dt] @ inventory
                inventory = _clean_inventory_roundoff(inventory)

        while production_index < len(production_events):
            event = production_events[production_index]
            if event["time_s"] > t:
                break
            inventory[isotope_index[event["isotope"]]] += event["amount"]
            production_index += 1

        if t in output_times:
            output_rows.append(_activity_row(t, isotopes, inventory, decay_info_cache))

        last_time = t

    decay_df = pd.DataFrame(output_rows)
    isotope_cols = []
    for isotope in isotopes:
        isotope_cols.extend([isotope + "_N", isotope + "_Bq"])
    ordered_cols = ["time_s", "total_Bq", "total_Ci"] + isotope_cols
    decay_df = decay_df[ordered_cols]
    return decay_df, production_events


def plot_isotope_activities(activity_df, plt_module):
    """Plot isotope-specific activity curves using matplotlib.pyplot passed in."""
    activity_cols = [col for col in activity_df.columns if col.endswith("_Bq") and col != "total_Bq"]
    if len(activity_cols) == 0:
        print("No isotope activity columns found.")
        return
    plt = plt_module
    plt.figure(figsize=(7, 5))
    for col in activity_cols:
        isotope = col.replace("_Bq", "")
        plt.plot(activity_df["time_s"], activity_df[col], label=isotope)
    plt.xlabel("time [s]")
    plt.ylabel("activity [Bq]")
    plt.title("Activity by isotope")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
