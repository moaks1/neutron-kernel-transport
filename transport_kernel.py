"""
Precomputed material transport kernels for neutron transport.

Current kernel layers:
    - precompute material reaction lists on a fixed log-energy grid
    - precompute total macroscopic cross section per energy bin
    - precompute reaction probabilities and CDF values per energy bin
    - precompute residual/transmutation product rows and MF10 branch CDFs
    - precompute outgoing-neutron multiplicity metadata and integer rules
    - precompute MF4 elastic angular CDFs for MT=2 when available
    - precompute MF6 outgoing-neutron energy/angle kernels for supported
      LAW=1 and LAW=2 data
"""

import hashlib
import json
import re
from pathlib import Path
 
import numpy as np
 
from nuclear_data import (
    build_material_reaction_list_for_energy,
    total_macroscopic_xs,
    get_target_A_from_data,
    count_outgoing_neutrons,
    legendre_pdf_from_coeffs,
    base_residual_product_for_mt,
    mf10_product_rows_at_energy,
)
 
from mf4 import (
    mf4_info,
    mf4_legendre_coeffs_at,
    mf4_tabulated_pdf_at,
    elastic_energy_from_mu_cm,
    elastic_mu_lab_from_mu_cm,
)

from mf6 import (
    mf6_neutron_product_indices,
    mf6_yield_at_product_index,
    mf6_law1_energy_pdf_at,
    mf6_law2_coeffs_at,
    get_target_A_from_threshold_curve,
)


def slugify_kernel_text(value, default="material"):
    """Return a filesystem-safe lowercase label for kernel cache filenames."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or default


def material_kernel_fingerprint(material, digest_chars=10):
    """Build a short stable fingerprint from material density and isotope densities."""
    pieces = [
        str(material.get("name", "")),
        f"density={float(material.get('density_g_cm3', 0.0)):.12e}",
    ]

    isotopes = material.get("isotopes", {})
    for label in sorted(isotopes):
        info = isotopes[label]
        pieces.append(
            f"{label}:"
            f"nd={float(info.get('number_density_m3', 0.0)):.12e}:"
            f"mf={float(info.get('mass_fraction', 0.0)):.12e}"
        )

    digest = hashlib.sha1("|".join(pieces).encode("utf-8")).hexdigest()
    return digest[: int(digest_chars)]


def material_kernel_label(material):
    """Return a readable, material-specific label for kernel cache files."""
    name = slugify_kernel_text(material.get("name", "material"))
    fingerprint = material_kernel_fingerprint(material)
    return f"{name}_{fingerprint}"


def material_transport_kernel_filename(material, version="v1"):
    """Return the default NPZ filename for a material transport kernel."""
    label = material_kernel_label(material)
    version = slugify_kernel_text(version, default="v1")
    return f"material_transport_kernel_{label}_{version}.npz"


def material_transport_kernel_path(kernel_dir, material, version="v1"):
    """Return the default material-specific kernel cache path."""
    return Path(kernel_dir) / material_transport_kernel_filename(material, version=version)


def cdf_from_pdf_grid(x, pdf):
    """
    Build a normalized CDF from a tabulated PDF using trapezoid integration.
    """
    x = np.array(x, dtype=float)
    pdf = np.array(pdf, dtype=float)

    good = np.isfinite(x) & np.isfinite(pdf)
    x = x[good]
    pdf = pdf[good]

    if len(x) <= 1:
        return None, None

    order = np.argsort(x)
    x = x[order]
    pdf = np.maximum(pdf[order], 0.0)

    dx = np.diff(x)
    if np.any(dx <= 0.0):
        return None, None

    area_pieces = 0.5 * (pdf[:-1] + pdf[1:]) * dx
    cdf = np.zeros(len(x))
    cdf[1:] = np.cumsum(area_pieces)

    total_area = cdf[-1]
    if total_area <= 0.0:
        return None, None

    cdf = cdf / total_area
    cdf[-1] = 1.0

    return x, cdf


def build_elastic_mf4_kernel_for_channel(target_data, incoming_energy_eV, n_mu_grid=801):
    """
    Build a precomputed MF4 elastic angular CDF for MT=2.

    This stores only angular sampling information.
    Runtime still computes the final outgoing energy from sampled mu.
    """
    info = mf4_info(target_data, mt=2)
    if info is None:
        return {}

    LCT = int(info["LCT"])
    LI = int(info["LI"])
    LTT = int(info["LTT"])

    mu_grid = np.linspace(-1.0, 1.0, int(n_mu_grid))
    pdf = None
    source = None

    if LI == 1:
        pdf = np.ones(len(mu_grid), dtype=float) * 0.5
        source = "kernel-MF4-isotropic-LI"

    elif LTT in [1, 3] and int(info["legendre_count"]) > 0:
        coeffs = mf4_legendre_coeffs_at(
            data=target_data,
            mt=2,
            energy_eV=incoming_energy_eV,
        )
        if coeffs is not None:
            pdf = legendre_pdf_from_coeffs(mu_grid, coeffs)
            source = "kernel-MF4-Legendre"

    if pdf is None and LTT in [2, 3] and int(info["tabulated_count"]) > 0:
        tab_mu, tab_pdf = mf4_tabulated_pdf_at(
            data=target_data,
            mt=2,
            energy_eV=incoming_energy_eV,
        )
        if tab_mu is not None and tab_pdf is not None:
            pdf = np.interp(mu_grid, tab_mu, tab_pdf, left=0.0, right=0.0)
            source = "kernel-MF4-tabulated"

    if pdf is None:
        return {}

    mu_grid, mu_cdf = cdf_from_pdf_grid(mu_grid, pdf)
    if mu_grid is None or mu_cdf is None:
        return {}

    A = get_target_A_from_data(target_data)

    return {
        "elastic_has_mf4_kernel": True,
        "elastic_mu_grid": mu_grid,
        "elastic_mu_cdf": mu_cdf,
        "elastic_LCT": LCT,
        "elastic_A": float(A),
        "elastic_angle_source": source,
        "elastic_angle_frame": "center-of-mass" if LCT == 2 else "lab",
    }


def sample_elastic_mf4_update_from_kernel(chosen_reaction, incoming_energy_eV):
    """
    Sample elastic scattering from the precomputed MF4 angular kernel.

    Returns the same style of dictionary as sample_elastic_mf4_update(...).
    """
    if not chosen_reaction.get("elastic_has_mf4_kernel", False):
        return None

    mu_grid = chosen_reaction.get("elastic_mu_grid")
    mu_cdf = chosen_reaction.get("elastic_mu_cdf")

    if mu_grid is None or mu_cdf is None:
        return None

    mu_grid = np.array(mu_grid, dtype=float)
    mu_cdf = np.array(mu_cdf, dtype=float)

    if len(mu_grid) <= 1 or len(mu_cdf) != len(mu_grid):
        return None

    R = np.random.random()
    mu = float(np.interp(R, mu_cdf, mu_grid))
    mu = float(np.clip(mu, -1.0, 1.0))

    LCT = int(chosen_reaction.get("elastic_LCT", 0))
    A = float(chosen_reaction.get("elastic_A", 0.0))

    if LCT == 2:
        mu_cm = mu
        outgoing_energy = elastic_energy_from_mu_cm(
            incoming_energy_eV=incoming_energy_eV,
            mu_cm=mu_cm,
            A=A,
        )
        mu_lab = elastic_mu_lab_from_mu_cm(mu_cm, A)
        theta_lab = np.arccos(np.clip(mu_lab, -1.0, 1.0))

        return {
            "energy_eV": outgoing_energy,
            "mu_cm": mu_cm,
            "mu_lab": mu_lab,
            "theta_lab_rad": theta_lab,
            "theta_lab_deg": np.degrees(theta_lab),
            "angle_source": chosen_reaction.get("elastic_angle_source"),
            "angle_frame": chosen_reaction.get("elastic_angle_frame"),
            "energy_update_source": "kernel-MF4-elastic-two-body",
            "elastic_angle_random_number": float(R),
        }

    if LCT == 1:
        mu_lab = mu
        theta_lab = np.arccos(np.clip(mu_lab, -1.0, 1.0))

        return {
            "energy_eV": float(incoming_energy_eV),
            "mu_cm": None,
            "mu_lab": mu_lab,
            "theta_lab_rad": theta_lab,
            "theta_lab_deg": np.degrees(theta_lab),
            "angle_source": chosen_reaction.get("elastic_angle_source"),
            "angle_frame": chosen_reaction.get("elastic_angle_frame"),
            "energy_update_source": "kernel-MF4-lab-angle-energy-unchanged",
            "elastic_angle_random_number": float(R),
        }

    return None
    

def mf6_product_cdf_from_kernel_products(products):
    """
    Add product-level CDF values to usable MF6 neutron-product kernels.

    Positive MF6 yields are used as product-selection weights. If all usable
    products have zero yield at this incident-energy bin, fall back to a uniform
    choice among the usable products so the old runtime behavior still has a
    sensible kernel fallback.
    """
    if len(products) == 0:
        return []

    weights = np.array([max(float(p.get("yield", 0.0)), 0.0) for p in products], dtype=float)
    if np.sum(weights) <= 0.0:
        weights = np.ones(len(products), dtype=float)
        weight_source = "kernel-MF6-product-uniform-zero-yield"
    else:
        weight_source = "kernel-MF6-product-yield-cdf"

    weights = weights / np.sum(weights)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0

    out = []
    for i, product in enumerate(products):
        row = dict(product)
        row["product_probability"] = float(weights[i])
        row["product_cdf"] = float(cdf[i])
        row["product_weight_source"] = weight_source
        out.append(row)
    return out


def two_body_outgoing_neutron_energy_from_kernel(incoming_energy_eV, mu_cm, A, Q_eV):
    """
    Compute outgoing neutron lab energy for a precomputed MF6 LAW=2 kernel.

    This mirrors mf6.two_body_outgoing_neutron_energy(...), but uses A and Q
    already stored in the kernel so runtime does not need to inspect MF6/MF3
    tables for every collision.
    """
    E = float(incoming_energy_eV)
    mu_cm = float(np.clip(mu_cm, -1.0, 1.0))
    A = float(A)
    Q_eV = float(Q_eV)

    if A <= 0.0:
        return 0.0

    K_cm_out = (A / (A + 1.0)) * E + Q_eV
    if K_cm_out <= 0.0:
        return 0.0

    E_neutron_cm = (A / (A + 1.0)) * K_cm_out
    if E_neutron_cm <= 0.0:
        return 0.0

    term_cm_motion = E / ((A + 1.0) ** 2)
    term_cross = 2.0 * np.sqrt(max(E * E_neutron_cm, 0.0)) / (A + 1.0) * mu_cm
    E_out = term_cm_motion + E_neutron_cm + term_cross
    return max(float(E_out), 0.0)


def two_body_lab_angle_from_kernel(incoming_energy_eV, mu_cm, A, Q_eV):
    """
    Convert MF6 LAW=2 center-of-mass mu into a lab-frame neutron angle.

    This uses the same two-body kinematics as
    two_body_outgoing_neutron_energy_from_kernel(...). In 2D transport, runtime
    samples the azimuthal sign separately and rotates by theta_lab_rad.
    """
    E = float(incoming_energy_eV)
    mu_cm = float(np.clip(mu_cm, -1.0, 1.0))
    A = float(A)
    Q_eV = float(Q_eV)

    if E <= 0.0 or A <= 0.0:
        return None

    K_cm_out = (A / (A + 1.0)) * E + Q_eV
    if K_cm_out <= 0.0:
        return None

    E_neutron_cm = (A / (A + 1.0)) * K_cm_out
    if E_neutron_cm <= 0.0:
        return None

    E_out = two_body_outgoing_neutron_energy_from_kernel(
        incoming_energy_eV=E,
        mu_cm=mu_cm,
        A=A,
        Q_eV=Q_eV,
    )
    if E_out <= 0.0:
        return None

    parallel = np.sqrt(E) / (A + 1.0) + np.sqrt(E_neutron_cm) * mu_cm
    mu_lab = float(parallel / np.sqrt(E_out))
    mu_lab = float(np.clip(mu_lab, -1.0, 1.0))
    theta_lab = float(np.arccos(mu_lab))

    return {
        "mu_lab": mu_lab,
        "theta_lab_rad": theta_lab,
        "theta_lab_deg": float(np.degrees(theta_lab)),
    }


def build_mf6_law1_product_kernel(target_data, mt, product_index, incoming_energy_eV):
    """Build one precomputed MF6 LAW=1 outgoing-energy CDF."""
    Ep, pdf = mf6_law1_energy_pdf_at(
        data=target_data,
        product_index=product_index,
        energy_eV=incoming_energy_eV,
    )
    if Ep is None or pdf is None:
        return None

    Ep_grid, Ep_cdf = cdf_from_pdf_grid(Ep, pdf)
    if Ep_grid is None or Ep_cdf is None:
        return None

    return {
        "product_index": int(product_index),
        "law": 1,
        "sample_kind": 1,
        "grid": Ep_grid,
        "cdf": Ep_cdf,
        "A": 0.0,
        "Q_eV": 0.0,
        "source": "kernel-MF6-LAW1",
    }


def build_mf6_law2_product_kernel(target_data, mt, product_index, incoming_energy_eV, n_mu_grid=801):
    """Build one precomputed MF6 LAW=2 center-of-mass angular CDF."""
    coeffs = mf6_law2_coeffs_at(
        data=target_data,
        product_index=product_index,
        energy_eV=incoming_energy_eV,
    )
    if coeffs is None:
        return None

    mu_grid = np.linspace(-1.0, 1.0, int(n_mu_grid))
    pdf = legendre_pdf_from_coeffs(mu_grid, coeffs)
    mu_grid, mu_cdf = cdf_from_pdf_grid(mu_grid, pdf)
    if mu_grid is None or mu_cdf is None:
        return None

    A = get_target_A_from_data(target_data)
    excitation_eV = get_target_A_from_threshold_curve(target_data, mt, A=A)
    Q_eV = -float(excitation_eV)

    return {
        "product_index": int(product_index),
        "law": 2,
        "sample_kind": 2,
        "grid": mu_grid,
        "cdf": mu_cdf,
        "A": float(A),
        "Q_eV": float(Q_eV),
        "source": "kernel-MF6-LAW2-two-body",
    }


def build_mf6_kernel_for_channel(target_data, mt, incoming_energy_eV, n_mu_grid=801):
    """
    Build a precomputed MF6 outgoing-neutron kernel for one isotope/MT/bin.

    Current scope:
        - LAW=1 outgoing-energy distributions
        - LAW=2 two-body angular distributions plus precomputed A and Q

    Unsupported or unusable MF6 products are skipped. If no usable outgoing
    neutron product remains, an empty dict is returned and transport will use
    the old runtime MF6 fallback.
    """
    mt = int(mt)
    incoming_energy_eV = float(incoming_energy_eV)

    neutron_products = mf6_neutron_product_indices(target_data, mt)
    if len(neutron_products) == 0:
        return {}

    product_kernels = []
    for product_index in neutron_products:
        law = int(target_data["mf6_prod_LAW"][int(product_index)])
        y = mf6_yield_at_product_index(target_data, product_index, incoming_energy_eV)

        product_kernel = None
        if law == 1:
            product_kernel = build_mf6_law1_product_kernel(
                target_data=target_data,
                mt=mt,
                product_index=product_index,
                incoming_energy_eV=incoming_energy_eV,
            )
        elif law == 2:
            product_kernel = build_mf6_law2_product_kernel(
                target_data=target_data,
                mt=mt,
                product_index=product_index,
                incoming_energy_eV=incoming_energy_eV,
                n_mu_grid=n_mu_grid,
            )

        if product_kernel is not None:
            product_kernel["yield"] = max(float(y), 0.0)
            product_kernels.append(product_kernel)

    product_kernels = mf6_product_cdf_from_kernel_products(product_kernels)
    if len(product_kernels) == 0:
        return {}

    return {
        "mf6_has_kernel": True,
        "mf6_products": product_kernels,
        "mf6_product_count": len(product_kernels),
        "mf6_total_neutron_product_count": len(neutron_products),
        "mf6_missing_product_count": len(neutron_products) - len(product_kernels),
    }


def sample_mf6_outgoing_neutron_energy_from_kernel(chosen_reaction, incoming_energy_eV):
    """
    Sample one outgoing neutron from a precomputed MF6 channel kernel.

    Returns the same style of dictionary as mf6.sample_mf6_outgoing_neutron_energy(...).
    """
    if not chosen_reaction.get("mf6_has_kernel", False):
        return None

    products = chosen_reaction.get("mf6_products", [])
    if len(products) == 0:
        return None

    R_product = np.random.random()
    chosen_product = None
    for product in products:
        if R_product <= float(product.get("product_cdf", 0.0)):
            chosen_product = product
            break
    if chosen_product is None:
        chosen_product = products[-1]

    grid = np.array(chosen_product.get("grid", []), dtype=float)
    cdf = np.array(chosen_product.get("cdf", []), dtype=float)
    if len(grid) <= 1 or len(grid) != len(cdf):
        return None

    R_sample = np.random.random()
    sampled_value = float(np.interp(R_sample, cdf, grid))
    sample_kind = int(chosen_product.get("sample_kind", 0))
    law = int(chosen_product.get("law", 0))

    if sample_kind == 1:
        outgoing_energy = max(sampled_value, 0.0)
        mu_cm = None
        angle_row = None

    elif sample_kind == 2:
        mu_cm = float(np.clip(sampled_value, -1.0, 1.0))
        outgoing_energy = two_body_outgoing_neutron_energy_from_kernel(
            incoming_energy_eV=incoming_energy_eV,
            mu_cm=mu_cm,
            A=chosen_product.get("A", 0.0),
            Q_eV=chosen_product.get("Q_eV", 0.0),
        )
        angle_row = two_body_lab_angle_from_kernel(
            incoming_energy_eV=incoming_energy_eV,
            mu_cm=mu_cm,
            A=chosen_product.get("A", 0.0),
            Q_eV=chosen_product.get("Q_eV", 0.0),
        )

    else:
        return None

    return {
        "mt": int(chosen_reaction["mt"]),
        "incident_energy_eV": float(incoming_energy_eV),
        "energy_eV": float(outgoing_energy),
        "product_index": int(chosen_product.get("product_index", -1)),
        "law": law,
        "yield_at_E": float(chosen_product.get("yield", 0.0)),
        "mu_cm": mu_cm,
        "mu_lab": angle_row.get("mu_lab") if angle_row is not None else None,
        "theta_lab_rad": angle_row.get("theta_lab_rad") if angle_row is not None else None,
        "theta_lab_deg": angle_row.get("theta_lab_deg") if angle_row is not None else None,
        "angle_source": str(chosen_product.get("source", "kernel-MF6")) if angle_row is not None else None,
        "angle_frame": "lab-from-center-of-mass" if angle_row is not None else None,
        "source": str(chosen_product.get("source", "kernel-MF6")),
        "mf6_product_random_number": float(R_product),
        "mf6_sample_random_number": float(R_sample),
        "mf6_product_probability": chosen_product.get("product_probability"),
        "mf6_product_cdf": chosen_product.get("product_cdf"),
        "mf6_product_weight_source": chosen_product.get("product_weight_source"),
    }

def make_log_energy_edges(E_min_eV=1.0e-5, E_max_eV=2.0e7, bins_per_decade=50):
    """
    Build logarithmic energy-bin edges.

    Default range:
        1e-5 eV to 20 MeV

    bins_per_decade controls accuracy/speed.
    Larger value = better energy resolution, larger kernel.
    """
    E_min_eV = float(E_min_eV)
    E_max_eV = float(E_max_eV)
    bins_per_decade = int(bins_per_decade)

    if E_min_eV <= 0.0:
        raise ValueError("E_min_eV must be positive for log energy bins.")
    if E_max_eV <= E_min_eV:
        raise ValueError("E_max_eV must be greater than E_min_eV.")
    if bins_per_decade <= 0:
        raise ValueError("bins_per_decade must be positive.")

    log10_min = np.log10(E_min_eV)
    log10_max = np.log10(E_max_eV)

    n_bins = int(np.ceil((log10_max - log10_min) * bins_per_decade))
    n_bins = max(n_bins, 1)

    return np.logspace(log10_min, log10_max, n_bins + 1)


def make_energy_centers_from_edges(energy_edges_eV):
    """
    Return geometric bin centers for log-spaced bins.
    """
    edges = np.array(energy_edges_eV, dtype=float)
    return np.sqrt(edges[:-1] * edges[1:])


def is_flat_kernel(kernel):
    """Return True when kernel runtime storage is compact flat arrays."""
    return bool(kernel.get("flat_storage", False))


def kernel_bin_count(kernel):
    if is_flat_kernel(kernel):
        return len(kernel["bin_index"])
    return len(kernel["bins"])


def make_flat_kernel_bin_view(kernel, bin_index):
    """Return a lightweight view of one flat kernel bin."""
    bin_index = int(bin_index)
    return {
        "flat_kernel_bin_view": True,
        "kernel": kernel,
        "bin_index": bin_index,
        "channel_offset": int(kernel["bin_channel_offsets"][bin_index]),
        "channel_count": int(kernel["bin_n_channels"][bin_index]),
    }


def flat_channel_to_reaction_row(kernel, channel_index):
    """
    Materialize one selected flat-array channel as the reaction-row dictionary
    expected by the transport code.

    Only the sampled channel is materialized. Full bin channel lists stay flat.
    """
    j = int(channel_index)
    isotope = str(kernel["channel_target_isotope"][j])

    if isotope not in kernel["isotope_data_by_label"]:
        raise KeyError(f"Kernel requires isotope {isotope}, but it is not loaded in material.")

    bin_i = int(kernel["channel_bin_index"][j])
    row = {
        "target_isotope": isotope,
        "target_data": kernel["isotope_data_by_label"][isotope],
        "mt": int(kernel["channel_mt"][j]),
        "sigma_barns": float(kernel["channel_sigma_barns"][j]),
        "sigma_m2": float(kernel["channel_sigma_m2"][j]),
        "number_density_m3": float(kernel["channel_number_density_m3"][j]),
        "macro_xs_1_per_m": float(kernel["channel_macro_xs_1_per_m"][j]),
        "probability": float(kernel["channel_probability"][j]),
        "cdf": float(kernel["channel_cdf"][j]),
        "kernel_bin_index": bin_i,
        "kernel_energy_eV": float(kernel["bin_energy_eV"][bin_i]),
        "kernel_E_low_eV": float(kernel["bin_E_low_eV"][bin_i]),
        "kernel_E_high_eV": float(kernel["bin_E_high_eV"][bin_i]),
    }

    if kernel.get("has_multiplicity_arrays", False):
        row["n_out_expected"] = float(kernel["n_out_expected"][j])
        row["n_out_integer_rule"] = empty_string_to_none(kernel["n_out_integer_rule"][j])
        row["n_out_source"] = empty_string_to_none(kernel["n_out_source"][j])
        row["n_out_mt_count"] = int(kernel["n_out_mt_count"][j])
        row["n_out_mf6_total_yield"] = float(kernel["n_out_mf6_total_yield"][j])
        row["n_out_mf6_product_count"] = int(kernel["n_out_mf6_product_count"][j])

    if kernel.get("has_residual_arrays", False) and bool(kernel["residual_product_kernel"][j]):
        row["residual_product_kernel"] = True
        row["residual_product"] = empty_string_to_none(kernel["residual_product"][j])
        row["residual_product_Z"] = nan_to_none(kernel["residual_product_Z"][j])
        row["residual_product_A"] = nan_to_none(kernel["residual_product_A"][j])
        row["product_note"] = empty_string_to_none(kernel["product_note"][j])
        row["product_state"] = missing_int_to_none(kernel["product_state"][j])
        row["product_state_source"] = empty_string_to_none(kernel["product_state_source"][j])
        row["product_branch_probability"] = nan_to_none(kernel["product_branch_probability"][j])
        row["product_branch_total_xs"] = nan_to_none(kernel["product_branch_total_xs"][j])

        branches = []
        if bool(kernel["residual_has_mf10_branch_kernel"][j]):
            offset = int(kernel["residual_branch_offsets"][j])
            count = int(kernel["residual_branch_counts"][j])
            for k in range(offset, offset + count):
                branches.append({
                    "product": empty_string_to_none(kernel["residual_branch_product"][k]),
                    "product_Z": nan_to_none(kernel["residual_branch_product_Z"][k]),
                    "product_A": nan_to_none(kernel["residual_branch_product_A"][k]),
                    "product_state": missing_int_to_none(kernel["residual_branch_product_state"][k]),
                    "mf10_xs": float(kernel["residual_branch_mf10_xs"][k]),
                    "branch_probability": float(kernel["residual_branch_probability"][k]),
                    "branch_cdf": float(kernel["residual_branch_cdf"][k]),
                    "branch_total_xs": float(kernel["residual_branch_total_xs"][k]),
                })

        row["residual_has_mf10_branch_kernel"] = len(branches) > 0
        row["residual_mf10_branches"] = branches

    if bool(kernel["elastic_has_mf4_kernel"][j]):
        row["elastic_has_mf4_kernel"] = True
        row["elastic_LCT"] = int(kernel["elastic_LCT"][j])
        row["elastic_A"] = float(kernel["elastic_A"][j])
        row["elastic_angle_source"] = str(kernel["elastic_angle_source"][j])
        row["elastic_angle_frame"] = str(kernel["elastic_angle_frame"][j])
        row["elastic_mu_grid"] = kernel["elastic_mu_grid"]
        row["elastic_mu_cdf"] = kernel["elastic_mu_cdf"][j, :]

    if kernel.get("has_mf6_arrays", False) and bool(kernel["mf6_has_kernel"][j]):
        offset = int(kernel["mf6_product_offsets"][j])
        count = int(kernel["mf6_product_counts"][j])
        products = []

        for k in range(offset, offset + count):
            grid_offset = int(kernel["mf6_product_grid_offsets"][k])
            grid_length = int(kernel["mf6_product_grid_lengths"][k])
            products.append({
                "product_index": int(kernel["mf6_product_index"][k]),
                "law": int(kernel["mf6_product_law"][k]),
                "sample_kind": int(kernel["mf6_product_sample_kind"][k]),
                "yield": float(kernel["mf6_product_yield"][k]),
                "product_probability": float(kernel["mf6_product_probability"][k]),
                "product_cdf": float(kernel["mf6_product_cdf"][k]),
                "A": float(kernel["mf6_product_A"][k]),
                "Q_eV": float(kernel["mf6_product_Q_eV"][k]),
                "source": str(kernel["mf6_product_source"][k]),
                "product_weight_source": str(kernel["mf6_product_weight_source"][k]),
                "grid": kernel["mf6_product_grid"][grid_offset : grid_offset + grid_length],
                "cdf": kernel["mf6_product_grid_cdf"][grid_offset : grid_offset + grid_length],
            })

        row["mf6_has_kernel"] = True
        row["mf6_products"] = products
        row["mf6_product_count"] = len(products)
        row["mf6_total_neutron_product_count"] = int(kernel["mf6_total_neutron_product_count"][j])
        row["mf6_missing_product_count"] = int(kernel["mf6_missing_product_count"][j])

    return row


class FlatKernelBinsView:
    """Lazy compatibility view for old diagnostics that iterate kernel["bins"]."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __len__(self):
        return len(self.kernel["bin_index"])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        offset = int(self.kernel["bin_channel_offsets"][index])
        count = int(self.kernel["bin_n_channels"][index])
        channels = [
            flat_channel_to_reaction_row(self.kernel, j)
            for j in range(offset, offset + count)
        ]

        return {
            "bin_index": int(self.kernel["bin_index"][index]),
            "energy_eV": float(self.kernel["bin_energy_eV"][index]),
            "E_low_eV": float(self.kernel["bin_E_low_eV"][index]),
            "E_high_eV": float(self.kernel["bin_E_high_eV"][index]),
            "Sigma_total_1_per_m": float(self.kernel["bin_Sigma_total_1_per_m"][index]),
            "mean_free_path_m": float(self.kernel["bin_mean_free_path_m"][index]),
            "channels": channels,
            "n_channels": count,
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def energy_bin_index(kernel, energy_eV):
    """
    Find the kernel energy-bin index for a neutron energy.

    Returns None if energy is outside the kernel range.
    """
    energy_eV = float(energy_eV)

    if energy_eV <= 0.0:
        return None

    edges = kernel["energy_edges_eV"]

    if energy_eV < edges[0] or energy_eV > edges[-1]:
        return None

    idx = int(np.searchsorted(edges, energy_eV, side="right") - 1)

    if idx < 0:
        return None

    n_bins = kernel_bin_count(kernel)
    if idx >= n_bins:
        idx = n_bins - 1

    return idx


def add_cdf_to_channels(channels):
    """
    Add cumulative distribution values to reaction channels.

    Each channel already has:
        probability = Sigma_channel / Sigma_total

    This adds:
        cdf
    """
    if len(channels) == 0:
        return channels

    running = 0.0
    new_channels = []

    for row in channels:
        new_row = dict(row)
        running += float(new_row.get("probability", 0.0))
        new_row["cdf"] = running
        new_channels.append(new_row)

    # Protect against roundoff so the final channel always catches R near 1.
    new_channels[-1]["cdf"] = 1.0

    return new_channels


def string_or_empty(value):
    """Convert optional strings to a saveable NPZ string value."""
    if value is None:
        return ""
    return str(value)


def empty_string_to_none(value):
    text = str(value)
    if text == "":
        return None
    return text


def float_or_nan(value):
    if value is None:
        return np.nan
    return float(value)


def nan_to_none(value):
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def int_or_missing(value):
    if value is None:
        return -1
    return int(value)


def missing_int_to_none(value):
    value = int(value)
    if value < 0:
        return None
    return value


FISSION_MTS = {18, 19, 20, 21, 38}


def integer_rule_for_expected_multiplicity(expected):
    expected = max(float(expected), 0.0)
    if expected <= 0.0:
        return "zero"
    if np.isclose(expected, round(expected), rtol=0.0, atol=1.0e-8):
        return "fixed-integer"
    return "stochastic-floor-plus-bernoulli"


def build_neutron_multiplicity_kernel_for_channel(target_data, mt, incoming_energy_eV):
    """
    Precompute outgoing-neutron multiplicity metadata for one reaction row.

    The row stores the expected count and the integer rule. Runtime still makes
    any needed random draw, so non-integer MF6 yields remain analog MC behavior.
    """
    mt = int(mt)
    incoming_energy_eV = float(incoming_energy_eV)

    mt_count = int(count_outgoing_neutrons(mt))
    neutron_products = mf6_neutron_product_indices(target_data, mt)

    mf6_yields = []
    for product_index in neutron_products:
        y = mf6_yield_at_product_index(target_data, product_index, incoming_energy_eV)
        mf6_yields.append(max(float(y), 0.0))

    mf6_total_yield = float(np.sum(mf6_yields)) if len(mf6_yields) > 0 else 0.0

    if mf6_total_yield > 0.0:
        expected = mf6_total_yield
        source = "fission-nu-from-MF6-yield" if mt in FISSION_MTS else "MF6-neutron-yield"
    elif mt in FISSION_MTS:
        expected = float(mt_count)
        source = "fission-nu-missing-zero-fallback" if mt_count <= 0 else "fission-nu-missing-MT-fallback"
    else:
        expected = float(mt_count)
        source = "MT-emitted-particle-logic"

    integer_rule = integer_rule_for_expected_multiplicity(expected)

    return {
        "n_out_expected": float(expected),
        "n_out_integer_rule": integer_rule,
        "n_out_source": source,
        "n_out_mt_count": int(mt_count),
        "n_out_mf6_total_yield": float(mf6_total_yield),
        "n_out_mf6_product_count": int(len(neutron_products)),
    }


def sample_neutron_multiplicity_from_kernel(chosen_reaction):
    """Sample an integer outgoing-neutron count from precomputed row metadata."""
    if "n_out_expected" not in chosen_reaction:
        return None

    expected = max(float(chosen_reaction.get("n_out_expected", 0.0)), 0.0)
    integer_rule = chosen_reaction.get("n_out_integer_rule")
    if integer_rule is None:
        integer_rule = integer_rule_for_expected_multiplicity(expected)
    integer_rule = str(integer_rule)

    random_number = None
    if integer_rule == "zero" or expected <= 0.0:
        n_out = 0
    elif integer_rule == "fixed-integer":
        n_out = int(round(expected))
    else:
        base = int(np.floor(expected))
        frac = expected - base
        random_number = float(np.random.random())
        n_out = base + (1 if random_number < frac else 0)

    return {
        "n_out": int(max(n_out, 0)),
        "n_out_expected": float(expected),
        "n_out_integer_rule": integer_rule,
        "n_out_source": chosen_reaction.get("n_out_source"),
        "n_out_random_number": random_number,
        "n_out_mt_count": chosen_reaction.get("n_out_mt_count"),
        "n_out_mf6_total_yield": chosen_reaction.get("n_out_mf6_total_yield"),
        "n_out_mf6_product_count": chosen_reaction.get("n_out_mf6_product_count"),
    }


def mf10_branch_cdf_from_rows(rows):
    """Convert MF10 product-state XS rows into precomputed branch CDF rows."""
    if len(rows) == 0:
        return []

    total = 0.0
    for row in rows:
        total += max(float(row.get("mf10_xs", 0.0)), 0.0)

    if total <= 0.0:
        return []

    running = 0.0
    branches = []

    for row in rows:
        probability = max(float(row.get("mf10_xs", 0.0)), 0.0) / total
        running += probability
        branches.append({
            "product": row.get("product"),
            "product_Z": row.get("product_Z"),
            "product_A": row.get("product_A"),
            "product_state": row.get("product_state"),
            "mf10_xs": float(row.get("mf10_xs", 0.0)),
            "branch_probability": float(probability),
            "branch_cdf": float(running),
            "branch_total_xs": float(total),
        })

    branches[-1]["branch_cdf"] = 1.0
    return branches


def build_residual_product_kernel_for_channel(target_data, mt, incoming_energy_eV):
    """
    Precompute residual-product data for one isotope/MT/energy-bin channel.

    The MT-implied residual product is stored directly. MF10 metastable
    branching stays stochastic by storing a small branch CDF in the row instead
    of choosing a branch during kernel construction.
    """
    mt = int(mt)
    product_info = base_residual_product_for_mt(target_data, mt)

    row = {
        "residual_product_kernel": True,
        "residual_product": product_info.get("product"),
        "residual_product_Z": product_info.get("product_Z"),
        "residual_product_A": product_info.get("product_A"),
        "product_note": product_info.get("note"),
        "product_state": product_info.get("product_state"),
        "product_state_source": product_info.get("product_state_source"),
        "product_branch_probability": product_info.get("product_branch_probability"),
        "product_branch_total_xs": product_info.get("product_branch_total_xs"),
        "residual_has_mf10_branch_kernel": False,
        "residual_mf10_branches": [],
    }

    if product_info.get("product") is not None and product_info.get("note") == "residual product":
        branch_rows = mf10_product_rows_at_energy(
            data=target_data,
            mt=mt,
            energy_eV=incoming_energy_eV,
            product_Z=product_info.get("product_Z"),
            product_A=product_info.get("product_A"),
        )
        branches = mf10_branch_cdf_from_rows(branch_rows)
        if len(branches) > 0:
            row["residual_has_mf10_branch_kernel"] = True
            row["residual_mf10_branches"] = branches

    return row


def sample_residual_product_from_kernel(chosen_reaction):
    """
    Return residual-product info from a precomputed reaction row.

    Returns None for old rows/kernels that do not carry residual-product data.
    """
    if not chosen_reaction.get("residual_product_kernel", False):
        return None

    product_info = {
        "target": chosen_reaction.get("target_isotope"),
        "product": chosen_reaction.get("residual_product"),
        "product_Z": chosen_reaction.get("residual_product_Z"),
        "product_A": chosen_reaction.get("residual_product_A"),
        "note": chosen_reaction.get("product_note"),
        "product_state": chosen_reaction.get("product_state"),
        "product_state_source": chosen_reaction.get("product_state_source"),
        "product_branch_probability": chosen_reaction.get("product_branch_probability"),
        "product_branch_total_xs": chosen_reaction.get("product_branch_total_xs"),
        "residual_product_sampling_source": "kernel-residual-product",
        "residual_product_random_number": None,
        "residual_product_branch_cdf": None,
    }

    if chosen_reaction.get("residual_has_mf10_branch_kernel", False):
        branches = chosen_reaction.get("residual_mf10_branches", [])
        if len(branches) > 0:
            R = np.random.random()
            chosen_branch = branches[-1]
            for branch in branches:
                if R <= float(branch.get("branch_cdf", 0.0)):
                    chosen_branch = branch
                    break

            product_info["product"] = chosen_branch.get("product")
            product_info["product_Z"] = chosen_branch.get("product_Z")
            product_info["product_A"] = chosen_branch.get("product_A")
            product_info["note"] = "residual product from MF10 branching"
            product_info["product_state"] = chosen_branch.get("product_state")
            product_info["product_state_source"] = "MF10"
            product_info["product_branch_probability"] = chosen_branch.get("branch_probability")
            product_info["product_branch_total_xs"] = chosen_branch.get("branch_total_xs")
            product_info["residual_product_sampling_source"] = "kernel-MF10-branch-cdf"
            product_info["residual_product_random_number"] = float(R)
            product_info["residual_product_branch_cdf"] = chosen_branch.get("branch_cdf")

    return product_info


def build_material_transport_kernel(
    material,
    E_min_eV=1.0e-5,
    E_max_eV=2.0e7,
    bins_per_decade=50,
    print_progress=True,
    build_residual_products=True,
    build_mf4_elastic=True,
    mf4_mu_grid_count=801,
    build_mf6_neutrons=True,
    mf6_mu_grid_count=801,
):
    """
    Precompute the material reaction kernel.

    This evaluates the full material reaction list once per energy bin center.
    During transport, the neutron will only do a bin lookup.

    Optional kernel layers:
        build_residual_products=True precomputes residual-product rows and
        MF10 branch CDFs.
        build_mf4_elastic=True precomputes MT=2 angular CDFs.
        build_mf6_neutrons=True precomputes MF6 outgoing-neutron CDFs for
        supported non-elastic channels.
    """
    energy_edges_eV = make_log_energy_edges(
        E_min_eV=E_min_eV,
        E_max_eV=E_max_eV,
        bins_per_decade=bins_per_decade,
    )

    energy_centers_eV = make_energy_centers_from_edges(energy_edges_eV)

    bins = []

    for i, E in enumerate(energy_centers_eV):
        if print_progress and (i % 50 == 0 or i == len(energy_centers_eV) - 1):
            print(f"Building kernel bin {i + 1}/{len(energy_centers_eV)} at E = {E:.6e} eV")

        channels = build_material_reaction_list_for_energy(
            material=material,
            energy_eV=E,
        )

        Sigma_total = total_macroscopic_xs(channels)

        channels = add_cdf_to_channels(channels)

        for row in channels:
            mt = int(row["mt"])
            row["kernel_bin_index"] = int(i)
            row["kernel_energy_eV"] = float(E)
            row["kernel_E_low_eV"] = float(energy_edges_eV[i])
            row["kernel_E_high_eV"] = float(energy_edges_eV[i + 1])

            row.update(
                build_neutron_multiplicity_kernel_for_channel(
                    target_data=row["target_data"],
                    mt=mt,
                    incoming_energy_eV=E,
                )
            )

            if build_residual_products:
                row.update(
                    build_residual_product_kernel_for_channel(
                        target_data=row["target_data"],
                        mt=mt,
                        incoming_energy_eV=E,
                    )
                )

            if build_mf4_elastic and mt == 2:
                row.update(
                    build_elastic_mf4_kernel_for_channel(
                        target_data=row["target_data"],
                        incoming_energy_eV=E,
                        n_mu_grid=mf4_mu_grid_count,
                    )
                )

            if build_mf6_neutrons and mt != 2:
                row.update(
                    build_mf6_kernel_for_channel(
                        target_data=row["target_data"],
                        mt=mt,
                        incoming_energy_eV=E,
                        n_mu_grid=mf6_mu_grid_count,
                    )
                )

        bins.append({
            "bin_index": i,
            "energy_eV": float(E),
            "E_low_eV": float(energy_edges_eV[i]),
            "E_high_eV": float(energy_edges_eV[i + 1]),
            "Sigma_total_1_per_m": float(Sigma_total),
            "mean_free_path_m": float(1.0 / Sigma_total) if Sigma_total > 0.0 else float("inf"),
            "channels": channels,
            "n_channels": len(channels),
        })

    kernel = {
        "kind": "material_reaction_kernel_v1",
        "material_name": material.get("name", "unknown material"),
        "material_kernel_label": material_kernel_label(material),
        "material_fingerprint": material_kernel_fingerprint(material),
        "E_min_eV": float(E_min_eV),
        "E_max_eV": float(E_max_eV),
        "bins_per_decade": int(bins_per_decade),
        "build_neutron_multiplicity": True,
        "build_residual_products": bool(build_residual_products),
        "build_mf4_elastic": bool(build_mf4_elastic),
        "mf4_mu_grid_count": int(mf4_mu_grid_count),
        "build_mf6_neutrons": bool(build_mf6_neutrons),
        "mf6_mu_grid_count": int(mf6_mu_grid_count),
        "energy_edges_eV": energy_edges_eV,
        "energy_centers_eV": energy_centers_eV,
        "bins": bins,
    }

    return kernel

def get_kernel_bin(kernel, energy_eV):
    """
    Return the precomputed kernel bin for one neutron energy.
    """
    idx = energy_bin_index(kernel, energy_eV)

    if idx is None:
        return None

    if is_flat_kernel(kernel):
        return make_flat_kernel_bin_view(kernel, idx)

    return kernel["bins"][idx]


def get_kernel_reaction_rows(kernel, energy_eV):
    """
    Return precomputed reaction rows and total Sigma for one neutron energy.

    Nested development kernels return copied reaction-row dictionaries. Flat
    loaded kernels return a lightweight bin view so runtime sampling can stay
    in compact arrays until one channel is selected.
    """
    if is_flat_kernel(kernel):
        idx = energy_bin_index(kernel, energy_eV)

        if idx is None:
            return [], 0.0

        return (
            make_flat_kernel_bin_view(kernel, idx),
            float(kernel["bin_Sigma_total_1_per_m"][idx]),
        )

    bin_data = get_kernel_bin(kernel, energy_eV)

    if bin_data is None:
        return [], 0.0

    rows = [dict(row) for row in bin_data["channels"]]
    Sigma_total = float(bin_data["Sigma_total_1_per_m"])

    return rows, Sigma_total


def kernel_summary(kernel, max_rows=10):
    """
    Print a short summary of the precomputed kernel.
    """
    print("Kernel kind:", kernel.get("kind"))
    print("Material:", kernel.get("material_name"))
    print("Energy range:", kernel["E_min_eV"], "to", kernel["E_max_eV"], "eV")
    print("Bins per decade:", kernel["bins_per_decade"])
    print("Storage:", "flat arrays" if is_flat_kernel(kernel) else "nested dictionaries")
    print("Total bins:", kernel_bin_count(kernel))

    if is_flat_kernel(kernel):
        nonzero_indices = np.where(kernel["bin_Sigma_total_1_per_m"] > 0.0)[0]
        print("Bins with open reactions:", len(nonzero_indices))

        multiplicity_kernel_count = len(kernel["channel_mt"]) if kernel.get("has_multiplicity_arrays", False) else 0
        if kernel.get("has_multiplicity_arrays", False):
            mf6_sources = np.isin(
                kernel["n_out_source"],
                ["MF6-neutron-yield", "fission-nu-from-MF6-yield"],
            )
            multiplicity_mf6_count = int(np.sum(mf6_sources))
            multiplicity_stochastic_count = int(
                np.sum(kernel["n_out_integer_rule"] == "stochastic-floor-plus-bernoulli")
            )
        else:
            multiplicity_mf6_count = 0
            multiplicity_stochastic_count = 0

        residual_kernel_count = (
            int(np.sum(kernel["residual_product_kernel"]))
            if kernel.get("has_residual_arrays", False)
            else 0
        )
        residual_mf10_branch_count = (
            int(np.sum(kernel["residual_has_mf10_branch_kernel"]))
            if kernel.get("has_residual_arrays", False)
            else 0
        )

        print("Total channels:", len(kernel["channel_mt"]))
        print("Multiplicity kernel channels:", multiplicity_kernel_count)
        print("Multiplicity MF6-yield channels:", multiplicity_mf6_count)
        print("Multiplicity stochastic-integer channels:", multiplicity_stochastic_count)
        print("Residual-product kernel channels:", residual_kernel_count)
        print("Residual MF10 branch-CDF channels:", residual_mf10_branch_count)

        print("\nFirst nonzero bins:")
        for idx in nonzero_indices[:max_rows]:
            print(
                f"bin {int(kernel['bin_index'][idx]):5d} | "
                f"E = {kernel['bin_energy_eV'][idx]:.6e} eV | "
                f"Sigma = {kernel['bin_Sigma_total_1_per_m'][idx]:.6e} 1/m | "
                f"channels = {int(kernel['bin_n_channels'][idx])}"
            )
        return

    nonzero_bins = [b for b in kernel["bins"] if b["Sigma_total_1_per_m"] > 0.0]
    print("Bins with open reactions:", len(nonzero_bins))

    residual_kernel_count = 0
    residual_mf10_branch_count = 0
    multiplicity_kernel_count = 0
    multiplicity_mf6_count = 0
    multiplicity_stochastic_count = 0
    for b in kernel["bins"]:
        for row in b["channels"]:
            if "n_out_expected" in row:
                multiplicity_kernel_count += 1
            if row.get("n_out_source") in ["MF6-neutron-yield", "fission-nu-from-MF6-yield"]:
                multiplicity_mf6_count += 1
            if row.get("n_out_integer_rule") == "stochastic-floor-plus-bernoulli":
                multiplicity_stochastic_count += 1
            if row.get("residual_product_kernel", False):
                residual_kernel_count += 1
            if row.get("residual_has_mf10_branch_kernel", False):
                residual_mf10_branch_count += 1
    print("Multiplicity kernel channels:", multiplicity_kernel_count)
    print("Multiplicity MF6-yield channels:", multiplicity_mf6_count)
    print("Multiplicity stochastic-integer channels:", multiplicity_stochastic_count)
    print("Residual-product kernel channels:", residual_kernel_count)
    print("Residual MF10 branch-CDF channels:", residual_mf10_branch_count)

    print("\nFirst nonzero bins:")
    for b in nonzero_bins[:max_rows]:
        print(
            f"bin {b['bin_index']:5d} | "
            f"E = {b['energy_eV']:.6e} eV | "
            f"Sigma = {b['Sigma_total_1_per_m']:.6e} 1/m | "
            f"channels = {b['n_channels']}"
        )

def sample_reaction_from_kernel_cdf(channels):
    """
    Sample one reaction channel from precomputed CDF values.

    This is the Step 2 runtime sampler.

    Preferred kernel path:
        row["cdf"] already exists from preprocessing.

    Fallback path:
        if cdf is missing, fall back to probability accumulation.
        This keeps old non-kernel transport from breaking.
    """
    if isinstance(channels, dict) and channels.get("flat_kernel_bin_view", False):
        kernel = channels["kernel"]
        count = int(channels["channel_count"])
        if count <= 0:
            return None

        offset = int(channels["channel_offset"])
        R = np.random.random()
        cdf = kernel["channel_cdf"][offset : offset + count]
        local_index = int(np.searchsorted(cdf, R, side="left"))

        if local_index >= count:
            local_index = count - 1
            source = "precomputed-flat-kernel-cdf-roundoff"
        else:
            source = "precomputed-flat-kernel-cdf"

        chosen = flat_channel_to_reaction_row(kernel, offset + local_index)
        chosen["reaction_sampling_source"] = source
        chosen["reaction_random_number"] = float(R)
        return chosen

    if len(channels) == 0:
        return None

    R = np.random.random()

    # Fast kernel path: use precomputed CDF.
    if "cdf" in channels[0]:
        for row in channels:
            if R <= float(row["cdf"]):
                chosen = dict(row)
                chosen["reaction_sampling_source"] = "precomputed-kernel-cdf"
                chosen["reaction_random_number"] = float(R)
                return chosen

        # Roundoff protection.
        chosen = dict(channels[-1])
        chosen["reaction_sampling_source"] = "precomputed-kernel-cdf-roundoff"
        chosen["reaction_random_number"] = float(R)
        return chosen

    # Fallback path for old non-kernel rows.
    running = 0.0
    for row in channels:
        running += float(row.get("probability", 0.0))
        if R <= running:
            chosen = dict(row)
            chosen["reaction_sampling_source"] = "runtime-probability-fallback"
            chosen["reaction_random_number"] = float(R)
            return chosen

    chosen = dict(channels[-1])
    chosen["reaction_sampling_source"] = "runtime-probability-fallback-roundoff"
    chosen["reaction_random_number"] = float(R)
    return chosen

def save_material_transport_kernel_npz(kernel, path):
    """
    Save the transport kernel to a compressed NPZ file.

    Important:
        target_data is NOT saved.
        It must be reattached from the loaded material when the kernel is loaded.
    """
    path = Path(path)

    bins = kernel["bins"]
    n_bins = len(bins)

    bin_index = np.array([b["bin_index"] for b in bins], dtype=np.int64)
    bin_energy_eV = np.array([b["energy_eV"] for b in bins], dtype=float)
    bin_E_low_eV = np.array([b["E_low_eV"] for b in bins], dtype=float)
    bin_E_high_eV = np.array([b["E_high_eV"] for b in bins], dtype=float)
    bin_Sigma_total = np.array([b["Sigma_total_1_per_m"] for b in bins], dtype=float)
    bin_mean_free_path = np.array([b["mean_free_path_m"] for b in bins], dtype=float)
    bin_n_channels = np.array([b["n_channels"] for b in bins], dtype=np.int64)

    bin_channel_offsets = np.zeros(n_bins, dtype=np.int64)
    running = 0
    for i, b in enumerate(bins):
        bin_channel_offsets[i] = running
        running += len(b["channels"])

    channel_rows = []
    for b in bins:
        for row in b["channels"]:
            channel_rows.append(row)

    n_channels = len(channel_rows)

    channel_bin_index = np.zeros(n_channels, dtype=np.int64)
    channel_target_isotope = np.empty(n_channels, dtype="<U32")
    channel_mt = np.zeros(n_channels, dtype=np.int64)
    channel_sigma_barns = np.zeros(n_channels, dtype=float)
    channel_sigma_m2 = np.zeros(n_channels, dtype=float)
    channel_number_density_m3 = np.zeros(n_channels, dtype=float)
    channel_macro_xs_1_per_m = np.zeros(n_channels, dtype=float)
    channel_probability = np.zeros(n_channels, dtype=float)
    channel_cdf = np.zeros(n_channels, dtype=float)

    n_out_expected = np.zeros(n_channels, dtype=float)
    n_out_integer_rule = np.empty(n_channels, dtype="<U64")
    n_out_source = np.empty(n_channels, dtype="<U64")
    n_out_mt_count = np.zeros(n_channels, dtype=np.int64)
    n_out_mf6_total_yield = np.zeros(n_channels, dtype=float)
    n_out_mf6_product_count = np.zeros(n_channels, dtype=np.int64)

    residual_product_kernel = np.zeros(n_channels, dtype=bool)
    residual_product = np.empty(n_channels, dtype="<U32")
    residual_product_Z = np.full(n_channels, np.nan, dtype=float)
    residual_product_A = np.full(n_channels, np.nan, dtype=float)
    product_note = np.empty(n_channels, dtype="<U128")
    product_state = np.full(n_channels, -1, dtype=np.int64)
    product_state_source = np.empty(n_channels, dtype="<U64")
    product_branch_probability = np.full(n_channels, np.nan, dtype=float)
    product_branch_total_xs = np.full(n_channels, np.nan, dtype=float)
    residual_has_mf10_branch_kernel = np.zeros(n_channels, dtype=bool)
    residual_branch_offsets = np.zeros(n_channels, dtype=np.int64)
    residual_branch_counts = np.zeros(n_channels, dtype=np.int64)

    elastic_has = np.zeros(n_channels, dtype=bool)
    elastic_LCT = np.zeros(n_channels, dtype=np.int64)
    elastic_A = np.zeros(n_channels, dtype=float)
    elastic_angle_source = np.empty(n_channels, dtype="<U64")
    elastic_angle_frame = np.empty(n_channels, dtype="<U32")

    n_mu = int(kernel.get("mf4_mu_grid_count", 801))
    elastic_mu_grid = np.linspace(-1.0, 1.0, n_mu)
    elastic_mu_cdf = np.full((n_channels, n_mu), np.nan, dtype=float)

    mf6_has = np.zeros(n_channels, dtype=bool)
    mf6_product_offsets = np.zeros(n_channels, dtype=np.int64)
    mf6_product_counts = np.zeros(n_channels, dtype=np.int64)
    mf6_total_neutron_product_count = np.zeros(n_channels, dtype=np.int64)
    mf6_missing_product_count = np.zeros(n_channels, dtype=np.int64)

    residual_branch_rows = []
    for j, row in enumerate(channel_rows):
        branches = (
            row.get("residual_mf10_branches", [])
            if row.get("residual_has_mf10_branch_kernel", False)
            else []
        )
        residual_branch_offsets[j] = len(residual_branch_rows)
        residual_branch_counts[j] = len(branches)
        if len(branches) > 0:
            residual_has_mf10_branch_kernel[j] = True
            for branch in branches:
                residual_branch_rows.append(branch)

    n_residual_branches = len(residual_branch_rows)
    residual_branch_product = np.empty(n_residual_branches, dtype="<U32")
    residual_branch_product_Z = np.full(n_residual_branches, np.nan, dtype=float)
    residual_branch_product_A = np.full(n_residual_branches, np.nan, dtype=float)
    residual_branch_product_state = np.full(n_residual_branches, -1, dtype=np.int64)
    residual_branch_mf10_xs = np.zeros(n_residual_branches, dtype=float)
    residual_branch_probability = np.zeros(n_residual_branches, dtype=float)
    residual_branch_cdf = np.zeros(n_residual_branches, dtype=float)
    residual_branch_total_xs = np.zeros(n_residual_branches, dtype=float)

    mf6_product_rows = []
    for j, row in enumerate(channel_rows):
        products = row.get("mf6_products", []) if row.get("mf6_has_kernel", False) else []
        mf6_product_offsets[j] = len(mf6_product_rows)
        mf6_product_counts[j] = len(products)
        mf6_total_neutron_product_count[j] = int(row.get("mf6_total_neutron_product_count", len(products)))
        mf6_missing_product_count[j] = int(row.get("mf6_missing_product_count", 0))
        if len(products) > 0:
            mf6_has[j] = True
            for product in products:
                mf6_product_rows.append(product)

    n_mf6_products = len(mf6_product_rows)
    mf6_product_index = np.zeros(n_mf6_products, dtype=np.int64)
    mf6_product_law = np.zeros(n_mf6_products, dtype=np.int64)
    mf6_product_sample_kind = np.zeros(n_mf6_products, dtype=np.int64)
    mf6_product_yield = np.zeros(n_mf6_products, dtype=float)
    mf6_product_probability = np.zeros(n_mf6_products, dtype=float)
    mf6_product_cdf = np.zeros(n_mf6_products, dtype=float)
    mf6_product_A = np.zeros(n_mf6_products, dtype=float)
    mf6_product_Q_eV = np.zeros(n_mf6_products, dtype=float)
    mf6_product_source = np.empty(n_mf6_products, dtype="<U64")
    mf6_product_weight_source = np.empty(n_mf6_products, dtype="<U64")
    mf6_product_grid_offsets = np.zeros(n_mf6_products, dtype=np.int64)
    mf6_product_grid_lengths = np.zeros(n_mf6_products, dtype=np.int64)

    total_mf6_grid_length = 0
    for product in mf6_product_rows:
        total_mf6_grid_length += len(np.array(product.get("grid", []), dtype=float))

    mf6_product_grid = np.zeros(total_mf6_grid_length, dtype=float)
    mf6_product_grid_cdf = np.zeros(total_mf6_grid_length, dtype=float)

    for j, row in enumerate(channel_rows):
        channel_bin_index[j] = int(row.get("kernel_bin_index", row.get("bin_index", -1)))
        channel_target_isotope[j] = str(row["target_isotope"])
        channel_mt[j] = int(row["mt"])
        channel_sigma_barns[j] = float(row.get("sigma_barns", 0.0))
        channel_sigma_m2[j] = float(row.get("sigma_m2", 0.0))
        channel_number_density_m3[j] = float(row.get("number_density_m3", 0.0))
        channel_macro_xs_1_per_m[j] = float(row.get("macro_xs_1_per_m", 0.0))
        channel_probability[j] = float(row.get("probability", 0.0))
        channel_cdf[j] = float(row.get("cdf", 0.0))

        n_out_expected[j] = float(row.get("n_out_expected", 0.0))
        n_out_integer_rule[j] = string_or_empty(row.get("n_out_integer_rule"))
        n_out_source[j] = string_or_empty(row.get("n_out_source"))
        n_out_mt_count[j] = int(row.get("n_out_mt_count", 0))
        n_out_mf6_total_yield[j] = float(row.get("n_out_mf6_total_yield", 0.0))
        n_out_mf6_product_count[j] = int(row.get("n_out_mf6_product_count", 0))

        if row.get("residual_product_kernel", False):
            residual_product_kernel[j] = True
            residual_product[j] = string_or_empty(row.get("residual_product"))
            residual_product_Z[j] = float_or_nan(row.get("residual_product_Z"))
            residual_product_A[j] = float_or_nan(row.get("residual_product_A"))
            product_note[j] = string_or_empty(row.get("product_note"))
            product_state[j] = int_or_missing(row.get("product_state"))
            product_state_source[j] = string_or_empty(row.get("product_state_source"))
            product_branch_probability[j] = float_or_nan(row.get("product_branch_probability"))
            product_branch_total_xs[j] = float_or_nan(row.get("product_branch_total_xs"))
        else:
            residual_product[j] = ""
            product_note[j] = ""
            product_state_source[j] = ""

        if row.get("elastic_has_mf4_kernel", False):
            elastic_has[j] = True
            elastic_LCT[j] = int(row.get("elastic_LCT", 0))
            elastic_A[j] = float(row.get("elastic_A", 0.0))
            elastic_angle_source[j] = str(row.get("elastic_angle_source", ""))
            elastic_angle_frame[j] = str(row.get("elastic_angle_frame", ""))

            row_mu = np.array(row["elastic_mu_grid"], dtype=float)
            row_cdf = np.array(row["elastic_mu_cdf"], dtype=float)

            if len(row_mu) == n_mu and np.allclose(row_mu, elastic_mu_grid):
                elastic_mu_cdf[j, :] = row_cdf
            else:
                elastic_mu_cdf[j, :] = np.interp(
                    elastic_mu_grid,
                    row_mu,
                    row_cdf,
                    left=0.0,
                    right=1.0,
                )
        else:
            elastic_angle_source[j] = ""
            elastic_angle_frame[j] = ""

    for k, branch in enumerate(residual_branch_rows):
        residual_branch_product[k] = string_or_empty(branch.get("product"))
        residual_branch_product_Z[k] = float_or_nan(branch.get("product_Z"))
        residual_branch_product_A[k] = float_or_nan(branch.get("product_A"))
        residual_branch_product_state[k] = int_or_missing(branch.get("product_state"))
        residual_branch_mf10_xs[k] = float(branch.get("mf10_xs", 0.0))
        residual_branch_probability[k] = float(branch.get("branch_probability", 0.0))
        residual_branch_cdf[k] = float(branch.get("branch_cdf", 0.0))
        residual_branch_total_xs[k] = float(branch.get("branch_total_xs", 0.0))

    grid_running = 0
    for k, product in enumerate(mf6_product_rows):
        grid = np.array(product.get("grid", []), dtype=float)
        cdf = np.array(product.get("cdf", []), dtype=float)
        n_grid = min(len(grid), len(cdf))

        mf6_product_index[k] = int(product.get("product_index", -1))
        mf6_product_law[k] = int(product.get("law", 0))
        mf6_product_sample_kind[k] = int(product.get("sample_kind", 0))
        mf6_product_yield[k] = float(product.get("yield", 0.0))
        mf6_product_probability[k] = float(product.get("product_probability", 0.0))
        mf6_product_cdf[k] = float(product.get("product_cdf", 0.0))
        mf6_product_A[k] = float(product.get("A", 0.0))
        mf6_product_Q_eV[k] = float(product.get("Q_eV", 0.0))
        mf6_product_source[k] = str(product.get("source", ""))
        mf6_product_weight_source[k] = str(product.get("product_weight_source", ""))
        mf6_product_grid_offsets[k] = grid_running
        mf6_product_grid_lengths[k] = n_grid

        if n_grid > 0:
            mf6_product_grid[grid_running : grid_running + n_grid] = grid[:n_grid]
            mf6_product_grid_cdf[grid_running : grid_running + n_grid] = cdf[:n_grid]
            grid_running += n_grid

    metadata = {
        "kind": kernel.get("kind", "material_reaction_kernel_v1"),
        "material_name": kernel.get("material_name", "unknown material"),
        "material_kernel_label": kernel.get("material_kernel_label", ""),
        "material_fingerprint": kernel.get("material_fingerprint", ""),
        "E_min_eV": float(kernel["E_min_eV"]),
        "E_max_eV": float(kernel["E_max_eV"]),
        "bins_per_decade": int(kernel["bins_per_decade"]),
        "build_neutron_multiplicity": bool(kernel.get("build_neutron_multiplicity", False)),
        "build_residual_products": bool(kernel.get("build_residual_products", False)),
        "build_mf4_elastic": bool(kernel.get("build_mf4_elastic", False)),
        "mf4_mu_grid_count": int(kernel.get("mf4_mu_grid_count", n_mu)),
        "build_mf6_neutrons": bool(kernel.get("build_mf6_neutrons", False)),
        "mf6_mu_grid_count": int(kernel.get("mf6_mu_grid_count", 801)),
    }

    kernel["source_path"] = str(path)

    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata)),
        energy_edges_eV=np.array(kernel["energy_edges_eV"], dtype=float),
        energy_centers_eV=np.array(kernel["energy_centers_eV"], dtype=float),

        bin_index=bin_index,
        bin_energy_eV=bin_energy_eV,
        bin_E_low_eV=bin_E_low_eV,
        bin_E_high_eV=bin_E_high_eV,
        bin_Sigma_total_1_per_m=bin_Sigma_total,
        bin_mean_free_path_m=bin_mean_free_path,
        bin_n_channels=bin_n_channels,
        bin_channel_offsets=bin_channel_offsets,

        channel_bin_index=channel_bin_index,
        channel_target_isotope=channel_target_isotope,
        channel_mt=channel_mt,
        channel_sigma_barns=channel_sigma_barns,
        channel_sigma_m2=channel_sigma_m2,
        channel_number_density_m3=channel_number_density_m3,
        channel_macro_xs_1_per_m=channel_macro_xs_1_per_m,
        channel_probability=channel_probability,
        channel_cdf=channel_cdf,

        n_out_expected=n_out_expected,
        n_out_integer_rule=n_out_integer_rule,
        n_out_source=n_out_source,
        n_out_mt_count=n_out_mt_count,
        n_out_mf6_total_yield=n_out_mf6_total_yield,
        n_out_mf6_product_count=n_out_mf6_product_count,

        residual_product_kernel=residual_product_kernel,
        residual_product=residual_product,
        residual_product_Z=residual_product_Z,
        residual_product_A=residual_product_A,
        product_note=product_note,
        product_state=product_state,
        product_state_source=product_state_source,
        product_branch_probability=product_branch_probability,
        product_branch_total_xs=product_branch_total_xs,
        residual_has_mf10_branch_kernel=residual_has_mf10_branch_kernel,
        residual_branch_offsets=residual_branch_offsets,
        residual_branch_counts=residual_branch_counts,
        residual_branch_product=residual_branch_product,
        residual_branch_product_Z=residual_branch_product_Z,
        residual_branch_product_A=residual_branch_product_A,
        residual_branch_product_state=residual_branch_product_state,
        residual_branch_mf10_xs=residual_branch_mf10_xs,
        residual_branch_probability=residual_branch_probability,
        residual_branch_cdf=residual_branch_cdf,
        residual_branch_total_xs=residual_branch_total_xs,

        elastic_has_mf4_kernel=elastic_has,
        elastic_LCT=elastic_LCT,
        elastic_A=elastic_A,
        elastic_angle_source=elastic_angle_source,
        elastic_angle_frame=elastic_angle_frame,
        elastic_mu_grid=elastic_mu_grid,
        elastic_mu_cdf=elastic_mu_cdf,

        mf6_has_kernel=mf6_has,
        mf6_product_offsets=mf6_product_offsets,
        mf6_product_counts=mf6_product_counts,
        mf6_total_neutron_product_count=mf6_total_neutron_product_count,
        mf6_missing_product_count=mf6_missing_product_count,
        mf6_product_index=mf6_product_index,
        mf6_product_law=mf6_product_law,
        mf6_product_sample_kind=mf6_product_sample_kind,
        mf6_product_yield=mf6_product_yield,
        mf6_product_probability=mf6_product_probability,
        mf6_product_cdf=mf6_product_cdf,
        mf6_product_A=mf6_product_A,
        mf6_product_Q_eV=mf6_product_Q_eV,
        mf6_product_source=mf6_product_source,
        mf6_product_weight_source=mf6_product_weight_source,
        mf6_product_grid_offsets=mf6_product_grid_offsets,
        mf6_product_grid_lengths=mf6_product_grid_lengths,
        mf6_product_grid=mf6_product_grid,
        mf6_product_grid_cdf=mf6_product_grid_cdf,
    )

    print("Saved transport kernel:", path)

def load_material_transport_kernel_npz(path, material):
    """
    Load a transport kernel from NPZ and reattach live target_data from material.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)

    metadata = json.loads(str(data["metadata_json"]))

    saved_fingerprint = metadata.get("material_fingerprint", "")
    if saved_fingerprint:
        current_fingerprint = material_kernel_fingerprint(material)
        if str(saved_fingerprint) != str(current_fingerprint):
            raise ValueError(
                "Kernel material fingerprint does not match the loaded material. "
                f"kernel={saved_fingerprint}, material={current_fingerprint}"
            )

    energy_edges_eV = np.array(data["energy_edges_eV"], dtype=float)
    energy_centers_eV = np.array(data["energy_centers_eV"], dtype=float)

    bin_index = np.array(data["bin_index"], dtype=np.int64)
    bin_energy_eV = np.array(data["bin_energy_eV"], dtype=float)
    bin_E_low_eV = np.array(data["bin_E_low_eV"], dtype=float)
    bin_E_high_eV = np.array(data["bin_E_high_eV"], dtype=float)
    bin_Sigma_total = np.array(data["bin_Sigma_total_1_per_m"], dtype=float)
    bin_mean_free_path = np.array(data["bin_mean_free_path_m"], dtype=float)
    bin_n_channels = np.array(data["bin_n_channels"], dtype=np.int64)
    bin_channel_offsets = np.array(data["bin_channel_offsets"], dtype=np.int64)

    channel_bin_index = np.array(data["channel_bin_index"], dtype=np.int64)
    channel_target_isotope = np.array(data["channel_target_isotope"]).astype(str)
    channel_mt = np.array(data["channel_mt"], dtype=np.int64)
    channel_sigma_barns = np.array(data["channel_sigma_barns"], dtype=float)
    channel_sigma_m2 = np.array(data["channel_sigma_m2"], dtype=float)
    channel_number_density_m3 = np.array(data["channel_number_density_m3"], dtype=float)
    channel_macro_xs_1_per_m = np.array(data["channel_macro_xs_1_per_m"], dtype=float)
    channel_probability = np.array(data["channel_probability"], dtype=float)
    channel_cdf = np.array(data["channel_cdf"], dtype=float)

    has_multiplicity_arrays = "n_out_expected" in data.files
    if has_multiplicity_arrays:
        n_out_expected = np.array(data["n_out_expected"], dtype=float)
        n_out_integer_rule = np.array(data["n_out_integer_rule"]).astype(str)
        n_out_source = np.array(data["n_out_source"]).astype(str)
        n_out_mt_count = np.array(data["n_out_mt_count"], dtype=np.int64)
        n_out_mf6_total_yield = np.array(data["n_out_mf6_total_yield"], dtype=float)
        n_out_mf6_product_count = np.array(data["n_out_mf6_product_count"], dtype=np.int64)
    else:
        n_out_expected = np.zeros(len(channel_mt), dtype=float)

    has_residual_arrays = "residual_product_kernel" in data.files
    if has_residual_arrays:
        residual_product_kernel = np.array(data["residual_product_kernel"], dtype=bool)
        residual_product = np.array(data["residual_product"]).astype(str)
        residual_product_Z = np.array(data["residual_product_Z"], dtype=float)
        residual_product_A = np.array(data["residual_product_A"], dtype=float)
        product_note = np.array(data["product_note"]).astype(str)
        product_state = np.array(data["product_state"], dtype=np.int64)
        product_state_source = np.array(data["product_state_source"]).astype(str)
        product_branch_probability = np.array(data["product_branch_probability"], dtype=float)
        product_branch_total_xs = np.array(data["product_branch_total_xs"], dtype=float)
        residual_has_mf10_branch_kernel = np.array(data["residual_has_mf10_branch_kernel"], dtype=bool)
        residual_branch_offsets = np.array(data["residual_branch_offsets"], dtype=np.int64)
        residual_branch_counts = np.array(data["residual_branch_counts"], dtype=np.int64)
        residual_branch_product = np.array(data["residual_branch_product"]).astype(str)
        residual_branch_product_Z = np.array(data["residual_branch_product_Z"], dtype=float)
        residual_branch_product_A = np.array(data["residual_branch_product_A"], dtype=float)
        residual_branch_product_state = np.array(data["residual_branch_product_state"], dtype=np.int64)
        residual_branch_mf10_xs = np.array(data["residual_branch_mf10_xs"], dtype=float)
        residual_branch_probability = np.array(data["residual_branch_probability"], dtype=float)
        residual_branch_cdf = np.array(data["residual_branch_cdf"], dtype=float)
        residual_branch_total_xs = np.array(data["residual_branch_total_xs"], dtype=float)
    else:
        residual_product_kernel = np.zeros(len(channel_mt), dtype=bool)

    elastic_has = np.array(data["elastic_has_mf4_kernel"], dtype=bool)
    elastic_LCT = np.array(data["elastic_LCT"], dtype=np.int64)
    elastic_A = np.array(data["elastic_A"], dtype=float)
    elastic_angle_source = np.array(data["elastic_angle_source"]).astype(str)
    elastic_angle_frame = np.array(data["elastic_angle_frame"]).astype(str)
    elastic_mu_grid = np.array(data["elastic_mu_grid"], dtype=float)
    elastic_mu_cdf = np.array(data["elastic_mu_cdf"], dtype=float)

    has_mf6_arrays = "mf6_has_kernel" in data.files
    if has_mf6_arrays:
        mf6_has = np.array(data["mf6_has_kernel"], dtype=bool)
        mf6_product_offsets = np.array(data["mf6_product_offsets"], dtype=np.int64)
        mf6_product_counts = np.array(data["mf6_product_counts"], dtype=np.int64)
        mf6_total_neutron_product_count = np.array(data["mf6_total_neutron_product_count"], dtype=np.int64)
        mf6_missing_product_count = np.array(data["mf6_missing_product_count"], dtype=np.int64)
        mf6_product_index = np.array(data["mf6_product_index"], dtype=np.int64)
        mf6_product_law = np.array(data["mf6_product_law"], dtype=np.int64)
        mf6_product_sample_kind = np.array(data["mf6_product_sample_kind"], dtype=np.int64)
        mf6_product_yield = np.array(data["mf6_product_yield"], dtype=float)
        mf6_product_probability = np.array(data["mf6_product_probability"], dtype=float)
        mf6_product_cdf = np.array(data["mf6_product_cdf"], dtype=float)
        mf6_product_A = np.array(data["mf6_product_A"], dtype=float)
        mf6_product_Q_eV = np.array(data["mf6_product_Q_eV"], dtype=float)
        mf6_product_source = np.array(data["mf6_product_source"]).astype(str)
        mf6_product_weight_source = np.array(data["mf6_product_weight_source"]).astype(str)
        mf6_product_grid_offsets = np.array(data["mf6_product_grid_offsets"], dtype=np.int64)
        mf6_product_grid_lengths = np.array(data["mf6_product_grid_lengths"], dtype=np.int64)
        mf6_product_grid = np.array(data["mf6_product_grid"], dtype=float)
        mf6_product_grid_cdf = np.array(data["mf6_product_grid_cdf"], dtype=float)
    else:
        mf6_has = np.zeros(len(channel_mt), dtype=bool)

    isotope_data_by_label = {}
    for isotope in sorted(set(channel_target_isotope)):
        isotope = str(isotope)
        if isotope not in material["isotopes"]:
            raise KeyError(
                f"Kernel requires isotope {isotope}, but it is not loaded in material."
            )
        isotope_data_by_label[isotope] = material["isotopes"][isotope]["data"]

    kernel = {
        "kind": metadata.get("kind", "material_reaction_kernel_v1"),
        "flat_storage": True,
        "material_name": metadata.get("material_name", material.get("name", "unknown material")),
        "material_kernel_label": metadata.get("material_kernel_label", ""),
        "material_fingerprint": metadata.get("material_fingerprint", ""),
        "source_path": str(path),
        "E_min_eV": float(metadata["E_min_eV"]),
        "E_max_eV": float(metadata["E_max_eV"]),
        "bins_per_decade": int(metadata["bins_per_decade"]),
        "build_neutron_multiplicity": bool(metadata.get("build_neutron_multiplicity", has_multiplicity_arrays)),
        "build_residual_products": bool(metadata.get("build_residual_products", has_residual_arrays)),
        "build_mf4_elastic": bool(metadata.get("build_mf4_elastic", False)),
        "mf4_mu_grid_count": int(metadata.get("mf4_mu_grid_count", len(elastic_mu_grid))),
        "build_mf6_neutrons": bool(metadata.get("build_mf6_neutrons", has_mf6_arrays)),
        "mf6_mu_grid_count": int(metadata.get("mf6_mu_grid_count", 801)),
        "energy_edges_eV": energy_edges_eV,
        "energy_centers_eV": energy_centers_eV,
        "isotope_data_by_label": isotope_data_by_label,

        "bin_index": bin_index,
        "bin_energy_eV": bin_energy_eV,
        "bin_E_low_eV": bin_E_low_eV,
        "bin_E_high_eV": bin_E_high_eV,
        "bin_Sigma_total_1_per_m": bin_Sigma_total,
        "bin_mean_free_path_m": bin_mean_free_path,
        "bin_n_channels": bin_n_channels,
        "bin_channel_offsets": bin_channel_offsets,

        "channel_target_isotope": channel_target_isotope,
        "channel_bin_index": channel_bin_index,
        "channel_mt": channel_mt,
        "channel_sigma_barns": channel_sigma_barns,
        "channel_sigma_m2": channel_sigma_m2,
        "channel_number_density_m3": channel_number_density_m3,
        "channel_macro_xs_1_per_m": channel_macro_xs_1_per_m,
        "channel_probability": channel_probability,
        "channel_cdf": channel_cdf,

        "elastic_has_mf4_kernel": elastic_has,
        "elastic_LCT": elastic_LCT,
        "elastic_A": elastic_A,
        "elastic_angle_source": elastic_angle_source,
        "elastic_angle_frame": elastic_angle_frame,
        "elastic_mu_grid": elastic_mu_grid,
        "elastic_mu_cdf": elastic_mu_cdf,

        "has_multiplicity_arrays": has_multiplicity_arrays,
        "has_residual_arrays": has_residual_arrays,
        "has_mf6_arrays": has_mf6_arrays,
    }

    if has_multiplicity_arrays:
        kernel.update({
            "n_out_expected": n_out_expected,
            "n_out_integer_rule": n_out_integer_rule,
            "n_out_source": n_out_source,
            "n_out_mt_count": n_out_mt_count,
            "n_out_mf6_total_yield": n_out_mf6_total_yield,
            "n_out_mf6_product_count": n_out_mf6_product_count,
        })

    if has_residual_arrays:
        kernel.update({
            "residual_product_kernel": residual_product_kernel,
            "residual_product": residual_product,
            "residual_product_Z": residual_product_Z,
            "residual_product_A": residual_product_A,
            "product_note": product_note,
            "product_state": product_state,
            "product_state_source": product_state_source,
            "product_branch_probability": product_branch_probability,
            "product_branch_total_xs": product_branch_total_xs,
            "residual_has_mf10_branch_kernel": residual_has_mf10_branch_kernel,
            "residual_branch_offsets": residual_branch_offsets,
            "residual_branch_counts": residual_branch_counts,
            "residual_branch_product": residual_branch_product,
            "residual_branch_product_Z": residual_branch_product_Z,
            "residual_branch_product_A": residual_branch_product_A,
            "residual_branch_product_state": residual_branch_product_state,
            "residual_branch_mf10_xs": residual_branch_mf10_xs,
            "residual_branch_probability": residual_branch_probability,
            "residual_branch_cdf": residual_branch_cdf,
            "residual_branch_total_xs": residual_branch_total_xs,
        })

    if has_mf6_arrays:
        kernel.update({
            "mf6_has_kernel": mf6_has,
            "mf6_product_offsets": mf6_product_offsets,
            "mf6_product_counts": mf6_product_counts,
            "mf6_total_neutron_product_count": mf6_total_neutron_product_count,
            "mf6_missing_product_count": mf6_missing_product_count,
            "mf6_product_index": mf6_product_index,
            "mf6_product_law": mf6_product_law,
            "mf6_product_sample_kind": mf6_product_sample_kind,
            "mf6_product_yield": mf6_product_yield,
            "mf6_product_probability": mf6_product_probability,
            "mf6_product_cdf": mf6_product_cdf,
            "mf6_product_A": mf6_product_A,
            "mf6_product_Q_eV": mf6_product_Q_eV,
            "mf6_product_source": mf6_product_source,
            "mf6_product_weight_source": mf6_product_weight_source,
            "mf6_product_grid_offsets": mf6_product_grid_offsets,
            "mf6_product_grid_lengths": mf6_product_grid_lengths,
            "mf6_product_grid": mf6_product_grid,
            "mf6_product_grid_cdf": mf6_product_grid_cdf,
        })

    kernel["bins"] = FlatKernelBinsView(kernel)

    print("Loaded flat transport kernel:", path)
    return kernel
