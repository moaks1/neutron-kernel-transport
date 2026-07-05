"""
MF6 outgoing-neutron energy sampling utilities.

Supports:
    - LAW=1 outgoing-energy distributions
    - LAW=2 two-body angular distributions + kinematics
"""

import numpy as np

from nuclear_data import (
    mt_index,
    get_mt_curve,
    get_target_A_from_data,
    interp_zero_outside,
    sample_from_tabulated_pdf,
    legendre_pdf_from_coeffs,
    choose_weighted,
)


def mf6_product_indices_for_mt(data, mt):
    """Return MF6 product-block indices for one MT."""
    needed = [
        "mf6_mt_prod_offsets",
        "mf6_mt_prod_counts",
        "mf6_prod_ZAP",
        "mf6_prod_LAW",
    ]
    for key in needed:
        if key not in data.files:
            return []

    i = mt_index(data, mt)
    if i is None:
        return []

    offset = int(data["mf6_mt_prod_offsets"][i])
    count = int(data["mf6_mt_prod_counts"][i])

    return list(range(offset, offset + count))


def mf6_neutron_product_indices(data, mt):
    """Return MF6 product-block indices for outgoing neutron product, usually ZAP=1."""
    products = []
    for p in mf6_product_indices_for_mt(data, mt):
        zap = int(round(float(data["mf6_prod_ZAP"][p])))
        if zap == 1:
            products.append(p)
    return products


def mf6_yield_at_product_index(data, product_index, energy_eV):
    """Interpolate MF6 product yield at incident energy."""
    needed = ["mf6_y_E_offsets", "mf6_y_E_lengths", "mf6_y_E", "mf6_y"]
    for key in needed:
        if key not in data.files:
            return 0.0

    p = int(product_index)
    offset = int(data["mf6_y_E_offsets"][p])
    length = int(data["mf6_y_E_lengths"][p])
    if length <= 0:
        return 0.0

    E = np.array(data["mf6_y_E"][offset : offset + length], dtype=float)
    Y = np.array(data["mf6_y"][offset : offset + length], dtype=float)

    return interp_zero_outside(energy_eV, E, Y)


def mf6_neutron_product_summary(data, mt, energy_eV=None):
    """Return readable summary rows for outgoing-neutron MF6 product blocks."""
    rows = []
    for p in mf6_neutron_product_indices(data, mt):
        row = {
            "mt": int(mt),
            "product_index": int(p),
            "zap": float(data["mf6_prod_ZAP"][p]),
            "law": int(data["mf6_prod_LAW"][p]),
            "has_law1": bool(data["mf6_prod_law1_has"][p]) if "mf6_prod_law1_has" in data.files else False,
            "has_law2": bool(data["mf6_prod_law2_has"][p]) if "mf6_prod_law2_has" in data.files else False,
            "has_law6": bool(data["mf6_prod_law6_has"][p]) if "mf6_prod_law6_has" in data.files else False,
        }
        if energy_eV is not None:
            row["yield_at_E"] = mf6_yield_at_product_index(data, p, energy_eV)
        rows.append(row)
    return rows


def mf6_law1_energy_pdf_at(data, product_index, energy_eV):
    """Return outgoing neutron energy grid and PDF from MF6 LAW=1."""
    p = int(product_index)

    if "mf6_prod_law1_has" not in data.files:
        return None, None
    if int(data["mf6_prod_law1_has"][p]) == 0:
        return None, None

    needed = [
        "mf6_law1_Ein_offsets",
        "mf6_law1_Ein_lengths",
        "mf6_law1_Ein",
        "mf6_law1_NA",
        "mf6_law1_Ep_offsets",
        "mf6_law1_Ep_lengths",
        "mf6_law1_Ep",
        "mf6_law1_b_offsets",
        "mf6_law1_b_lengths",
        "mf6_law1_b",
    ]
    for key in needed:
        if key not in data.files:
            return None, None

    ein_offset = int(data["mf6_law1_Ein_offsets"][p])
    ein_count = int(data["mf6_law1_Ein_lengths"][p])
    if ein_count <= 0:
        return None, None

    Ein = np.array(data["mf6_law1_Ein"][ein_offset : ein_offset + ein_count], dtype=float)
    energy_eV = float(energy_eV)

    def get_one_incident_table(local_j):
        global_j = ein_offset + int(local_j)

        ep_offset = int(data["mf6_law1_Ep_offsets"][global_j])
        ep_length = int(data["mf6_law1_Ep_lengths"][global_j])
        b_offset = int(data["mf6_law1_b_offsets"][global_j])
        b_length = int(data["mf6_law1_b_lengths"][global_j])
        if ep_length <= 0 or b_length <= 0:
            return None, None

        Ep = np.array(data["mf6_law1_Ep"][ep_offset : ep_offset + ep_length], dtype=float)
        b = np.array(data["mf6_law1_b"][b_offset : b_offset + b_length], dtype=float)
        NA = int(data["mf6_law1_NA"][global_j])
        stride = NA + 1

        if NA > 0 and b_length >= ep_length * stride:
            pdf = b[0 : ep_length * stride : stride]
        else:
            pdf = b[:ep_length]

        n = min(len(Ep), len(pdf))
        if n <= 1:
            return None, None

        Ep = Ep[:n]
        pdf = pdf[:n]
        good = np.isfinite(Ep) & np.isfinite(pdf) & (Ep >= 0.0)
        Ep = Ep[good]
        pdf = pdf[good]
        if len(Ep) <= 1:
            return None, None

        order = np.argsort(Ep)
        return Ep[order], pdf[order]

    if energy_eV <= Ein[0]:
        return get_one_incident_table(0)
    if energy_eV >= Ein[-1]:
        return get_one_incident_table(ein_count - 1)

    hi = int(np.searchsorted(Ein, energy_eV))
    lo = hi - 1
    E0 = float(Ein[lo])
    E1 = float(Ein[hi])
    Ep0, pdf0 = get_one_incident_table(lo)
    Ep1, pdf1 = get_one_incident_table(hi)
    if Ep0 is None or Ep1 is None:
        return Ep0, pdf0
    if E1 <= E0:
        return Ep0, pdf0

    w = (energy_eV - E0) / (E1 - E0)
    if len(Ep0) == len(Ep1) and np.allclose(Ep0, Ep1):
        pdf = (1.0 - w) * pdf0 + w * pdf1
        return Ep0, pdf
    if w < 0.5:
        return Ep0, pdf0
    return Ep1, pdf1


def get_target_A_from_threshold_curve(data, mt, A=None):
    """Estimate excitation energy from MF3 threshold for MT=51-90."""
    if A is None:
        A = get_target_A_from_data(data)
    E, xs = get_mt_curve(data, mt)
    if len(E) == 0:
        return 0.0
    positive = np.where(xs > 0.0)[0]
    if len(positive) == 0:
        return 0.0
    threshold = float(E[int(positive[0])])
    if threshold <= 0.0:
        return 0.0
    return float(threshold * A / (A + 1.0))


def mf6_law2_coeffs_at(data, product_index, energy_eV):
    """Return interpolated MF6 LAW=2 Legendre coefficients."""
    p = int(product_index)
    if "mf6_prod_law2_has" not in data.files:
        return None
    if int(data["mf6_prod_law2_has"][p]) == 0:
        return None

    needed = [
        "mf6_law2_Ein_offsets",
        "mf6_law2_Ein_lengths",
        "mf6_law2_Ein",
        "mf6_law2_LANG",
        "mf6_law2_data_offsets",
        "mf6_law2_data_lengths",
        "mf6_law2_data",
    ]
    for key in needed:
        if key not in data.files:
            return None

    ein_offset = int(data["mf6_law2_Ein_offsets"][p])
    ein_count = int(data["mf6_law2_Ein_lengths"][p])
    if ein_count <= 0:
        return None

    Ein = np.array(data["mf6_law2_Ein"][ein_offset : ein_offset + ein_count], dtype=float)
    energy_eV = float(energy_eV)

    def get_one_table(local_j):
        global_j = ein_offset + int(local_j)
        LANG = int(data["mf6_law2_LANG"][global_j])
        if LANG != 0:
            return None
        offset = int(data["mf6_law2_data_offsets"][global_j])
        length = int(data["mf6_law2_data_lengths"][global_j])
        if length <= 0:
            return None
        return np.array(data["mf6_law2_data"][offset : offset + length], dtype=float)

    if energy_eV <= Ein[0]:
        return get_one_table(0)
    if energy_eV >= Ein[-1]:
        return get_one_table(ein_count - 1)

    hi = int(np.searchsorted(Ein, energy_eV))
    lo = hi - 1
    E0 = float(Ein[lo])
    E1 = float(Ein[hi])
    c0 = get_one_table(lo)
    c1 = get_one_table(hi)
    if c0 is None or c1 is None:
        return c0
    if E1 <= E0:
        return c0
    w = (energy_eV - E0) / (E1 - E0)
    if len(c0) != len(c1):
        return c0 if w < 0.5 else c1
    return (1.0 - w) * c0 + w * c1


def sample_mf6_law2_mu(data, product_index, energy_eV, n_mu_grid=801):
    """Sample center-of-mass cosine mu from MF6 LAW=2."""
    coeffs = mf6_law2_coeffs_at(data, product_index, energy_eV)
    if coeffs is None:
        return None
    mu_grid = np.linspace(-1.0, 1.0, int(n_mu_grid))
    pdf = legendre_pdf_from_coeffs(mu_grid, coeffs)
    mu = sample_from_tabulated_pdf(mu_grid, pdf)
    if mu is None:
        return None
    return float(np.clip(mu, -1.0, 1.0))


def two_body_outgoing_neutron_energy(data, mt, incoming_energy_eV, mu_cm, A=None):
    """Compute outgoing neutron lab energy for two-body inelastic scattering."""
    if A is None:
        A = get_target_A_from_data(data)
    E = float(incoming_energy_eV)
    mu_cm = float(np.clip(mu_cm, -1.0, 1.0))
    A = float(A)

    Ex = get_target_A_from_threshold_curve(data, mt, A=A)
    Q = -Ex
    K_cm_out = (A / (A + 1.0)) * E + Q
    if K_cm_out <= 0.0:
        return 0.0

    E_neutron_cm = (A / (A + 1.0)) * K_cm_out
    if E_neutron_cm <= 0.0:
        return 0.0

    term_cm_motion = E / ((A + 1.0) ** 2)
    term_cross = 2.0 * np.sqrt(max(E * E_neutron_cm, 0.0)) / (A + 1.0) * mu_cm
    E_out = term_cm_motion + E_neutron_cm + term_cross
    return max(float(E_out), 0.0)


def sample_mf6_outgoing_neutron_energy(data, mt, energy_eV):
    """Sample one outgoing neutron energy from MF6 LAW=1 or LAW=2."""
    mt = int(mt)
    energy_eV = float(energy_eV)
    neutron_products = mf6_neutron_product_indices(data, mt)
    if len(neutron_products) == 0:
        return None

    choices = []
    for p in neutron_products:
        law = int(data["mf6_prod_LAW"][p])
        y = mf6_yield_at_product_index(data, p, energy_eV)
        choices.append({"product_index": int(p), "law": law, "yield": max(float(y), 0.0)})

    usable = [row for row in choices if row["yield"] > 0.0]
    if len(usable) == 0:
        usable = choices
    if len(usable) == 0:
        return None

    chosen = usable[0] if len(usable) == 1 else choose_weighted(usable, "yield")
    if chosen is None:
        return None

    p = int(chosen["product_index"])
    law = int(chosen["law"])

    if law == 1:
        Ep, pdf = mf6_law1_energy_pdf_at(data, p, energy_eV)
        if Ep is None or pdf is None:
            return None
        sampled_Ep = sample_from_tabulated_pdf(Ep, pdf)
        if sampled_Ep is None:
            return None
        return {
            "mt": mt,
            "incident_energy_eV": energy_eV,
            "energy_eV": max(float(sampled_Ep), 0.0),
            "product_index": p,
            "law": 1,
            "yield_at_E": mf6_yield_at_product_index(data, p, energy_eV),
            "source": "MF6-LAW1",
        }

    if law == 2:
        mu_cm = sample_mf6_law2_mu(data, p, energy_eV)
        if mu_cm is None:
            return None
        E_out = two_body_outgoing_neutron_energy(data, mt, energy_eV, mu_cm)
        return {
            "mt": mt,
            "incident_energy_eV": energy_eV,
            "energy_eV": E_out,
            "product_index": p,
            "law": 2,
            "yield_at_E": mf6_yield_at_product_index(data, p, energy_eV),
            "mu_cm": mu_cm,
            "source": "MF6-LAW2-two-body",
        }

    return None
