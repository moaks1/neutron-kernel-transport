"""MF4 angular sampling and elastic two-body kinematics."""

import numpy as np

from nuclear_data import (
    mt_index,
    get_target_A_from_data,
    sample_from_tabulated_pdf,
    legendre_pdf_from_coeffs,
)


def mf4_info(data, mt):
    """Return basic MF4 info for one MT."""
    i = mt_index(data, mt)
    if i is None or "mf4_ltt" not in data.files:
        return None
    return {
        "mt": int(mt),
        "index": i,
        "LTT": int(data["mf4_ltt"][i]),
        "LI": int(data["mf4_li"][i]),
        "LCT": int(data["mf4_lct"][i]),
        "legendre_count": int(data["mf4_leg_mt_counts"][i]) if "mf4_leg_mt_counts" in data.files else 0,
        "tabulated_count": int(data["mf4_tab_mt_counts"][i]) if "mf4_tab_mt_counts" in data.files else 0,
    }


def mf4_legendre_coeffs_at(data, mt, energy_eV):
    """Return interpolated MF4 Legendre coefficients."""
    i = mt_index(data, mt)
    if i is None or "mf4_leg_mt_counts" not in data.files:
        return None

    offset = int(data["mf4_leg_mt_offsets"][i])
    count = int(data["mf4_leg_mt_counts"][i])
    if count <= 0:
        return None

    Ein = np.array(data["mf4_leg_E"][offset : offset + count], dtype=float)
    energy_eV = float(energy_eV)

    def get_one_table(local_j):
        global_j = offset + int(local_j)
        coeff_offset = int(data["mf4_leg_coeff_offsets"][global_j])
        coeff_length = int(data["mf4_leg_coeff_lengths"][global_j])
        if coeff_length <= 0:
            return None
        return np.array(data["mf4_leg_coeff"][coeff_offset : coeff_offset + coeff_length], dtype=float)

    if energy_eV <= Ein[0]:
        return get_one_table(0)
    if energy_eV >= Ein[-1]:
        return get_one_table(count - 1)

    hi = int(np.searchsorted(Ein, energy_eV))
    lo = hi - 1
    E0 = float(Ein[lo])
    E1 = float(Ein[hi])
    c0 = get_one_table(lo)
    c1 = get_one_table(hi)
    if c0 is None or c1 is None:
        return c0
    if len(c0) != len(c1):
        return c0 if abs(energy_eV - E0) < abs(E1 - energy_eV) else c1
    if E1 <= E0:
        return c0
    w = (energy_eV - E0) / (E1 - E0)
    return (1.0 - w) * c0 + w * c1


def sample_mf4_legendre_mu(data, mt, energy_eV, n_mu_grid=801):
    """Sample mu from MF4 Legendre coefficients."""
    coeffs = mf4_legendre_coeffs_at(data, mt, energy_eV)
    if coeffs is None:
        return None
    mu_grid = np.linspace(-1.0, 1.0, int(n_mu_grid))
    pdf = legendre_pdf_from_coeffs(mu_grid, coeffs)
    mu = sample_from_tabulated_pdf(mu_grid, pdf)
    if mu is None:
        return None
    return float(np.clip(mu, -1.0, 1.0))


def mf4_tabulated_pdf_at(data, mt, energy_eV):
    """Return mu grid and PDF from MF4 tabulated angular distributions."""
    i = mt_index(data, mt)
    if i is None or "mf4_tab_mt_counts" not in data.files:
        return None, None

    offset = int(data["mf4_tab_mt_offsets"][i])
    count = int(data["mf4_tab_mt_counts"][i])
    if count <= 0:
        return None, None

    Ein = np.array(data["mf4_tab_E"][offset : offset + count], dtype=float)
    energy_eV = float(energy_eV)

    def get_one_table(local_j):
        global_j = offset + int(local_j)
        mu_offset = int(data["mf4_tab_mu_offsets"][global_j])
        mu_length = int(data["mf4_tab_mu_lengths"][global_j])
        if mu_length <= 1:
            return None, None
        mu = np.array(data["mf4_tab_mu"][mu_offset : mu_offset + mu_length], dtype=float)
        pdf = np.array(data["mf4_tab_pdf"][mu_offset : mu_offset + mu_length], dtype=float)
        good = np.isfinite(mu) & np.isfinite(pdf)
        mu = mu[good]
        pdf = pdf[good]
        if len(mu) <= 1:
            return None, None
        order = np.argsort(mu)
        return mu[order], np.maximum(pdf[order], 0.0)

    if energy_eV <= Ein[0]:
        return get_one_table(0)
    if energy_eV >= Ein[-1]:
        return get_one_table(count - 1)

    hi = int(np.searchsorted(Ein, energy_eV))
    lo = hi - 1
    E0 = float(Ein[lo])
    E1 = float(Ein[hi])
    mu0, pdf0 = get_one_table(lo)
    mu1, pdf1 = get_one_table(hi)
    if mu0 is None or mu1 is None:
        return mu0, pdf0
    if E1 <= E0:
        return mu0, pdf0
    w = (energy_eV - E0) / (E1 - E0)
    if len(mu0) == len(mu1) and np.allclose(mu0, mu1):
        return mu0, (1.0 - w) * pdf0 + w * pdf1
    return (mu0, pdf0) if w < 0.5 else (mu1, pdf1)


def sample_mf4_tabulated_mu(data, mt, energy_eV):
    """Sample scattering cosine mu from MF4 tabulated angular data."""
    mu_grid, pdf = mf4_tabulated_pdf_at(data, mt, energy_eV)
    if mu_grid is None or pdf is None:
        return None
    mu = sample_from_tabulated_pdf(mu_grid, pdf)
    if mu is None:
        return None
    return float(np.clip(mu, -1.0, 1.0))


def sample_mf4_mu(data, mt, energy_eV):
    """Sample scattering cosine from MF4."""
    info = mf4_info(data, mt)
    if info is None:
        return None

    LTT = info["LTT"]
    LI = info["LI"]
    LCT = info["LCT"]

    if LI == 1:
        mu = 2.0 * np.random.random() - 1.0
        return {
            "mu": float(mu),
            "LCT": LCT,
            "frame": "center-of-mass" if LCT == 2 else "lab",
            "source": "MF4-isotropic-LI",
        }

    mu = None
    source = None

    if LTT in [1, 3] and info["legendre_count"] > 0:
        mu = sample_mf4_legendre_mu(data, mt, energy_eV)
        source = "MF4-Legendre"

    if mu is None and LTT in [2, 3] and info["tabulated_count"] > 0:
        mu = sample_mf4_tabulated_mu(data, mt, energy_eV)
        source = "MF4-tabulated"

    if mu is None:
        return None

    return {
        "mu": float(mu),
        "LCT": LCT,
        "frame": "center-of-mass" if LCT == 2 else "lab",
        "source": source,
    }


def elastic_energy_from_mu_cm(incoming_energy_eV, mu_cm, A):
    """Compute outgoing neutron lab energy for elastic scattering."""
    E = float(incoming_energy_eV)
    mu_cm = float(np.clip(mu_cm, -1.0, 1.0))
    A = float(A)
    ratio = (A**2 + 2.0 * A * mu_cm + 1.0) / ((A + 1.0) ** 2)
    return max(float(E * ratio), 0.0)


def elastic_mu_lab_from_mu_cm(mu_cm, A):
    """Convert elastic center-of-mass cosine to lab scattering cosine."""
    mu_cm = float(np.clip(mu_cm, -1.0, 1.0))
    A = float(A)
    denom = np.sqrt(A**2 + 2.0 * A * mu_cm + 1.0)
    if denom <= 0.0:
        return 1.0
    mu_lab = (1.0 + A * mu_cm) / denom
    return float(np.clip(mu_lab, -1.0, 1.0))


def rotate_direction_2d(direction, theta_rad):
    """Rotate a 2D direction vector by theta_rad."""
    direction = np.array(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("Direction vector cannot be zero.")
    direction = direction / norm
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]])
    new_direction = R @ direction
    return new_direction / np.linalg.norm(new_direction)


def sample_elastic_mf4_update(data, incoming_energy_eV):
    """Use MF4 angle data plus elastic two-body kinematics for MT=2."""
    A = get_target_A_from_data(data)
    angle_row = sample_mf4_mu(data=data, mt=2, energy_eV=incoming_energy_eV)
    if angle_row is None:
        return None

    mu = float(angle_row["mu"])
    LCT = int(angle_row["LCT"])

    if LCT == 2:
        mu_cm = mu
        outgoing_energy = elastic_energy_from_mu_cm(incoming_energy_eV, mu_cm, A)
        mu_lab = elastic_mu_lab_from_mu_cm(mu_cm, A)
        theta_lab = np.arccos(np.clip(mu_lab, -1.0, 1.0))
        return {
            "energy_eV": outgoing_energy,
            "mu_cm": mu_cm,
            "mu_lab": mu_lab,
            "theta_lab_rad": theta_lab,
            "theta_lab_deg": np.degrees(theta_lab),
            "angle_source": angle_row["source"],
            "angle_frame": angle_row["frame"],
            "energy_update_source": "MF4-elastic-two-body",
        }

    if LCT == 1:
        mu_lab = mu
        theta_lab = np.arccos(np.clip(mu_lab, -1.0, 1.0))
        return {
            "energy_eV": incoming_energy_eV,
            "mu_cm": None,
            "mu_lab": mu_lab,
            "theta_lab_rad": theta_lab,
            "theta_lab_deg": np.degrees(theta_lab),
            "angle_source": angle_row["source"],
            "angle_frame": angle_row["frame"],
            "energy_update_source": "MF4-lab-angle-energy-unchanged",
        }

    return None
