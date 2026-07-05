"""
Nuclear-data, material, and reaction-list utilities for the analog
neutron Monte Carlo transport notebook.

This file keeps the functions from the learning notebook together:
    - isotope labels and file names
    - MF3 curve grabbing
    - microscopic/macroscopic cross sections
    - MT names and emitted-particle logic
    - summary-channel cleaning
    - material composition loading
    - reaction probabilities and reaction chooser
"""

import os
import json
import importlib.util
import numpy as np

AVOGADRO = 6.02214076e23
BARN_TO_M2 = 1.0e-28
EV_TO_J = 1.602176634e-19
M_NEUTRON = 1.67492749804e-27
CI_TO_BQ = 3.7e10
BQ_TO_CI = 1.0 / CI_TO_BQ

ELEMENT_SYMBOLS = [
    None,
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

SYMBOL_TO_Z = {}
for Z, symbol in enumerate(ELEMENT_SYMBOLS):
    if symbol is not None:
        SYMBOL_TO_Z[symbol] = Z


def make_isotope_label(symbol, A, metastable=""):
    """Make a label like O-18, Mg-24, or F-26m."""
    return f"{symbol}-{int(A)}{metastable}"


def symbol_from_Z(Z):
    """Convert proton number Z to element symbol."""
    Z = int(Z)
    if Z <= 0 or Z >= len(ELEMENT_SYMBOLS):
        return None
    return ELEMENT_SYMBOLS[Z]


def split_isotope_label(isotope_label):
    """Split isotope label like Mg-24 or F-26m into symbol, A, metastable."""
    symbol, A_text = isotope_label.split("-")
    if A_text.endswith("m"):
        A = int(A_text[:-1])
        metastable = "m"
    else:
        A = int(A_text)
        metastable = ""
    return symbol, A, metastable


def isotope_label_to_neutron_filename(isotope_label):
    """Convert Mg-24 -> n-012_Mg_024.npz."""
    symbol, A, metastable = split_isotope_label(isotope_label)
    if metastable:
        A_text = f"{A:03d}{metastable}"
    else:
        A_text = f"{A:03d}"
    Z = SYMBOL_TO_Z[symbol]
    return f"n-{Z:03d}_{symbol}_{A_text}.npz"


def isotope_label_to_decay_filename_candidates(isotope_label):
    """Return possible decay file names for an isotope label."""
    symbol, A, metastable = split_isotope_label(isotope_label)
    Z = SYMBOL_TO_Z[symbol]
    return [
        f"dec-{Z:03d}_{symbol}_{A}{metastable}.npz",
        f"dec-{Z:03d}_{symbol}_{A:03d}{metastable}.npz",
    ]


def load_neutron_npz(xs_dir, isotope_label):
    """Load one neutron cross-section NPZ file. Returns None if missing."""
    filename = isotope_label_to_neutron_filename(isotope_label)
    path = os.path.join(xs_dir, filename)
    if not os.path.exists(path):
        print("Missing neutron file:", path)
        return None
    return np.load(path, allow_pickle=True)


def find_decay_file(decay_dir, isotope_label):
    """Find the decay NPZ file for one isotope."""
    for filename in isotope_label_to_decay_filename_candidates(isotope_label):
        path = os.path.join(decay_dir, filename)
        if os.path.exists(path):
            return path, filename
    return None, None


def mt_index(data, mt):
    """Return index of MT inside data['mt_ids']."""
    if "mt_ids" not in data.files:
        return None
    matches = np.where(data["mt_ids"] == int(mt))[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def get_target_info_from_data(data):
    """Get target isotope information from meta_json."""
    if "meta_json" not in data.files:
        raise ValueError("No meta_json found in this NPZ file.")

    meta = json.loads(str(data["meta_json"]))
    nuclide = meta["nuclide"]

    Z = int(nuclide["Z"])
    A = int(nuclide["A"])

    if "symbol" in nuclide:
        symbol = nuclide["symbol"]
    elif "element" in nuclide:
        symbol = nuclide["element"]
    else:
        symbol = symbol_from_Z(Z)

    if symbol is None:
        raise ValueError(f"Could not determine element symbol for Z={Z}")

    return {
        "Z": Z,
        "A": A,
        "symbol": symbol,
        "label": make_isotope_label(symbol, A),
    }


def get_target_A_from_data(data):
    """Return target mass number A from data meta_json."""
    return int(get_target_info_from_data(data)["A"])


def get_mt_curve(data, mt):
    """Get energy grid and cross section curve for one MT reaction."""
    needed = ["mt_ids", "mt_offsets", "mt_lengths", "E_concat", "xs_concat"]
    for key in needed:
        if key not in data.files:
            return np.array([]), np.array([])

    i = mt_index(data, mt)
    if i is None:
        return np.array([]), np.array([])

    offset = int(data["mt_offsets"][i])
    length = int(data["mt_lengths"][i])

    if length <= 0:
        return np.array([]), np.array([])

    E = np.array(data["E_concat"][offset : offset + length], dtype=float)
    xs = np.array(data["xs_concat"][offset : offset + length], dtype=float)

    return E, xs


# Notebook alias that may be convenient.
def get_mt_data(data, mt):
    return get_mt_curve(data, mt)


def interp_zero_outside(x, xp, fp):
    """Interpolate y(x), but return 0 outside the tabulated range."""
    xp = np.array(xp, dtype=float)
    fp = np.array(fp, dtype=float)
    if len(xp) == 0:
        return 0.0
    x = float(x)
    if x < xp[0] or x > xp[-1]:
        return 0.0
    return float(np.interp(x, xp, fp))


def get_cross_section_at_energy(data, mt, energy_eV):
    """Get sigma_MT(E) in barns for one reaction channel."""
    E, xs = get_mt_curve(data, mt)
    if len(E) == 0:
        return 0.0
    return interp_zero_outside(energy_eV, E, xs)


def get_open_reactions(data, energy_eV):
    """Find every MT channel with nonzero cross section at energy_eV."""
    if "mt_ids" not in data.files:
        return []
    open_reactions = []
    for mt in data["mt_ids"]:
        mt = int(mt)
        sigma_barns = get_cross_section_at_energy(data, mt, energy_eV)
        if sigma_barns > 0.0:
            open_reactions.append({"mt": mt, "sigma_barns": sigma_barns})
    return open_reactions


def microscopic_to_macroscopic_xs(sigma_barns, number_density_m3):
    """Convert microscopic cross section in barns to macroscopic XS in 1/m."""
    sigma_m2 = float(sigma_barns) * BARN_TO_M2
    return float(number_density_m3) * sigma_m2


def get_open_reactions_with_macro_xs(data, energy_eV, number_density_m3):
    """Find open MT channels and add macroscopic cross sections."""
    rows = get_open_reactions(data, energy_eV)
    for row in rows:
        sigma_barns = row["sigma_barns"]
        sigma_m2 = sigma_barns * BARN_TO_M2
        row["sigma_m2"] = sigma_m2
        row["number_density_m3"] = float(number_density_m3)
        row["macro_xs_1_per_m"] = float(number_density_m3) * sigma_m2
    return rows


def get_open_reactions_with_macro_xs_sorted(data, energy_eV, number_density_m3):
    rows = get_open_reactions_with_macro_xs(data, energy_eV, number_density_m3)
    rows.sort(key=lambda row: row["macro_xs_1_per_m"], reverse=True)
    return rows


def ordinal(n):
    """Convert 1 -> 1st, 2 -> 2nd, etc."""
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    elif n % 10 == 1:
        suffix = "st"
    elif n % 10 == 2:
        suffix = "nd"
    elif n % 10 == 3:
        suffix = "rd"
    else:
        suffix = "th"
    return f"{n}{suffix}"


def inelastic_level_name(mt):
    """Return readable name for ENDF detailed inelastic channels."""
    mt = int(mt)
    if 51 <= mt <= 90:
        level = mt - 50
        return f"(n,n{level}) inelastic to {ordinal(level)} excited state"
    if mt == 91:
        return "(n,nc) continuum inelastic"
    return None


def mt_name(mt):
    """Return simple readable name for common ENDF MT reaction channels."""
    mt = int(mt)

    detailed = inelastic_level_name(mt)
    if detailed is not None:
        return detailed

    names = {
        1: "total",
        2: "elastic scattering",
        3: "nonelastic total",
        4: "total inelastic scattering",
        5: "(n,anything)",
        10: "total continuum reaction",
        11: "(n,2nd)",
        16: "(n,2n)",
        17: "(n,3n)",
        22: "(n,nalpha)",
        23: "(n,n3alpha)",
        24: "(n,2nalpha)",
        25: "(n,3nalpha)",
        28: "(n,np)",
        29: "(n,n2alpha)",
        30: "(n,2n2alpha)",
        32: "(n,nd)",
        33: "(n,nt)",
        34: "(n,nHe3)",
        35: "(n,nd2alpha)",
        36: "(n,nt2alpha)",
        37: "(n,4n)",
        41: "(n,2np)",
        42: "(n,3np)",
        44: "(n,n2p)",
        45: "(n,npalpha)",
        101: "neutron disappearance",
        102: "(n,gamma)",
        103: "(n,p)",
        104: "(n,d)",
        105: "(n,t)",
        106: "(n,He3)",
        107: "(n,alpha)",
        108: "(n,2alpha)",
        109: "(n,3alpha)",
        111: "(n,2p)",
        112: "(n,palpha)",
        113: "(n,t2alpha)",
        114: "(n,d2alpha)",
        115: "(n,pd)",
        116: "(n,pt)",
        117: "(n,dalpha)",
        151: "resonance parameters",
        152: "(n,5n)",
        153: "(n,6n)",
        154: "(n,2nt)",
        155: "(n,talpha)",
        156: "(n,4np)",
        157: "(n,3nd)",
        158: "(n,ndalpha)",
        159: "(n,2npalpha)",
        160: "(n,7n)",
        161: "(n,8n)",
        162: "(n,5np)",
        163: "(n,6np)",
        164: "(n,7np)",
        165: "(n,4nalpha)",
        166: "(n,5nalpha)",
        167: "(n,6nalpha)",
        168: "(n,7nalpha)",
        169: "(n,4nd)",
        170: "(n,5nd)",
        171: "(n,6nd)",
        172: "(n,3nt)",
        173: "(n,4nt)",
        174: "(n,5nt)",
        175: "(n,6nt)",
        176: "(n,2nHe3)",
        177: "(n,3nHe3)",
        178: "(n,4nHe3)",
        179: "(n,3n2p)",
        180: "(n,3n2alpha)",
        181: "(n,3npalpha)",
        182: "(n,dt)",
        183: "(n,npd)",
        184: "(n,npt)",
        185: "(n,ndt)",
        186: "(n,npHe3)",
        187: "(n,ndHe3)",
        188: "(n,ntHe3)",
        189: "(n,ntalpha)",
        190: "(n,2n2p)",
        191: "(n,pHe3)",
        192: "(n,dHe3)",
        193: "(n,alphaHe3)",
        194: "(n,4n2p)",
        195: "(n,4n2alpha)",
        196: "(n,4npalpha)",
        197: "(n,3p)",
        198: "(n,n3p)",
        199: "(n,3n2palpha)",
        200: "(n,5n2p)",
    }
    return names.get(mt, "unknown reaction")


MT_EMITTED = {
    2: "n", 4: "n",
    11: "2nd", 16: "2n", 17: "3n", 22: "nalpha", 23: "n3alpha",
    24: "2nalpha", 25: "3nalpha", 28: "np", 29: "n2alpha",
    30: "2n2alpha", 32: "nd", 33: "nt", 34: "nHe3",
    35: "nd2alpha", 36: "nt2alpha", 37: "4n", 41: "2np",
    42: "3np", 44: "n2p", 45: "npalpha",
    102: "", 103: "p", 104: "d", 105: "t", 106: "He3",
    107: "alpha", 108: "2alpha", 109: "3alpha",
    111: "2p", 112: "palpha", 113: "t2alpha", 114: "d2alpha",
    115: "pd", 116: "pt", 117: "dalpha",
    152: "5n", 153: "6n", 154: "2nt", 155: "talpha",
    156: "4np", 157: "3nd", 158: "ndalpha", 159: "2npalpha",
    160: "7n", 161: "8n", 162: "5np", 163: "6np", 164: "7np",
    165: "4nalpha", 166: "5nalpha", 167: "6nalpha", 168: "7nalpha",
    169: "4nd", 170: "5nd", 171: "6nd", 172: "3nt", 173: "4nt",
    174: "5nt", 175: "6nt", 176: "2nHe3", 177: "3nHe3",
    178: "4nHe3", 179: "3n2p", 180: "3n2alpha", 181: "3npalpha",
    182: "dt", 183: "npd", 184: "npt", 185: "ndt", 186: "npHe3",
    187: "ndHe3", 188: "ntHe3", 189: "ntalpha", 190: "2n2p",
    191: "pHe3", 192: "dHe3", 193: "alphaHe3", 194: "4n2p",
    195: "4n2alpha", 196: "4npalpha", 197: "3p", 198: "n3p",
    199: "3n2palpha", 200: "5n2p",
}

PARTICLE_INFO = {
    "n": {"A": 1, "Z": 0},
    "p": {"A": 1, "Z": 1},
    "d": {"A": 2, "Z": 1},
    "t": {"A": 3, "Z": 1},
    "He3": {"A": 3, "Z": 2},
    "alpha": {"A": 4, "Z": 2},
}


def emitted_string_for_mt(mt):
    """Return emitted-particle string for an MT."""
    mt = int(mt)
    if 51 <= mt <= 91:
        return "n"
    return MT_EMITTED.get(mt, None)


def parse_emitted_particles(emitted):
    """Parse emitted particle string into counts."""
    counts = {}
    if emitted is None or emitted == "":
        return counts

    i = 0
    particle_names = ["alpha", "He3", "n", "p", "d", "t"]

    while i < len(emitted):
        number_text = ""
        while i < len(emitted) and emitted[i].isdigit():
            number_text += emitted[i]
            i += 1
        count = int(number_text) if number_text else 1

        matched = False
        for particle in particle_names:
            if emitted.startswith(particle, i):
                counts[particle] = counts.get(particle, 0) + count
                i += len(particle)
                matched = True
                break
        if not matched:
            i += 1

    return counts


def count_outgoing_neutrons(mt):
    """Count outgoing neutrons represented by this MT."""
    emitted = emitted_string_for_mt(mt)
    counts = parse_emitted_particles(emitted)
    return counts.get("n", 0)




def label_from_za_and_state(za, state):
    """Convert ENDF ZA plus product state into an isotope label.

    Examples:
        ZA=11024, state=0 -> Na-24
        ZA=11024, state=1 -> Na-24m
    """
    za = int(round(float(za)))
    state = int(state)

    Z = za // 1000
    A = za - 1000 * Z
    symbol = symbol_from_Z(Z)

    if symbol is None or A <= 0:
        return None

    if state == 0:
        metastable = ""
    elif state == 1:
        metastable = "m"
    else:
        metastable = f"m{state}"

    return make_isotope_label(symbol, A, metastable)


def za_to_Z_A(za):
    """Convert ENDF ZA into Z and A."""
    za = int(round(float(za)))
    Z = za // 1000
    A = za - 1000 * Z
    return Z, A


def mf10_has_data(data):
    """Return True if the NPZ file has the basic MF10 arrays."""
    needed = [
        "mf10_mt_ids",
        "mf10_prod_offsets",
        "mf10_prod_counts",
        "mf10_prod_za",
        "mf10_prod_state",
        "mf10_tab_offsets",
        "mf10_tab_counts",
        "mf10_E_data",
        "mf10_Y_data",
    ]
    for key in needed:
        if key not in data.files:
            return False
    if len(data["mf10_mt_ids"]) == 0:
        return False
    if len(data["mf10_E_data"]) == 0:
        return False
    return True


def mf10_product_rows_at_energy(data, mt, energy_eV, product_Z=None, product_A=None):
    """Return MF10 product-state rows for one MT at one neutron energy.

    This does not choose a branch. It only reads every available MF10
    product state and interpolates its production cross section at energy_eV.
    """
    rows = []

    if energy_eV is None:
        return rows
    if not mf10_has_data(data):
        return rows

    mt = int(mt)
    energy_eV = float(energy_eV)

    matches = np.where(data["mf10_mt_ids"] == mt)[0]
    if len(matches) == 0:
        return rows

    mt_i = int(matches[0])
    product_offset = int(data["mf10_prod_offsets"][mt_i])
    product_count = int(data["mf10_prod_counts"][mt_i])

    for product_i in range(product_offset, product_offset + product_count):
        za = int(data["mf10_prod_za"][product_i])
        state = int(data["mf10_prod_state"][product_i])
        Z, A = za_to_Z_A(za)

        if product_Z is not None and Z != int(product_Z):
            continue
        if product_A is not None and A != int(product_A):
            continue

        table_offset = int(data["mf10_tab_offsets"][product_i])
        table_count = int(data["mf10_tab_counts"][product_i])
        if table_count <= 0:
            continue

        E = np.array(data["mf10_E_data"][table_offset : table_offset + table_count], dtype=float)
        y = np.array(data["mf10_Y_data"][table_offset : table_offset + table_count], dtype=float)

        sigma = interp_zero_outside(energy_eV, E, y)
        if sigma <= 0.0:
            continue

        label = label_from_za_and_state(za, state)
        if label is None:
            continue

        rows.append({
            "product": label,
            "product_Z": Z,
            "product_A": A,
            "product_state": state,
            "mf10_xs": float(sigma),
        })

    return rows


def sample_mf10_product_state(data, mt, energy_eV, product_Z, product_A):
    """Sample ground/metastable product state using MF10 production cross sections.

    Returns None when MF10 has no useful product-state information.
    """
    rows = mf10_product_rows_at_energy(
        data=data,
        mt=mt,
        energy_eV=energy_eV,
        product_Z=product_Z,
        product_A=product_A,
    )

    if len(rows) == 0:
        return None

    total = 0.0
    for row in rows:
        total += float(row["mf10_xs"])

    if total <= 0.0:
        return None

    R = np.random.random() * total
    running = 0.0
    chosen = rows[-1]

    for row in rows:
        running += float(row["mf10_xs"])
        if R <= running:
            chosen = row
            break

    chosen = dict(chosen)
    chosen["branch_probability"] = float(chosen["mf10_xs"] / total)
    chosen["branch_total_xs"] = float(total)

    return chosen

def base_residual_product_for_mt(data, mt):
    """Compute the residual product implied by MT before MF10 branching."""
    mt = int(mt)
    target = get_target_info_from_data(data)
    target_Z = target["Z"]
    target_A = target["A"]
    target_label = target["label"]

    if mt == 2:
        return {
            "target": target_label,
            "product": None,
            "product_Z": None,
            "product_A": None,
            "emitted": None,
            "emitted_counts": {},
            "note": "elastic scattering, no residual transmutation",
            "product_state": None,
            "product_state_source": None,
            "product_branch_probability": None,
            "product_branch_total_xs": None,
        }

    emitted = emitted_string_for_mt(mt)
    if emitted is None:
        return {
            "target": target_label,
            "product": None,
            "product_Z": None,
            "product_A": None,
            "emitted": None,
            "emitted_counts": {},
            "note": "unknown emitted particles for this MT",
            "product_state": None,
            "product_state_source": None,
            "product_branch_probability": None,
            "product_branch_total_xs": None,
        }

    emitted_counts = parse_emitted_particles(emitted)

    emitted_A = 0
    emitted_Z = 0
    for particle, count in emitted_counts.items():
        info = PARTICLE_INFO[particle]
        emitted_A += count * info["A"]
        emitted_Z += count * info["Z"]

    product_A = target_A + 1 - emitted_A
    product_Z = target_Z - emitted_Z
    product_symbol = symbol_from_Z(product_Z)

    product_state = 0
    product_state_source = "ground-state-default"
    product_branch_probability = None
    product_branch_total_xs = None

    if product_symbol is None or product_A <= 0:
        product_label = None
        note = "computed invalid residual product"
        product_state = None
        product_state_source = None
    else:
        product_label = make_isotope_label(product_symbol, product_A)
        note = "residual product"

    if 51 <= mt <= 91 or mt == 4:
        product_label = target_label
        product_Z = target_Z
        product_A = target_A
        product_state = 0
        product_state_source = "same-isotope-inelastic"
        note = "inelastic scattering, residual nucleus is same isotope excited/de-exciting"

    return {
        "target": target_label,
        "product": product_label,
        "product_Z": product_Z,
        "product_A": product_A,
        "emitted": emitted,
        "emitted_counts": emitted_counts,
        "note": note,
        "product_state": product_state,
        "product_state_source": product_state_source,
        "product_branch_probability": product_branch_probability,
        "product_branch_total_xs": product_branch_total_xs,
    }


def residual_product_for_mt(data, mt, energy_eV=None):
    """Compute residual product isotope from target isotope and MT.

    If MF10 product-state data are available, this can split the product into
    ground/metastable states. For example, Mg-24(n,p) can become Na-24 or
    Na-24m instead of always becoming Na-24.
    """
    mt = int(mt)
    product_info = base_residual_product_for_mt(data, mt)

    # MF10 product-state branching should happen only after we know the normal
    # residual product. MF3 chooses the reaction MT. MF10 chooses which product
    # state was populated by that MT.
    if product_info["product"] is not None and product_info["note"] == "residual product":
        mf10_choice = sample_mf10_product_state(
            data=data,
            mt=mt,
            energy_eV=energy_eV,
            product_Z=product_info["product_Z"],
            product_A=product_info["product_A"],
        )

        if mf10_choice is not None:
            product_info = dict(product_info)
            product_info["product"] = mf10_choice["product"]
            product_info["product_Z"] = mf10_choice["product_Z"]
            product_info["product_A"] = mf10_choice["product_A"]
            product_info["product_state"] = mf10_choice["product_state"]
            product_info["product_state_source"] = "MF10"
            product_info["product_branch_probability"] = mf10_choice["branch_probability"]
            product_info["product_branch_total_xs"] = mf10_choice["branch_total_xs"]
            product_info["note"] = "residual product from MF10 branching"

    return product_info


ALWAYS_REMOVE_MTS = {1, 3, 5, 10, 50, 101, 151}
SUMMARY_FALLBACK_GROUPS = {
    4: set(range(51, 92)),
    16: set(range(875, 892)),
    103: set(range(600, 650)),
    104: set(range(650, 700)),
    105: set(range(700, 750)),
    106: set(range(750, 800)),
    107: set(range(800, 850)),
}


def any_open_mt_in_group(open_reactions, mt_group):
    """Return True if any MT in mt_group is open."""
    for row in open_reactions:
        if int(row["mt"]) in mt_group and float(row["sigma_barns"]) > 0.0:
            return True
    return False


def clean_reaction_list(open_reactions):
    """Remove summary/redundant MT channels from the reaction list."""
    cleaned = []
    for row in open_reactions:
        mt = int(row["mt"])
        if mt in ALWAYS_REMOVE_MTS:
            continue
        if mt in SUMMARY_FALLBACK_GROUPS:
            if any_open_mt_in_group(open_reactions, SUMMARY_FALLBACK_GROUPS[mt]):
                continue
        cleaned.append(row)
    return cleaned


def total_macroscopic_xs(open_reactions):
    """Sum all macroscopic cross sections."""
    total = 0.0
    for row in open_reactions:
        total += float(row["macro_xs_1_per_m"])
    return total


def add_reaction_probabilities(open_reactions):
    """Add probability = Sigma_channel / Sigma_total to each row."""
    Sigma_total = total_macroscopic_xs(open_reactions)
    if Sigma_total <= 0.0:
        for row in open_reactions:
            row["probability"] = 0.0
        return open_reactions
    for row in open_reactions:
        row["probability"] = float(row["macro_xs_1_per_m"]) / Sigma_total
    return open_reactions


def choose_reaction(open_reactions):
    """Randomly choose one reaction using row['probability']."""
    if len(open_reactions) == 0:
        return None
    R = np.random.random()
    running = 0.0
    for row in open_reactions:
        running += float(row.get("probability", 0.0))
        if R < running:
            return row
    return open_reactions[-1]


def choose_weighted(rows, weight_key):
    """Choose one row with probability proportional to row[weight_key]."""
    total = 0.0
    for row in rows:
        total += float(row[weight_key])
    if total <= 0.0:
        return None
    R = np.random.random() * total
    running = 0.0
    for row in rows:
        running += float(row[weight_key])
        if R <= running:
            return row
    return rows[-1]


def sample_free_path(xs_total):
    """Sample free path distance in meters from total macroscopic XS."""
    if xs_total <= 0.0:
        return float("inf")
    R = max(np.random.random(), 1.0e-300)
    return -np.log(R) / xs_total


def sample_from_tabulated_pdf(x_values, pdf_values):
    """Sample x from a tabulated probability density."""
    x = np.array(x_values, dtype=float)
    pdf = np.array(pdf_values, dtype=float)
    if len(x) < 2 or len(pdf) < 2:
        return None

    n = min(len(x), len(pdf))
    x = x[:n]
    pdf = pdf[:n]

    order = np.argsort(x)
    x = x[order]
    pdf = np.maximum(pdf[order], 0.0)

    dx = np.diff(x)
    if len(dx) == 0 or np.any(dx <= 0.0):
        return None

    area_pieces = 0.5 * (pdf[:-1] + pdf[1:]) * dx
    cdf = np.zeros(len(x))
    cdf[1:] = np.cumsum(area_pieces)
    total_area = cdf[-1]
    if total_area <= 0.0:
        return None

    R = np.random.random() * total_area
    return float(np.interp(R, cdf, x))


def legendre_pdf_from_coeffs(mu_values, coeffs):
    """Convert Legendre coefficients into p(mu)."""
    mu = np.array(mu_values, dtype=float)
    coeffs = np.array(coeffs, dtype=float)
    pdf = np.ones(len(mu), dtype=float) * 0.5
    for j, a_l in enumerate(coeffs):
        ell = j + 1
        basis = np.zeros(ell + 1)
        basis[ell] = 1.0
        P_l = np.polynomial.legendre.legval(mu, basis)
        pdf += 0.5 * (2 * ell + 1) * float(a_l) * P_l
    return np.maximum(pdf, 0.0)


def neutron_speed_from_energy(energy_eV):
    """Convert neutron kinetic energy in eV to speed in m/s."""
    energy_J = float(energy_eV) * EV_TO_J
    if energy_J <= 0.0:
        return 0.0
    return float(np.sqrt(2.0 * energy_J / M_NEUTRON))


def total_atom_number_density_from_density(rho_g_cm3, molar_mass_g_mol):
    """Convert density and molar mass into total atom number density [atoms/m^3]."""
    rho_g_m3 = float(rho_g_cm3) * 1.0e6
    mol_per_m3 = rho_g_m3 / float(molar_mass_g_mol)
    return mol_per_m3 * AVOGADRO


def make_single_isotope_material(isotope_label, data, number_density_m3):
    """Make a one-isotope material dictionary."""
    return {
        "name": isotope_label + " material",
        "isotopes": {
            isotope_label: {
                "data": data,
                "number_density_m3": float(number_density_m3),
                "fraction": 1.0,
            }
        },
    }


def load_python_material_module(material_file):
    """Load a simple material Python file like Steel.py."""
    spec = importlib.util.spec_from_file_location("material_module", material_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def material_from_composition_file(material_file, xs_dir):
    """
    Load material from a composition file exposing:
        COMP_LABEL
        DENSITY
        general_comp
        elemental_isotopes

    general_comp[element]['mass_fraction'] gives elemental mass fraction.
    elemental_isotopes[element][isotope]['mass_fraction'] gives isotopic mass fraction.
    """
    module = load_python_material_module(material_file)

    label = getattr(module, "COMP_LABEL", os.path.basename(material_file).replace(".py", ""))
    density_g_cm3 = float(getattr(module, "DENSITY"))
    general_comp = getattr(module, "general_comp")
    elemental_isotopes = getattr(module, "elemental_isotopes")

    rho_g_m3 = density_g_cm3 * 1.0e6

    material = {
        "name": label,
        "density_g_cm3": density_g_cm3,
        "source_file": material_file,
        "isotopes": {},
        "missing_xs_files": [],
    }

    for element, element_info in general_comp.items():
        element_mf = float(element_info["mass_fraction"])
        iso_dict = elemental_isotopes.get(element, {})

        for isotope_label, iso_info in iso_dict.items():
            isotope_mf_inside_element = float(iso_info["mass_fraction"])
            total_mass_fraction = element_mf * isotope_mf_inside_element

            symbol, A, metastable = split_isotope_label(isotope_label)
            molar_mass_g_mol = float(A)

            number_density_m3 = rho_g_m3 * total_mass_fraction / molar_mass_g_mol * AVOGADRO

            isotope_data = load_neutron_npz(xs_dir, isotope_label)
            if isotope_data is None:
                material["missing_xs_files"].append(isotope_label)
                continue

            material["isotopes"][isotope_label] = {
                "data": isotope_data,
                "number_density_m3": number_density_m3,
                "mass_fraction": total_mass_fraction,
                "element": element,
            }

    return material


def build_isotope_reaction_list_for_energy(isotope_label, isotope_data, energy_eV, number_density_m3):
    """Build cleaned reaction list for one isotope at one neutron energy."""
    rows = get_open_reactions_with_macro_xs_sorted(
        data=isotope_data,
        energy_eV=energy_eV,
        number_density_m3=number_density_m3,
    )
    rows = clean_reaction_list(rows)

    for row in rows:
        row["target_isotope"] = isotope_label
        row["target_data"] = isotope_data
        row["number_density_m3"] = number_density_m3

    return rows


def build_material_reaction_list_for_energy(material, energy_eV):
    """Build reaction list for a material mixture."""
    all_rows = []
    for isotope_label, isotope_info in material["isotopes"].items():
        isotope_rows = build_isotope_reaction_list_for_energy(
            isotope_label=isotope_label,
            isotope_data=isotope_info["data"],
            energy_eV=energy_eV,
            number_density_m3=isotope_info["number_density_m3"],
        )
        all_rows.extend(isotope_rows)

    all_rows.sort(key=lambda row: row["macro_xs_1_per_m"], reverse=True)
    return add_reaction_probabilities(all_rows)


# Energy-bin reaction-list cache

def energy_bin_key(energy_eV, relative_width=0.01):
    """
    Convert continuous neutron energy into a log-energy cache key.
    """

    energy_eV = float(energy_eV)
    relative_width = float(relative_width)

    if energy_eV <= 0.0:
        return ("zero", relative_width, 0)

    logE = np.log(energy_eV)
    dlogE = np.log(1.0 + relative_width)

    index = int(np.floor(logE / dlogE))

    return ("log", relative_width, index)


def copy_reaction_rows(rows):
    """
    Make shallow copies of reaction row dictionaries.

    This prevents accidental mutation of cached rows during transport.
    """

    copied = []

    for row in rows:
        copied.append(dict(row))

    return copied


def build_material_reaction_list_for_energy_cached(
    material,
    energy_eV,
    relative_width=0.01,
):
    """
    Cached version of build_material_reaction_list_for_energy.

    Cache lives inside the material dictionary:

        material["_reaction_cache"]

    Cache key uses logarithmic energy bins, not exact energy.

    This is approximate:
        all energies in the same bin reuse the reaction list evaluated at the
        bin center energy.

    For learning / speed, relative_width = 0.01 is a good starting point.
    """

    if "_reaction_cache" not in material:
        material["_reaction_cache"] = {}

    cache = material["_reaction_cache"]

    key = energy_bin_key(
        energy_eV=energy_eV,
        relative_width=relative_width,
    )

    if key in cache:
        cache[key]["hits"] += 1
        return copy_reaction_rows(cache[key]["rows"])

    # Use the actual requested energy for the first energy that enters the bin.
    # This is simpler than calculating the exact bin-center energy.
    rows = build_material_reaction_list_for_energy(
        material=material,
        energy_eV=energy_eV,
    )

    cache[key] = {
        "energy_eV": float(energy_eV),
        "rows": copy_reaction_rows(rows),
        "hits": 0,
    }

    return copy_reaction_rows(rows)


def clear_material_reaction_cache(material):
    """
    Clear cached reaction lists from a material.
    """

    material["_reaction_cache"] = {}


def material_reaction_cache_summary(material):
    """
    Print a small summary of the material reaction cache.
    """

    cache = material.get("_reaction_cache", {})

    print("Number of cached energy bins:", len(cache))

    total_hits = 0

    for key, entry in cache.items():
        total_hits += int(entry.get("hits", 0))

    print("Total cache hits:", total_hits)

    if len(cache) > 0:
        energies = []

        for entry in cache.values():
            energies.append(entry["energy_eV"])

        print("Cached energy range:")
        print("  min =", min(energies), "eV")
        print("  max =", max(energies), "eV")
