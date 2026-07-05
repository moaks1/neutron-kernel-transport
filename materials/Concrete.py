"""
Building-material composition module (Concrete (Naqvi et al. 2004)).

Source file: gf_concrete.txt
Concrete rounded weight-fraction values from Naqvi, Nagadi, and Al-Amoundi (2004).

The source table gives oxide/carbonate compound weight fractions and density. Compound
rows were converted to elemental mass fractions using standard atomic weights.
Elemental fractions are normalized to sum to 1.0.

Expose:
  - COMP_LABEL: short label that will appear in the PDF
  - DENSITY: bulk density (g/cc)
  - general_comp: dict[element] -> mass fraction (sums to 1)
  - elemental_isotopes: dict[element] -> dict[isotope] -> abundance fraction
"""

COMP_LABEL = 'Concrete (Naqvi et al. 2004)'

DENSITY = 2.2  # g/cc

general_comp = {
    'C' : {"mass_fraction": 0.06000135881918613},
    'O' : {"mass_fraction": 0.4725642144574509},
    'Na': {"mass_fraction": 0.0007418574701063741},
    'Mg': {"mass_fraction": 0.006030358968251604},
    'Al': {"mass_fraction": 0.01058501395583737},
    'Si': {"mass_fraction": 0.1589278730050945},
    'S' : {"mass_fraction": 0.002002480540373105},
    'K' : {"mass_fraction": 0.003320591107902671},
    'Ca': {"mass_fraction": 0.2788319966212598},
    'Fe': {"mass_fraction": 0.00699425505453753},
}

# natural isotopic vectors --------------------------------------------------
# keys must match symbols in general_comp

elemental_isotopes = {
    'C': {
        'C-12': {"mass_fraction": 0.9893},
        'C-13': {"mass_fraction": 0.0107},
    },
    'O': {
        'O-16': {"mass_fraction": 0.9976},
        'O-17': {"mass_fraction": 0.0004},
        'O-18': {"mass_fraction": 0.002},
    },
    'Na': {
        'Na-23': {"mass_fraction": 1},
    },
    'Mg': {
        'Mg-24': {"mass_fraction": 0.7899},
        'Mg-25': {"mass_fraction": 0.1},
        'Mg-26': {"mass_fraction": 0.1101},
    },
    'Al': {
        'Al-27': {"mass_fraction": 1},
    },
    'Si': {
        'Si-28': {"mass_fraction": 0.9223},
        'Si-29': {"mass_fraction": 0.0467},
        'Si-30': {"mass_fraction": 0.031},
    },
    'S': {
        'S-32': {"mass_fraction": 0.9499},
        'S-33': {"mass_fraction": 0.0075},
        'S-34': {"mass_fraction": 0.0425},
        'S-36': {"mass_fraction": 0.0001},
    },
    'K': {
        'K-39': {"mass_fraction": 0.932581},
        'K-40': {"mass_fraction": 0.000117},
        'K-41': {"mass_fraction": 0.067302},
    },
    'Ca': {
        'Ca-40': {"mass_fraction": 0.96941},
        'Ca-42': {"mass_fraction": 0.00647},
        'Ca-43': {"mass_fraction": 0.00135},
        'Ca-44': {"mass_fraction": 0.02086},
        'Ca-46': {"mass_fraction": 4e-05},
        'Ca-48': {"mass_fraction": 0.00187},
    },
    'Fe': {
        'Fe-54': {"mass_fraction": 0.05845},
        'Fe-56': {"mass_fraction": 0.91754},
        'Fe-57': {"mass_fraction": 0.02119},
        'Fe-58': {"mass_fraction": 0.00282},
    },
}
