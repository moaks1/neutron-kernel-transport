"""
Building-material composition module (Steel, generic carbon steel).

Composition is an A36-like structural carbon steel approximation.
Only elements with atomic number Z < 34 are included.

Elemental fractions are normalized to sum to 1.0.

Expose:
  - COMP_LABEL: short label that will appear in the PDF
  - DENSITY: bulk density (g/cc)
  - general_comp: dict[element] -> mass fraction (sums to 1)
  - elemental_isotopes: dict[element] -> dict[isotope] -> abundance fraction
"""

COMP_LABEL = 'Steel (generic carbon steel, Z < 34)'

DENSITY = 7.85  # g/cc

general_comp = {
    'C' : {"mass_fraction": 0.0026},
    'Si': {"mass_fraction": 0.0040},
    'P' : {"mass_fraction": 0.0004},
    'S' : {"mass_fraction": 0.0005},
    'Mn': {"mass_fraction": 0.0103},
    'Cu': {"mass_fraction": 0.0020},
    'Fe': {"mass_fraction": 0.9802},
}

# natural isotopic vectors --------------------------------------------------
# keys must match symbols in general_comp

elemental_isotopes = {
    'C': {
        'C-12': {"mass_fraction": 0.9893},
        'C-13': {"mass_fraction": 0.0107},
    },
    'Si': {
        'Si-28': {"mass_fraction": 0.9223},
        'Si-29': {"mass_fraction": 0.0467},
        'Si-30': {"mass_fraction": 0.031},
    },
    'P': {
        'P-31': {"mass_fraction": 1},
    },
    'S': {
        'S-32': {"mass_fraction": 0.9499},
        'S-33': {"mass_fraction": 0.0075},
        'S-34': {"mass_fraction": 0.0425},
        'S-36': {"mass_fraction": 0.0001},
    },
    'Mn': {
        'Mn-55': {"mass_fraction": 1},
    },
    'Cu': {
        'Cu-63': {"mass_fraction": 0.6915},
        'Cu-65': {"mass_fraction": 0.3085},
    },
    'Fe': {
        'Fe-54': {"mass_fraction": 0.05845},
        'Fe-56': {"mass_fraction": 0.91754},
        'Fe-57': {"mass_fraction": 0.02119},
        'Fe-58': {"mass_fraction": 0.00282},
    },
}
