# Kernel-Based Event-Driven Monte Carlo Neutron Transport

Kernel-based, event-driven analog Monte Carlo neutron transport using processed ENDF-style nuclear data. The code simulates neutron histories through a homogeneous 2D material, uses a precomputed material transport kernel for fast per-collision sampling, records residual activation products, and computes post-transport direct and daughter-chain activity.

This is a research and learning prototype. It is meant to show how neutron transport, preprocessed nuclear data, reaction-channel sampling, residual product formation, secondary-neutron production, metastable product-state branching, and radioactive activity calculations can fit together in one workflow.

## What This Code Is

This is still an event-driven Monte Carlo transport code:

```text
source neutron
sample next free path
move to collision or boundary
sample reaction channel
update neutron state or absorb it
create secondary neutrons when needed
record event history
repeat until histories end
```

It is also kernel-based:

```text
expensive nuclear-data lookups and CDF construction happen up front
runtime transport uses fast energy-bin lookup and precomputed CDFs
random choices still happen during each Monte Carlo history
```

The kernel does not precompute neutron histories and it does not make the simulation deterministic. It precomputes distributions, CDFs, residual-product metadata, MF4/MF6 sampling tables, and multiplicity information. The Monte Carlo runtime still samples random free paths, reactions, outgoing neutron states, residual branches, and secondary-neutron multiplicities.

## Main Features

* Event-driven analog Monte Carlo neutron transport
* Precomputed material transport kernel for fast collision sampling
* Material-specific kernel cache names with composition fingerprints
* Notebook-controlled many-source parallel workers with `N_WORKERS`
* Single-source and many-source neutron modes in one notebook
* Homogeneous 2D square material geometry
* Material composition from Python material files
* Processed ENDF-style neutron and decay `.npz` data
* MF3-style cross-section driven reaction probabilities
* Energy-bin reaction CDF lookup
* Free-path sampling from total macroscopic cross section
* MF4 elastic angular scattering support
* MF6 LAW=1 outgoing-neutron energy sampling
* MF6 LAW=2 two-body outgoing-neutron energy and direction sampling
* Outgoing neutron multiplicity metadata and stochastic non-integer yields
* Secondary-neutron creation for multiplication reactions
* Residual/transmutation product tracking
* MF10 product-state and metastable branch sampling
* Transport history diagnostics with kernel provenance columns
* Activation product tables with decay status
* Direct residual-product activity curves
* Daughter-chain activity from matrix-exponential decay-network evolution
* Short-lived isotope collapse safeguard for prompt intermediates such as Be-8
* CSV outputs and PNG plots written under `outputs/`

## Repository Layout

```text
Kernel_Based_MC_NT/
    README.md
    LICENSE
    requirements.txt

    Kernel_MC_Neutron_Transport.ipynb

    nuclear_data.py
    transport.py
    transport_kernel.py
    mf4.py
    mf6.py
    decay_activity.py
    notebook_helpers.py

    materials/
        Steel.py
        Concrete.py

    kernels/
        material_transport_kernel_<material-label>_<fingerprint>_v1.npz

    outputs/
        transport_history.csv
        residual_products.csv
        activation_products_with_decay.csv
        activity_chain.csv
        neutron_traj.png
        total_activity.png
        isotope_activities.png

    tests/
        test_decay_activity.py
```

The processed nuclear data files are not included in this repository. Generated outputs and kernels can be large and should usually be treated as local artifacts.

## Nuclear Data

This repository contains code, not the processed nuclear-data library.

The notebook expects processed neutron and decay data in `.npz` form. The expected folders are:

```text
neutron_npz/
    n-026_Fe_056.npz
    n-008_O_016.npz
    ...

decay_npz/
    dec-026_Fe_56.npz
    dec-011_Na_24.npz
    ...
```

The exact isotope files required depend on the material composition and reaction products encountered during transport.

## Main Notebook

Use:

```text
Kernel_MC_Neutron_Transport.ipynb
```

The notebook contains the full workflow:

```text
folder setup
user controls
material loading
initial reaction preview
kernel load/build
single-source or many-source transport
transport diagnostics
trajectory plot
residual product summary
decay status lookup
direct and chain activity calculation
activity plots
CSV output saving
```

Single-source and many-source runs are controlled by:

```python
TRANSPORT_MODE = "single"  # "single" or "many"
```

For single-source mode:

```python
MAX_EVENTS = 500
MAX_NEUTRONS = 50
START_X = 0.0
START_Y = 0.0
START_DIRECTION = None
```

For many-source mode:

```python
N_SOURCE_NEUTRONS = 100
N_WORKERS = 1              # Serial. Use an integer > 1 for process workers, or "auto".
MAX_EVENTS_PER_SOURCE = 500
MAX_NEUTRONS_PER_SOURCE = 50
MAX_TRAJECTORY_PATHS = 80
```

For early testing, keep many-source values small. Large many-source runs can produce large transport-history CSV files. `N_WORKERS = 1` keeps the run serial. Larger integer values split independent source-neutron histories across worker processes.

## Transport Kernel

The material transport kernel is the fast path for runtime collision processing.

Kernel construction precomputes expensive per-energy-bin information:

```text
energy-bin edges and centers
open reaction rows per energy bin
total macroscopic cross section per bin
mean free path per bin
reaction probabilities and CDF values
residual_product, residual_product_Z, residual_product_A, product_note
MF10 product-state branch CDFs where available
n_out_expected, n_out_integer_rule, n_out_source
MF4 elastic angular CDFs for MT=2
MF6 outgoing-neutron product CDFs for supported LAW=1 and LAW=2 data
MF6 LAW=2 two-body angle/energy metadata
```

Runtime collision processing then uses:

```text
energy-bin lookup
free-path sampling
reaction CDF sampling
residual/MF10 branch sampling
outgoing-neutron multiplicity sampling
outgoing energy and direction sampling
secondary-neutron creation
history logging
```

The notebook now builds the kernel cache path from the loaded material:

```python
KERNEL_NPZ_PATH = material_transport_kernel_path(
    kernel_dir=KERNEL_DIR,
    material=material,
    version=KERNEL_VERSION,
)
```

The resulting filename includes a readable material label and a short composition fingerprint:

```text
kernels/material_transport_kernel_<material-label>_<fingerprint>_v1.npz
```

That keeps `Steel`, `Concrete`, and future materials from accidentally sharing one kernel file. If the material density or isotope number densities change, the fingerprint changes too. Loaded kernels with stored fingerprints are rejected if they do not match the currently loaded material.

Important kernel controls:

```python
USE_TRANSPORT_KERNEL = True
LOAD_KERNEL_IF_EXISTS = True
KERNEL_VERSION = "v1"
KERNEL_E_MIN_EV = 1.0e-5
KERNEL_E_MAX_EV = 2.0e7
KERNEL_BINS_PER_DECADE = 100
BUILD_MF4_ELASTIC_KERNEL = True
MF4_MU_GRID_COUNT = 801
```

Rebuild the kernel after changing material composition, energy range, bin density, angular grid settings, MF4/MF6 processing behavior, or kernel code.

## Transport Physics Currently Included

The transport engine supports:

* Analog event-by-event neutron tracking
* Free-path sampling from total macroscopic cross section
* Boundary escape from a 2D square box
* Elastic scattering with MF4 angular data when available
* Nonelastic outgoing-neutron energy sampling with MF6 LAW=1 when available
* Two-body MF6 LAW=2 outgoing-neutron energy and lab-angle sampling when available
* Random 2D fallback directions when angular data are unavailable
* Absorption and neutron disappearance reactions
* Secondary-neutron creation for reactions with multiple outgoing neutrons
* Residual product recording for activation analysis
* Kernel and fallback provenance tracking in the transport history

Transport events may include:

```text
start
scatter-like
neutron multiplication
absorbed
escaped
created secondary
no open reactions
```

## MF10 Metastable Product-State Branching

Some ENDF evaluations provide MF10 product-state production data. For example:

```text
Mg-24(n,p)Na-24
```

may produce:

```text
Na-24
Na-24m
```

When MF10 data are available, the kernel precomputes product-state branch CDFs at each material energy bin. During transport, the residual product branch is sampled stochastically, so the history can distinguish ground-state and metastable residual products.

The transport history records diagnostic product-state fields:

```text
product_state
product_state_source
product_branch_probability
product_branch_total_xs
residual_product_sampling_source
residual_product_random_number
residual_product_branch_cdf
```

## Python Modules

### `nuclear_data.py`

Nuclear-data utilities: isotope labels, file names, neutron-data loading, decay-file lookup, MT names, cross sections, reaction-list construction, residual product identification, emitted-particle parsing, MF10 product-state branch handling, and material loading.

### `transport.py`

The event-driven neutron transport engine. It defines neutron state, samples free paths, schedules collision or escape events, consumes kernel reaction rows, samples reactions, updates outgoing neutron energy/direction, creates secondary neutrons, and records detailed history rows.

### `transport_kernel.py`

Builds, saves, loads, and samples precomputed material transport kernels. The kernel stores flat arrays for fast lookup, including reaction CDFs, residual-product data, MF10 branch CDFs, multiplicity metadata, MF4 elastic CDFs, and supported MF6 outgoing-neutron CDFs.

### `mf4.py`

MF4 angular distributions and elastic scattering kinematics. Used for elastic collision direction and energy updates.

### `mf6.py`

MF6 outgoing-neutron distributions. Used for LAW=1 outgoing-energy sampling and LAW=2 two-body energy-angle behavior.

### `decay_activity.py`

Activation inventory and decay activity logic. It extracts residual products, attaches decay status, computes direct activity, builds a decay network, evolves daughter chains with matrix exponentials, and collapses very short-lived prompt intermediates below a configurable half-life threshold.

### `notebook_helpers.py`

Notebook workflow helpers for material loading, reaction previews, single-source and many-source runs, diagnostics, trajectory plots, activity grid generation, activity computation, and plotting.

## Installation

Create a Python environment and install:

```bash
pip install -r requirements.txt
```

Expected packages:

```text
numpy
pandas
matplotlib
scipy
jupyter
```

`scipy` is preferred for matrix exponentials in the decay-chain solver. The code also has a NumPy fallback for the matrix exponential.

## Quick Start

Create local folders for data, kernels, and outputs:

```bash
mkdir -p data/neutron_npz
mkdir -p data/decay_npz
mkdir -p kernels
mkdir -p outputs
```

Place processed neutron files in:

```text
data/neutron_npz/
```

Place processed decay files in:

```text
data/decay_npz/
```

Place or edit material files in:

```text
materials/
```

Start Jupyter:

```bash
jupyter lab
```

Open:

```text
Kernel_MC_Neutron_Transport.ipynb
```

Set the data folder paths, material, transport mode, kernel options, and run the notebook from top to bottom.

## Important User Controls

Typical material and source controls:

```python
MATERIAL_NAME = "Steel"
INITIAL_ENERGY_EV = 14.0e6
BOX_SIZE_M = 5.0
RANDOM_SEED = None
RESET_CACHE_BEFORE_RUN = True
```

Transport mode:

```python
TRANSPORT_MODE = "single"  # "single" or "many"
```

Kernel controls:

```python
USE_TRANSPORT_KERNEL = True
LOAD_KERNEL_IF_EXISTS = True
KERNEL_VERSION = "v1"
KERNEL_E_MIN_EV = 1.0e-5
KERNEL_E_MAX_EV = 2.0e7
KERNEL_BINS_PER_DECADE = 100
BUILD_MF4_ELASTIC_KERNEL = True
MF4_MU_GRID_COUNT = 801
```

Activity controls:

```python
ACTIVITY_POINTS_PER_DECADE = 100
ACTIVITY_MIN_LINEAR_POINTS = 400
ACTIVITY_HALF_LIVES_TO_SHOW = 8.0
MIN_ACTIVITY_HALF_LIFE_S = 1.0e-12
```

`MIN_ACTIVITY_HALF_LIFE_S` prevents prompt intermediates such as Be-8 from appearing as unphysical delta-like spikes in ordinary activation plots. Nuclides below the threshold are collapsed into their daughters for observable activity calculations.

## Main Outputs

The notebook writes CSV and PNG artifacts to:

```text
outputs/
```

Typical saved files:

```text
outputs/transport_history.csv
outputs/residual_products.csv
outputs/activation_products_with_decay.csv
outputs/activity_chain.csv
outputs/neutron_traj.png
outputs/total_activity.png
outputs/isotope_activities.png
```

Important in-memory DataFrames:

```text
hist_df
products_df
activation_df
activation_df_with_decay
direct_activity_df
chain_activity_df
```

## Transport History Columns

The transport history is intentionally verbose so that individual events can be audited. Common columns include:

```text
source_id
neutron_id
generation
parent_id
t
x
y
energy_eV
dir_x
dir_y
event
mt
reaction_name
distance_m
target_isotope
residual_product
residual_product_Z
residual_product_A
product_note
product_state
product_state_source
product_branch_probability
product_branch_total_xs
residual_product_sampling_source
residual_product_random_number
residual_product_branch_cdf
n_out
alive_after_collision
Sigma_total_1_per_m
mean_free_path_m
chosen_probability
chosen_cdf
reaction_random_number
reaction_sampling_source
kernel_bin_index
kernel_energy_eV
kernel_E_low_eV
kernel_E_high_eV
incoming_energy_eV
outgoing_energy_eV
energy_update_source
mu_cm
mu_lab
theta_lab_deg
angle_source
angle_frame
angle_azimuth_random_number
n_out_expected
n_out_integer_rule
n_out_source
n_out_random_number
n_out_mt_count
n_out_mf6_total_yield
n_out_mf6_product_count
elastic_has_mf4_kernel
mf6_has_kernel
mf6_product_count
mf6_total_neutron_product_count
mf6_missing_product_count
num_secondaries_created
```

These fields make it possible to inspect which reaction was sampled, which kernel bin was used, whether MF4/MF6 kernel data were available, how neutron energy changed, whether a neutron was absorbed, and what residual product was produced.

## Activation and Activity

Residual products from transport are converted into activation inventories.

The activity workflow:

```text
extract residual activation products
attach decay file status and half-lives
build automatic activity time grid
compute direct residual-product activity
build daughter decay network
evolve decay network with matrix exponentials
collapse very short-lived prompt intermediates
write activity CSV
plot total and isotope activities
```

Activity is reported in:

```text
Bq
Ci
```

The chain solver treats production events as impulse sources and evolves inventories between output times using:

```text
dN/dt = A N
N(t + dt) = exp(A dt) N(t)
```

This is much closer to the approach used by depletion and activation tools than a simple grid-step daughter bookkeeping method.

## Tests

Run the current test suite with:

```bash
python -m unittest tests/test_decay_activity.py
```

The tests currently focus on direct activity, exact daughter-chain behavior, stable daughters, chain-depth limiting, and short-lived isotope collapse.

## Current Limitations

This is a research prototype, not a production radiation transport code.

Important simplifications include:

* Geometry is a 2D square box
* Material is homogeneous
* Transport is analog and not variance-reduced
* No weight windows, splitting, or Russian roulette
* Charged-particle transport is not modeled
* Photon/gamma transport is not modeled
* Thermal scattering treatment is not included
* Doppler broadening is not included
* Material self-shielding is not fully treated
* Nuclear data must already be processed into the expected `.npz` format
* Kernel data are material- and settings-specific; cache filenames include the material label and composition fingerprint
* Unsupported MF6 laws and missing MF6 data use fallback runtime behavior
* Direction transport is still 2D
* MF6 LAW=2 angles are projected into the 2D transport model
* Fission multiplicity support depends on processed data availability
* Small Monte Carlo runs can have large statistical noise
* Activity results need external scaling for absolute source-yield normalization

## Possible Future Improvements

Potential extensions include:

* 3D geometry and direction transport
* More complete geometry support
* Variance reduction with particle weights
* Weight windows, splitting, and Russian roulette
* Source spectrum sampling
* Photon/gamma transport
* Charged-particle transport
* Better material self-shielding treatment
* Thermal scattering support
* Doppler broadening
* Lethargy-averaged or multigroup reaction handling
* More complete ENDF law support
* Full fission neutron multiplicity from processed `nu` data
* More complete correlated MF6 energy-angle laws
* More synthetic nuclear-data tests

## License

This project is released under the MIT License.

## Author

Mathew Oaks

## Project Status

Active research and learning prototype.

The current purpose of the repository is to demonstrate a kernel-based event-driven Monte Carlo neutron transport workflow connected to residual activation and radioactive decay-chain activity analysis.
