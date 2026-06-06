# OB-DPD: Automated Framework for Dissipative Particle Dynamics

**OB-DPD** (One-Button DPD) is an automated framework designed to streamline Dissipative Particle Dynamics simulations, from coarse-grained mapping to property prediction.

## 🛠️ Prerequisites

Before installing and running OB-DPD, please ensure you have the following prepared:

1. **External Software:**
   * **Packmol:** Required for generating the initial molecular packing configurations. *(Recommended version: 21.1.4)*
   * **LAMMPS:** Required for executing the Dissipative Particle Dynamics simulations. *(Recommended version: LAMMPS 29 Aug 2024 - Update 2)*
   *(Please ensure both are properly installed and accessible in your system's PATH.)*

2. **Python Dependencies:**
   * The portable Conda environment is defined in `environment.yml`.
   * `OB-DPD.yml` is retained as a full development-environment snapshot.
   * ⚠️ **Important:** Please pay special attention to the exact versions of `rdkit`, `numpy`, `pandas`, `shap`, `seaborn`, `matplotlib`, `numba`, and `scipy`. Strict adherence to these versions is highly recommended to avoid compatibility issues during ML predictions and data processing.
## 📦 Installation 

You can set up the environment using one of the following two methods:

### Method 1: Using Conda Configuration (Recommended)
If Packmol, MPI, and LAMMPS are already installed on your system, use the
automated installer to create the Conda environment and register the local
`auto_mapping/` source paths:

```bash
bash scripts/install_env.sh --lammps-bin lmp_mpi
conda activate dpd
```

The installer reads `environment.yml`, creates or updates the `dpd`
environment, and installs Conda activation hooks for:

```text
OB_DPD_HOME
OB_DPD_AUTO_MAPPING
OB_DPD_AUTO_MAPPING_M3
PYTHONPATH
```

It also runs a dependency check for RDKit, NumPy/SciPy, XGBoost, the local
`auto_mapping/` scripts, Packmol, MPI, and LAMMPS.

Manual installation is also supported:

```bash
conda env create -f environment.yml
conda activate dpd
python scripts/check_install.py --lammps-bin lmp_mpi
```

Verify that the external executables are available:

```bash
packmol < /dev/null
mpirun --version
lmp_mpi -h  # or: lmp -h
```

LAMMPS is resolved in this order: `LAMMPS_BIN`, `lmp_mpi`, then `lmp`. If your
LAMMPS executable has a different name or path, pass it explicitly:

```bash
python main.py --input_data input_data.txt --lammps_bin /path/to/lmp
```

The Martini-style mapping scripts under `auto_mapping/` are included in this
repository and are used as local source files. They are not installed from PyPI.
The portable `environment.yml` therefore installs only their Python
dependencies, such as RDKit, NumPy, SciPy, and requests.

For an exact snapshot of the original development environment, you may use
`OB-DPD.yml` instead:

```bash
conda env create -f OB-DPD.yml
conda activate dpd
```

### Method 2: Offline Installation (Using Release Package)
If you are working offline or require the exact pre-compiled environment, download OB-DPD_env.tar.gz from the Releases page.

```bash
# 1. Create a directory for the environment
mkdir -p envs/dpd
```

```bash
# 2. Extract the package (replace path with your actual download location)
tar -xzf OB-DPD_env.tar.gz -C envs/dpd
```

```bash
# 3. Activate the environment
source envs/dpd/bin/activate
```

## 🚀 Usage
To run the simulation, you need to configure the input_data.txt file and run the main script.

Configure input_data.txt

The input file requires strictly 4 lines of parameters:

**Line 1 (Mode)**: Specify the input mode (smiles or mol).

**Line 2 (Structure)**: Provide the SMILES string or the path to the .mol file.

**Line 3 (Count)**: Number of molecules to simulate.

**Line 4 (PEG Flag)**: Whether the structure contains PEG (1 or 0).


Example A: SMILES Mode

```textc

smiles
CCO(CH2CH2O)10H
50
1
```

Example B: MOL File Mode

```text
mol
./inputs/drug_molecule.mol
100
0
```

Example C: Multi-SMILES Mode

```text
smiles
CCO(CH2CH2O)10H
50
1
CCCCCO(CH2CH2O)10H
50
0
```

## 2. Run the Script
Once the input file is ready, execute the main program:

```bash
python main.py --input_data input_data.txt
```

Once the input file is ready, execute the main program. If you wish to customize the initial molecular packing, provide the Packmol input file as an argument:

```bash
python main.py --input_data input_data.txt --packmol_in packmol.inp
```

### Parameterization Modes

Choose the parameterization route with `--param_method`. The default
`solubility` mode uses the existing ML-predicted solubility parameter workflow:

```bash
python main.py --input_data input_data.txt --param_method solubility
```

The `logp` mode uses a separate Anderson-style bead mapper instead of the
solubility-parameter coarse graining. It keeps the input molecular structure
unchanged, groups atoms into the supported logP bead palette, and then reads
pair parameters from `pdf/logp/machine_readable_interactions.cvs`.

```bash
python main.py --input_data input_data.txt --param_method logp
```

Supported logP bead types are `H2O`, `CH3`, `CH2`, `CH2CH2`, `CH2OH`,
`CH2NH2`, `CH2OCH2`, `CH3OCH2`, and `aCHCH`. The logP mode is intentionally
strict: functional groups not represented by this table are not considered,
including amides, esters, ketones, aldehydes, nitriles, halogenated groups,
nitro groups, heteroaromatic rings, tertiary amines, and quaternary ammonium
groups.

This path does not call the ML solubility-parameter predictor. The atom-to-logP
bead grouping is written to files such as `0logp_article_mapping0.csv`, and the
assigned pair sources are written to `bead_type_assignment.csv` and
`a_ij_source.csv`.

When all bead pairs are covered by the literature table or supported
interpolation rules, the workflow uses those `A_ij` values directly. When a
pair is missing, `--logp_missing` controls the next step:

```bash
# Automatic derivative-free fitting of missing Aij values
python main.py --input_data input_data.txt --param_method logp \
  --logp_missing optimize --logp_fit_execute
```

In `optimize` mode, the missing pair starts from a solubility-parameter-based
initial `A_ij`; `logp_partition.py fit` then runs water/octanol partition DPD
simulations and updates the selected `A_ij` values with `nelder-mead` or
`coordinate` optimization. Bayesian optimization is also available and follows
the literature workflow: random initial sampling on a discrete `A_ij` grid,
`GaussianProcessRegressor` surrogate fitting, Expected Improvement candidate
selection, and a logP RMSE objective for multi-target fitting. Failed LAMMPS or
analysis runs are kept in the history with `observed_logp = +/-10` and a large
penalty loss by default.

```bash
python main.py --input_data input_data.txt --param_method logp \
  --logp_missing optimize --logp_fit_execute \
  --logp_fit_optimizer bayesian \
  --logp_bo_initial 10 \
  --logp_bo_grid 20:60:1 \
  --logp_fit_max_iter 40
```

Use pair-specific grids when a pair needs a narrower search range:

```bash
python main.py --input_data input_data.txt --param_method logp \
  --logp_missing optimize --logp_fit_execute \
  --logp_fit_optimizer bayesian \
  --logp_bo_grid CH3:H2O=20:45:1 \
  --logp_bo_grid CH2OH:CH2CH2=25:70:2
```

For a literature-style Bayesian logP fitting run, enable the article protocol.
This uses LAMMPS with a `20 x 20 x 60` DPD box, 4410 water molecules, 2205
organic solvent molecules, 180 solute molecules, 1,000,000 steps,
500,000-step equilibration, `dt = 0.01`, `fix nph iso 23.7 23.7 2.0`, and
trajectory output every 1000 steps. The DPD pair style supplies the thermostat;
`fix nph` is the LAMMPS approximation to the article's constant-pressure
barostat. The analysis switches to the UMMAP-style route: 30 z-slabs, solvent
and solute concentration profiles, gradient-based interface detection, failure
penalty for collapsed/mixed phases, and the last 50 post-equilibration frames.

```bash
python main.py --input_data input_data.txt --param_method logp \
  --logp_missing optimize --logp_fit_execute \
  --logp_fit_optimizer bayesian \
  --logp_fit_article_protocol \
  --logp_angle_param_mode article \
  --logp_bo_initial 10 \
  --logp_bo_grid 20:60:2 \
  --logp_fit_max_iter 40
```

Angle parameters for logP partition/fitting runs are controlled by
`--logp_angle_param_mode` (or the alias `--angle-param-mode`) in `main.py`, and
by `--angle-param-mode` in `logp_partition.py`. `article` applies the
literature rules for supported Anderson beads (`105` degrees for alkyl
backbones and `125` degrees next to `CH2OH`, with the literature force constant
converted to LAMMPS harmonic units). `geometry` uses template coordinates when
present and otherwise falls back to `heuristic`. `heuristic` keeps article rules
where possible and assigns weak fallback angles to aromatic, ionic, polar, or
unknown bead triplets. `none` disables angle terms entirely. When
`--logp_fit_article_protocol` is used and no angle mode is specified, the
fitting script defaults to `article`.

By default, Bayesian optimization evaluates Expected Improvement over the full
discrete `A_ij` grid in chunks. Disable this only for very large exploratory
runs:

```bash
python logp_partition.py fit-staged \
  --stage-config examples/staged_logp_fit_ethanol.json \
  --optimizer bayesian \
  --article-protocol \
  --bo-grid 20:60:2 \
  --no-bo-full-grid-ei
```

For bead types outside the table, logP fitting can estimate missing `R_ij` and
bonded `r0` from bead heavy atom counts. The default mode only fills missing
values and does not overwrite literature table values:

```bash
python logp_partition.py build \
  --solute X,Y --allow-custom-beads \
  --bead-heavy-atoms X=8 \
  --bead-heavy-atoms Y=1 \
  --default-missing-aij 25 \
  --heavy-atom-correction missing
```

The radius estimate uses `R_ij = r_i + r_j` with
`r_i = 0.5 * heavy_count_i^(1/3)` in reduced units, scaled by
`--heavy-radius-scale`. Missing bonded lengths use
`r0 = --bonded-r0-factor * R_ij` with a default factor of `0.4`. Use
`--heavy-atom-correction all` only when you intentionally want heavy atom counts
to override table `R_ij/r0` values.

If `--logp_fit_execute` is omitted, the workflow writes
`missing_logp_pairs.csv`, `logp_fit_required_pairs.csv`, and
`run_missing_logp_fit.sh`, then stops before production DPD.

```bash
# Manual per-iteration fitting of missing Aij values
python main.py --input_data input_data.txt --param_method logp \
  --logp_missing manual --logp_fit_execute
```

In `manual` mode, the first iteration also starts from the solubility-based
initial value unless initial guesses are supplied:

```bash
python main.py --input_data input_data.txt --param_method logp \
  --logp_missing manual --logp_fit_execute \
  --logp_manual_aij CH3:CH2OH=42.5
```

After each water/octanol simulation, the program prints the target logP,
observed logP, current error, and current `A_ij`. Enter the next trial values
as comma-separated numbers, for example `42.5,55.0`, or as explicit pair
updates such as `CH3:CH2OH=42.5`. Press Enter on a blank line to stop and keep
the best fitted values so far.

Fitting writes the merged parameter table to
`logp_missing_fit/fitted_interactions.csv`. The main workflow automatically
reloads this file after `--logp_fit_execute` and uses it for the final DPD input
generation. If the fitting script was run separately, pass the fitted table
explicitly:

```bash
python main.py --input_data input_data.txt --param_method logp \
  --logp_table logp_missing_fit/fitted_interactions.csv
```

For direct multi-target fitting, use `logp_partition.py fit-staged` with a JSON
stage file. Each evaluation runs the requested LAMMPS partition simulations,
computes per-target logP errors, and minimizes the stage RMSE:

```bash
python logp_partition.py fit-staged \
  --stage-config examples/staged_logp_fit_ethanol.json \
  --optimizer bayesian \
  --bo-initial 10 \
  --bo-grid 20:60:1 \
  --max-iter 40 \
  --job 8
```

In `solubility` mode, bead charges are assigned automatically from the RDKit
formal charge of each bead SMILES. With the default `--charge_method auto`, the
workflow keeps the original neutral DPD input when all charges are zero. If any
nonzero bead charge is found, it writes `atom_style full`, uses explicit bead
charges, and switches the LAMMPS pair style to `dpd/coul/slater/long` with PPPM
long-range electrostatics. LAMMPS input is written in reduced `units lj`: `T =
1.0`, `sigma = 1.0`, and formal charges are used directly as reduced charges by
default.

Charged solubility-mode systems are neutralized automatically with `Na+` or
`Cl-` counterions after packing the solute molecules. Positive solute charge
adds `Cl-`; negative solute charge adds `Na+`. Counterions are added directly in
the LAMMPS input with `create_atoms` and are not passed through the ML
solubility-parameter predictor.

After solutes are packed and counterions are assigned, the workflow fills the box
with water beads so that the final total bead count is exactly
`box_size^3 * 3`. One water bead represents one water molecule. The generated
`lammps.in` reads packed solutes first, creates counterions, and creates water
last. `system_bead_summary.csv` records the target total, solute beads,
counterion beads, water beads, and actual bead density.

Ion `A_ij` values are assigned conservatively: ion-ion and ion-water pairs use
`25.0`, while ion-organic-bead pairs copy the corresponding water-organic-bead
`A_ij`. The assigned charges are written to `charge_assignment.csv`,
`charge_summary.csv` reports type counts and net system charge, and
`smiles_SP.xlsx` records the counterion and water counts.

The same explicit-charge model is also available in `logp_partition.py` for
logP table fitting and Bayesian optimization. With `--charge-method auto`
(default), neutral targets continue to use ordinary `pair_style dpd`. If a logP
bead has a nonzero charge, for example `Na+`, `Cl-`, `COO-`, or
`CH2OSO3-`, the partition system switches to:

```lammps
atom_style      full
pair_style      dpd/coul/slater/long ...
kspace_style    pppm 1.0e-04
```

Charged logP solutes are neutralized by adding `Na+` or `Cl-` counterions in
the water phase. The counterions are excluded from the solute partition
statistics, while the charged solute molecules remain the logP target. Manual
charge overrides can be supplied with repeated `--bead-charge BEAD=CHARGE`.
Relevant options are `--charge-method`, `--charge-lambda`, `--coul-cutoff`,
`--kspace-accuracy`, and `--charge-unit-scale`.
When logP fitting is launched from `main.py`, the global charge options are
forwarded automatically; use `--logp_bead_charge BEAD=CHARGE` for logP-specific
charge overrides.

Packmol still writes PDB coordinates in its working coordinate scale
(`--box_size * 10`). During data generation, coordinates are scaled by `0.1` so
`packed_polymer_and_solution.data` and `lammps.in` both use the reduced LJ box
set by `--box_size`.

```bash
python main.py --input_data input_data.txt --param_method solubility --charge_method auto
```

To use a custom LAMMPS executable, pass `--lammps_bin` or set `LAMMPS_BIN`.
Without either option, OB-DPD tries `lmp_mpi` first and then `lmp`.
For the locally rebuilt CUDA executable with `dpd/coul/slater/long/gpu` support:

```bash
python main.py --input_data input_data.txt --param_method solubility --job 1 \
  --lammps_bin /home/ls/lammps/lammps-29Aug2024-obdpd-gpu/build/lmp
```
