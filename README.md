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
portable Conda file to install only the Python dependencies:

```bash
conda env create -f environment.yml
conda activate dpd
```

Verify that the external executables are available:

```bash
packmol < /dev/null
mpirun --version
lmp_mpi -h
```

If your LAMMPS executable has a different name or path, pass it explicitly:

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

This path does not call the ML solubility-parameter predictor and does not use
solubility fallback values. Unsupported bead types or bead pairs stop the
workflow and are reported through `bead_type_assignment.csv` and
`a_ij_source.csv`. The atom-to-logP-bead grouping is written to files such as
`0logp_article_mapping0.csv`.

Use `logp_partition.py` separately for water/octanol partition sampling and
research calibration of selected `A_ij` values.

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

Packmol still writes PDB coordinates in its working coordinate scale
(`--box_size * 10`). During data generation, coordinates are scaled by `0.1` so
`packed_polymer_and_solution.data` and `lammps.in` both use the reduced LJ box
set by `--box_size`.

```bash
python main.py --input_data input_data.txt --param_method solubility --charge_method auto
```

To use a custom LAMMPS executable, pass `--lammps_bin` or set `LAMMPS_BIN`.
For the locally rebuilt CUDA executable with `dpd/coul/slater/long/gpu` support:

```bash
python main.py --input_data input_data.txt --param_method solubility --job 1 \
  --lammps_bin /home/ls/lammps/lammps-29Aug2024-obdpd-gpu/build/lmp
```
