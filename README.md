# OB-DPD: Automated Framework for Dissipative Particle Dynamics

**OB-DPD** (One-Button DPD) is an automated framework designed to streamline Dissipative Particle Dynamics simulations, from coarse-grained mapping to property prediction.

## 🛠️ Prerequisites

Before installing and running OB-DPD, please ensure you have the following prepared:

1. **External Software:**
   * **Packmol:** Required for generating the initial molecular packing configurations. 
   * **LAMMPS:** Required for executing the Dissipative Particle Dynamics simulations. 
   *(Please ensure both are properly installed and accessible in your system's PATH.)*

2. **Python Dependencies:**
   * All required Python packages are listed in the `requirements.txt` file. 
   * ⚠️ **Important:** Please pay special attention to the exact versions of `rdkit`, `numpy`, `pandas`, `shap`, `seaborn`, `matplotlib`, `numba`, and `scipy`. Strict adherence to these versions is highly recommended to avoid compatibility issues during ML predictions and data processing.
## 📦 Installation 

You can set up the environment using one of the following two methods:

### Method 1: Using Conda Configuration (Recommended)
If you have internet access, use the provided YAML file to create the environment:

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
