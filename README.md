# OB-DPD: Automated Framework for Dissipative Particle Dynamics

**OB-DPD** (One-Button DPD) is an automated framework designed to streamline Dissipative Particle Dynamics simulations, from coarse-grained mapping to property prediction.

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

1. Configure input_data.txt
The input file requires strictly 4 lines of parameters:

Line 1 (Mode): Specify the input mode (smiles or mol).

Line 2 (Structure): Provide the SMILES string or the path to the .mol file.

Line 3 (Count): Number of molecules to simulate.

Line 4 (PEG Flag): Whether the structure contains PEG (1 or 0).


Example A: SMILES Mode
Plaintext

smiles
CCO(CH2CH2O)10H
50
1

Example B: MOL File Mode
Plaintext

mol
./inputs/drug_molecule.mol
100
0

Example B: SMILES Mode
Plaintext

smiles
CCO(CH2CH2O)10H
50
1
CCCCCO(CH2CH2O)10H
50
0

## 2. Run the Script
Once the input file is ready, execute the main program:

```bash
python main.py --input_data input_data.txt
```

Once the input file is ready, execute the main program. If you wish to customize the initial molecular packing, provide the Packmol input file as an argument:

```bash
Once the input file is ready, execute the main program. If you wish to customize the initial molecular packing, provide the Packmol input file as an argument:
```
