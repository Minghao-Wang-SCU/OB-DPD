# OB-DPD: Automated Framework for Dissipative Particle Dynamics

**OB-DPD** (One-Button DPD) is an automated framework designed to streamline Dissipative Particle Dynamics simulations, from coarse-grained mapping to property prediction.

## 📦 Installation

You can set up the environment using one of the following two methods:

### Method 1: Using Conda Configuration (Recommended)
If you have internet access, use the provided YAML file to create the environment:

```bash
conda env create -f OB-DPD.yml
conda activate dpd

Method 2: Offline Installation (Using Release Package)
If you are working offline or require the exact pre-compiled environment, download OB-DPD_env.tar.gz from the Releases page.

# 1. Create a directory for the environment
mkdir -p envs/dpd

# 2. Extract the package (replace path with your actual download location)
tar -xzf OB-DPD_env.tar.gz -C envs/dpd

# 3. Activate the environment
source envs/dpd/bin/activate

🚀 Usage
To run the simulation, you need to configure the input_data.txt file and run the main script.

1. Configure input_data.txt
The input file requires strictly 4 lines of parameters:

Line 1 (Mode): Specify the input mode (smiles or mol).

Line 2 (Structure): Provide the SMILES string or the path to the .mol file.

Line 3 (Count): Number of molecules to simulate.

Line 4 (PEG Flag): Whether the structure contains PEG (True or False).

这是为你根据最新要求定制的 README.md，重点清晰地描述了两种安装方式以及 input_data.txt 的详细格式。

你可以直接在服务器上编辑并提交。

第一步：编辑 README.md
在终端运行：

Bash

nano README.md
第二步：复制并粘贴以下内容
Markdown

# OB-DPD: Automated Framework for Dissipative Particle Dynamics

**OB-DPD** (One-Button DPD) is an automated framework designed to streamline Dissipative Particle Dynamics simulations, from coarse-grained mapping to property prediction.

## 📦 Installation

You can set up the environment using one of the following two methods:

### Method 1: Using Conda Configuration (Recommended)
If you have internet access, use the provided YAML file to create the environment:

```bash
conda env create -f OB-DPD.yml
conda activate dpd
Method 2: Offline Installation (Using Release Package)
If you are working offline or require the exact pre-compiled environment, download OB-DPD_env.tar.gz from the Releases page.

Bash

# 1. Create a directory for the environment
mkdir -p envs/dpd

# 2. Extract the package (replace path with your actual download location)
tar -xzf OB-DPD_env.tar.gz -C envs/dpd

# 3. Activate the environment
source envs/dpd/bin/activate
🚀 Usage
To run the simulation, you need to configure the input_data.txt file and run the main script.

1. Configure input_data.txt
The input file requires strictly 4 lines of parameters:

Line 1 (Mode): Specify the input mode (smiles or mol).

Line 2 (Structure): Provide the SMILES string or the path to the .mol file.

Line 3 (Count): Number of molecules to simulate.

Line 4 (PEG Flag): Whether the structure contains PEG (True or False).

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

2. Run the Script
Once the input file is ready, execute the main program:

Bash

python main.py --input_data input_data.txt
