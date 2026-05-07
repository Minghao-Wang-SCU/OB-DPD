# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-driven DPD workflow centered on [`main.py`](/data/wmh/DPD/one-button-dpd/main/OB-DPD/main.py). Core modules live in the repository root: `mapping.py` wraps coarse-graining, `cg_param_m3.py` contains mapping logic, `Pred_Solubility_Parameter.py` runs the ML predictor, and helper utilities such as `smiles2split.py`, `compute_partial_density.py`, and `read_table_data.py` support preprocessing and analysis. Large generated assets and intermediate files also appear at the root, including `.pdb`, `.gro`, `.itp`, `.csv`, `.xlsx`, and `lammps.in`. Reference mapping data is stored under `auto_mapping/`. There is no dedicated `tests/` directory today.

## Build, Test, and Development Commands
Create the environment with `conda env create -f OB-DPD.yml` and activate it with `conda activate dpd`.

Run the main workflow with:
```bash
python main.py --input_data input_data.txt
```

Use a custom Packmol input with:
```bash
python main.py --input_data input_data.txt --packmol_in packmol.inp
```

For quick syntax validation before a commit:
```bash
python -m py_compile main.py cg.py smiles2split.py mapping.py Pred_Solubility_Parameter.py
```

## Coding Style & Naming Conventions
Follow Python 3 style with 4-space indentation and snake_case for functions, variables, and filenames. Keep new modules at the repository root unless there is a clear reason to group them. Prefer explicit file existence checks and informative exceptions because many steps depend on external tools (`packmol`, `mpirun`, `lmp_mpi`) and generated files. Preserve existing CLI flags and file naming patterns such as `0one_chain.pdb`, `0Itp0.itp`, and `packed_polymer_and_solution.data`.

## Testing Guidelines
There is no formal automated test suite yet. At minimum, run `py_compile` on edited scripts and then exercise the affected path with a small `input_data.txt` example. If you add logic that transforms files, verify both console output and generated artifacts. For new tests, prefer `pytest` and place them under a new `tests/` directory using names like `test_mapping.py`.

## Commit & Pull Request Guidelines
The current history uses short, imperative commit messages, for example: `Initial commit with proper ignore rules`. Keep that style: concise subject line, focused change set. Pull requests should include a summary of behavior changes, commands used for validation, sample input files when relevant, and notes about generated artifacts or external tool requirements.

## Configuration & Data Notes
Do not commit large regenerated outputs unless they are intentional fixtures. Model files (`*.joblib`, feature index CSVs) are runtime dependencies; changes to them should explain provenance and compatibility.
