#!/usr/bin/env python
import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("joblib", "joblib"),
    ("rdkit", "rdkit"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("shap", "shap"),
    ("numba", "numba"),
    ("optuna", "optuna"),
    ("statsmodels", "statsmodels"),
    ("MDAnalysis", "MDAnalysis"),
    ("Bio", "biopython"),
    ("openpyxl", "openpyxl"),
    ("PIL", "pillow"),
    ("requests", "requests"),
    ("tqdm", "tqdm"),
]


def check_imports():
    failures = []
    for module_name, package_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"[ok] import {module_name} ({package_name})")
        except Exception as exc:
            print(f"[fail] import {module_name} ({package_name}): {exc}")
            failures.append(package_name)
    return failures


def check_file(path, label):
    if path.exists():
        print(f"[ok] {label}: {path}")
        return True
    print(f"[fail] missing {label}: {path}")
    return False


def check_executable(command, label, optional=False):
    exe = shutil.which(command) if not Path(command).exists() else command
    if not exe:
        status = "warn" if optional else "fail"
        print(f"[{status}] {label} not found: {command}")
        return optional
    print(f"[ok] {label}: {exe}")
    return True


def check_command(command, label, optional=False):
    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
        print(f"[ok] {label} command is callable")
        return True
    except Exception as exc:
        status = "warn" if optional else "fail"
        print(f"[{status}] {label} command failed: {exc}")
        return optional


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--lammps-bin", default="lmp_mpi")
    parser.add_argument("--packmol-bin", default="packmol")
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"[OB-DPD] checking installation under {root}")

    failures = []
    failures.extend(check_imports())

    required_paths = [
        (root / "main.py", "main workflow"),
        (root / "mapping.py", "mapping wrapper"),
        (root / "cg_param_m3.py", "local mapping implementation"),
        (root / "auto_mapping", "auto_mapping source directory"),
        (root / "auto_mapping" / "cg_param_m3-martini3_v2" / "cg_param_m3.py", "auto_mapping Martini 3 script"),
        (root / "finally_xgb_regression.joblib", "ML model"),
        (root / "scaler_params.joblib", "ML scaler"),
        (root / "top_10_feature_indices.csv", "feature index file"),
    ]
    for path, label in required_paths:
        if not check_file(path, label):
            failures.append(label)

    if not args.skip_external:
        if not check_executable(args.packmol_bin, "Packmol"):
            failures.append("packmol")
        if not check_executable("mpirun", "MPI launcher"):
            failures.append("mpirun")
        if not check_executable(args.lammps_bin, "LAMMPS"):
            failures.append("lammps")
        else:
            check_command([args.lammps_bin, "-h"], "LAMMPS", optional=True)

    if failures:
        print("\n[OB-DPD] installation check failed:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\n[OB-DPD] installation check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
