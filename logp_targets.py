#!/usr/bin/env python3
"""Collect predicted logP targets and build a robust consensus."""

import argparse
import json
import math
import shutil
import subprocess
import urllib.parse
import urllib.request

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen


def rdkit_crippen_logp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return float(Crippen.MolLogP(mol))


def openbabel_logp(smiles):
    obabel = shutil.which("obabel")
    if not obabel:
        return None
    process = subprocess.run(
        [obabel, f"-:{smiles}", "-osmi", "--append", "logP"],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0 or not process.stdout.strip():
        return None
    for token in reversed(process.stdout.replace("\t", " ").split()):
        try:
            return float(token)
        except ValueError:
            continue
    return None


def pubchem_xlogp(smiles, timeout=20):
    encoded = urllib.parse.quote(smiles, safe="")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/property/XLogP/JSON"
    request = urllib.request.Request(url, headers={"User-Agent": "OB-DPD-logp-target/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    props = payload.get("PropertyTable", {}).get("Properties", [])
    if not props or props[0].get("XLogP") is None:
        return None
    return float(props[0]["XLogP"])


def parse_external_values(values):
    rows = []
    for value in values or []:
        try:
            method, number = value.split("=", 1)
            rows.append(
                {
                    "method": method.strip(),
                    "value": float(number),
                    "status": "ok",
                    "source": "user",
                }
            )
        except ValueError as exc:
            raise ValueError(f"external logP value must look like METHOD=VALUE, got {value}") from exc
    return rows


def collect_logp_values(smiles, external_values=None, use_pubchem=True, use_openbabel=True):
    rows = []
    try:
        rows.append({"method": "RDKit_Crippen", "value": rdkit_crippen_logp(smiles), "status": "ok", "source": "local"})
    except Exception as exc:
        rows.append({"method": "RDKit_Crippen", "value": None, "status": f"failed: {exc}", "source": "local"})

    if use_openbabel:
        try:
            value = openbabel_logp(smiles)
            status = "ok" if value is not None else "unavailable"
            rows.append({"method": "OpenBabel", "value": value, "status": status, "source": "local"})
        except Exception as exc:
            rows.append({"method": "OpenBabel", "value": None, "status": f"failed: {exc}", "source": "local"})

    if use_pubchem:
        try:
            value = pubchem_xlogp(smiles)
            status = "ok" if value is not None else "unavailable"
            rows.append({"method": "PubChem_XLogP", "value": value, "status": status, "source": "pubchem"})
        except Exception as exc:
            rows.append({"method": "PubChem_XLogP", "value": None, "status": f"failed: {exc}", "source": "pubchem"})

    rows.extend(parse_external_values(external_values))
    return rows


def _quartile(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q))


def robust_consensus(rows, outlier_method="mad", mad_z=3.5, iqr_factor=1.5, min_methods=2):
    valid = [row for row in rows if row.get("value") is not None and math.isfinite(float(row["value"]))]
    if len(valid) < min_methods:
        raise ValueError(f"need at least {min_methods} valid logP estimates, got {len(valid)}")

    values = [float(row["value"]) for row in valid]
    keep = [True] * len(valid)
    method = outlier_method.lower()

    if method == "mad" and len(values) >= 3:
        median = float(np.median(values))
        deviations = [abs(value - median) for value in values]
        mad = float(np.median(deviations))
        if mad > 0:
            keep = [0.6745 * abs(value - median) / mad <= mad_z for value in values]
    elif method == "iqr" and len(values) >= 4:
        q1 = _quartile(values, 0.25)
        q3 = _quartile(values, 0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        keep = [lower <= value <= upper for value in values]
    elif method != "none":
        pass

    kept = []
    rejected = []
    for row, is_kept in zip(valid, keep):
        copied = dict(row)
        copied["outlier"] = not is_kept
        if is_kept:
            kept.append(copied)
        else:
            rejected.append(copied)
    if not kept:
        kept = [dict(row, outlier=False) for row in valid]
        rejected = []

    kept_values = np.asarray([float(row["value"]) for row in kept], dtype=float)
    mean = float(np.mean(kept_values))
    variance = float(np.var(kept_values, ddof=1)) if len(kept_values) > 1 else 0.0
    std = float(math.sqrt(variance))
    return {
        "target_logp": mean,
        "target_variance": variance,
        "target_std": std,
        "target_lower_1std": mean - std,
        "target_upper_1std": mean + std,
        "outlier_method": outlier_method,
        "valid_count": len(valid),
        "kept_count": len(kept),
        "kept": kept,
        "rejected": rejected,
        "all_methods": rows,
    }


def write_logp_report(path, report):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Estimate a robust target logP from multiple methods")
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--external", action="append", default=[], help="external estimate as METHOD=VALUE")
    parser.add_argument("--outlier-method", choices=["mad", "iqr", "none"], default="mad")
    parser.add_argument("--min-methods", type=int, default=2)
    parser.add_argument("--no-pubchem", action="store_true")
    parser.add_argument("--no-openbabel", action="store_true")
    parser.add_argument("--output", default="target_logp_report.json")
    args = parser.parse_args()
    rows = collect_logp_values(
        args.smiles,
        external_values=args.external,
        use_pubchem=not args.no_pubchem,
        use_openbabel=not args.no_openbabel,
    )
    report = robust_consensus(rows, outlier_method=args.outlier_method, min_methods=args.min_methods)
    write_logp_report(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
