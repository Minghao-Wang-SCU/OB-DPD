#!/usr/bin/env python3
"""Plot solubility-parameter prediction errors against polarity descriptors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from rdkit import Chem
from rdkit.Chem import Descriptors

import matplotlib as mpl
import matplotlib.pyplot as plt


OKABE_ITO = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#8C8C8C",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polar_heavy_atoms(mol: Chem.Mol) -> int:
    polar = {7, 8, 9, 15, 16, 17, 35, 53}
    return sum(atom.GetAtomicNum() in polar for atom in mol.GetAtoms())


def nonpolar_heavy_atoms(mol: Chem.Mol) -> int:
    return sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


def hydrogen_atoms(mol: Chem.Mol) -> int:
    explicit = sum(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    implicit = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
    return int(explicit + implicit)


def create_all_features(mol: Chem.Mol) -> tuple[list[float], list[str]]:
    features: list[float] = []
    names: list[str] = []
    for name, func in Descriptors.descList:
        names.append(name)
        features.append(func(mol))
    names.extend(["polar_heavy_atoms", "nonpolar_heavy_atoms", "hydrogen_atoms"])
    features.extend([polar_heavy_atoms(mol), nonpolar_heavy_atoms(mol), hydrogen_atoms(mol)])
    return features, names


def load_fragment_table(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    rename = {}
    for col in df.columns:
        lowered = str(col).strip().lower()
        if lowered in {"smiles", "smile"}:
            rename[col] = "smiles"
        elif lowered in {"solubility parameter", "target", "y", "label"}:
            rename[col] = "target_solubility_parameter"
        elif lowered == "id":
            rename[col] = "id"
    df = df.rename(columns=rename)
    required = {"smiles", "target_solubility_parameter"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required column(s) in {path}: {missing}")
    return df


def predict(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    feature_rows = []
    feature_names: list[str] | None = None
    for row in df.itertuples(index=False):
        smiles = str(getattr(row, "smiles")).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rows.append({**row._asdict(), "valid_smiles": False})
            continue
        features, names = create_all_features(mol)
        if feature_names is None:
            feature_names = names
        feature_rows.append(features)
        rows.append(
            {
                **row._asdict(),
                "valid_smiles": True,
                "canonical_smiles": Chem.MolToSmiles(mol),
                "TPSA": Descriptors.TPSA(mol),
                "MolLogP": Descriptors.MolLogP(mol),
                "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
                "polar_heavy_atoms": polar_heavy_atoms(mol),
                "nonpolar_heavy_atoms": nonpolar_heavy_atoms(mol),
            }
        )

    valid = pd.DataFrame([r for r in rows if r.get("valid_smiles")])
    if valid.empty:
        raise ValueError("No valid SMILES were found.")

    features_df = pd.DataFrame(feature_rows, columns=feature_names)
    scaler = load(root / "scaler_params.joblib")
    model = load(root / "finally_xgb_regression.joblib")
    top_indices = pd.read_csv(root / "top_10_feature_indices.csv").iloc[:, 0].astype(int).to_numpy()
    x_scaled = scaler.transform(features_df)
    y_pred = model.predict(x_scaled[:, top_indices].reshape(-1, len(top_indices)))

    valid["predicted_solubility_parameter"] = y_pred
    valid["error"] = valid["predicted_solubility_parameter"] - valid["target_solubility_parameter"]
    valid["absolute_error"] = valid["error"].abs()
    return valid


def regression_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.unique(x[mask]).size < 2:
        return None
    coef = np.polyfit(x[mask], y[mask], deg=1)
    xs = np.linspace(float(np.nanmin(x[mask])), float(np.nanmax(x[mask])), 100)
    return xs, np.polyval(coef, xs)


def binned_summary(df: pd.DataFrame, column: str, bins: int = 6) -> pd.DataFrame:
    tmp = df[[column, "error", "absolute_error"]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[column], q=min(bins, tmp[column].nunique()), duplicates="drop")
    grouped = tmp.groupby("bin", observed=True)
    out = grouped.agg(
        descriptor_min=(column, "min"),
        descriptor_max=(column, "max"),
        descriptor_mean=(column, "mean"),
        mean_error=("error", "mean"),
        sem_error=("error", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        mean_absolute_error=("absolute_error", "mean"),
        sem_absolute_error=("absolute_error", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        n=("absolute_error", "size"),
    ).reset_index(drop=True)
    out.insert(0, "descriptor", column)
    return out


def save_error_distribution(df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    ax = axes[0]
    ax.hist(df["error"], bins=28, color=OKABE_ITO["blue"], alpha=0.82, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=OKABE_ITO["black"], linewidth=0.9)
    ax.axvline(df["error"].mean(), color=OKABE_ITO["vermillion"], linewidth=1.0, linestyle="--", label="Mean")
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    ax.set_title("Signed error")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.hist(df["absolute_error"], bins=28, color=OKABE_ITO["orange"], alpha=0.86, edgecolor="white", linewidth=0.5)
    ax.axvline(df["absolute_error"].median(), color=OKABE_ITO["black"], linewidth=1.0, linestyle="--", label="Median")
    ax.set_xlabel("Absolute prediction error")
    ax.set_ylabel("Count")
    ax.set_title("Absolute error")
    ax.legend(frameon=False)

    for label, ax in zip(("a", "b"), axes):
        ax.text(-0.17, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.6)
    for ext in ("svg", "png", "pdf"):
        fig.savefig(outdir / f"prediction_error_distribution.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_error_vs_polarity(df: pd.DataFrame, binned: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), constrained_layout=True)
    specs = [
        ("TPSA", "TPSA", OKABE_ITO["green"]),
        ("MolLogP", "MolLogP", OKABE_ITO["purple"]),
    ]
    for label, ax, (column, xlabel, color) in zip(("a", "b"), axes, specs):
        sc = ax.scatter(
            df[column],
            df["error"],
            c=df["absolute_error"],
            cmap="viridis",
            s=22,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.25,
        )
        line = regression_line(df[column].to_numpy(float), df["error"].to_numpy(float))
        if line is not None:
            ax.plot(line[0], line[1], color=OKABE_ITO["black"], linewidth=1.1, linestyle="--", label="Linear trend")
        sub = binned[binned["descriptor"] == column]
        ax.errorbar(
            sub["descriptor_mean"],
            sub["mean_error"],
            yerr=sub["sem_error"],
            color=color,
            marker="o",
            markersize=4.0,
            linewidth=1.2,
            capsize=2.0,
            label="Binned mean",
        )
        ax.axhline(0, color=OKABE_ITO["gray"], linewidth=0.9, linestyle=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Prediction error")
        ax.text(-0.17, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")
        ax.legend(frameon=False, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label("Absolute error")
    for ext in ("svg", "png", "pdf"):
        fig.savefig(outdir / f"prediction_error_vs_polarity.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def kde_curve(values: np.ndarray, points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("At least two finite values are required for KDE.")
    std = float(np.std(values, ddof=1))
    if std == 0:
        std = 1.0
    bandwidth = 1.06 * std * len(values) ** (-1 / 5)
    bandwidth = max(bandwidth, 1e-6)
    pad = 0.08 * (float(values.max()) - float(values.min()) or 1.0)
    x = np.linspace(float(values.min()) - pad, float(values.max()) + pad, points)
    z = (x[:, None] - values[None, :]) / bandwidth
    y = np.exp(-0.5 * z**2).sum(axis=1) / (len(values) * bandwidth * np.sqrt(2 * np.pi))
    return x, y


def save_error_logp_distribution_curves(df: pd.DataFrame, outdir: Path) -> None:
    err_x, err_density = kde_curve(df["error"].to_numpy(float))
    logp_x, logp_density = kde_curve(df["MolLogP"].to_numpy(float))

    curve_data = pd.DataFrame(
        {
            "error_x": err_x,
            "error_density": err_density,
            "MolLogP_x": logp_x,
            "MolLogP_density": logp_density,
        }
    )
    curve_data.to_csv(outdir / "error_logp_distribution_curve_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
    axes[0].plot(err_x, err_density, color=OKABE_ITO["blue"], linewidth=1.8)
    axes[0].fill_between(err_x, err_density, color=OKABE_ITO["blue"], alpha=0.22, linewidth=0)
    axes[0].axvline(0, color=OKABE_ITO["black"], linewidth=0.9, linestyle="--")
    axes[0].set_xlabel("Prediction error")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Error distribution")

    axes[1].plot(logp_x, logp_density, color=OKABE_ITO["purple"], linewidth=1.8)
    axes[1].fill_between(logp_x, logp_density, color=OKABE_ITO["purple"], alpha=0.22, linewidth=0)
    axes[1].axvline(df["MolLogP"].median(), color=OKABE_ITO["black"], linewidth=0.9, linestyle="--")
    axes[1].set_xlabel("MolLogP")
    axes[1].set_ylabel("Density")
    axes[1].set_title("MolLogP distribution")

    for label, ax in zip(("a", "b"), axes):
        ax.text(-0.17, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.6)
    for ext in ("svg", "png", "pdf"):
        fig.savefig(outdir / f"error_logp_distribution_curves.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, outdir: Path) -> None:
    y = df["target_solubility_parameter"].to_numpy(float)
    yhat = df["predicted_solubility_parameter"].to_numpy(float)
    err = yhat - y
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    summary = pd.DataFrame(
        [
            {
                "n": len(df),
                "mean_error": float(np.mean(err)),
                "std_error": float(np.std(err, ddof=1)),
                "mae": float(np.mean(np.abs(err))),
                "median_absolute_error": float(np.median(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                "min_error": float(np.min(err)),
                "max_error": float(np.max(err)),
                "mean_tpsa": float(df["TPSA"].mean()),
                "mean_mollogp": float(df["MolLogP"].mean()),
            }
        ]
    )
    summary.to_csv(outdir / "prediction_error_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/output.xlsx", help="Fragment database Excel file.")
    parser.add_argument("--outdir", default="analysis/solubility_prediction_error_polarity", help="Output directory.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = (root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    outdir = (root / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    apply_style()
    source = load_fragment_table(input_path)
    results = predict(source, root)
    finite_cols = [
        "target_solubility_parameter",
        "predicted_solubility_parameter",
        "error",
        "absolute_error",
        "TPSA",
        "MolLogP",
    ]
    finite_mask = np.ones(len(results), dtype=bool)
    for col in finite_cols:
        finite_mask &= np.isfinite(results[col].to_numpy(float))
    results["used_for_error_analysis"] = finite_mask
    results.to_csv(outdir / "prediction_error_polarity_data.csv", index=False)

    plot_data = results[results["used_for_error_analysis"]].copy()
    if plot_data.empty:
        raise ValueError("No rows with finite target, prediction, error, TPSA and MolLogP values.")

    binned = pd.concat([binned_summary(plot_data, "TPSA"), binned_summary(plot_data, "MolLogP")], ignore_index=True)
    binned.to_csv(outdir / "prediction_error_polarity_binned_summary.csv", index=False)
    write_summary(plot_data, outdir)
    save_error_distribution(plot_data, outdir)
    save_error_vs_polarity(plot_data, binned, outdir)
    save_error_logp_distribution_curves(plot_data, outdir)

    summary = pd.read_csv(outdir / "prediction_error_summary.csv").iloc[0]
    print(f"Wrote {outdir}")
    print(
        "n={n:.0f}, MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}".format(
            n=summary["n"], mae=summary["mae"], rmse=summary["rmse"], r2=summary["r2"]
        )
    )


if __name__ == "__main__":
    main()
