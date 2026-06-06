#!/usr/bin/env python3
"""Plot target vs simulated logP for the best staged BO evaluation."""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_PATTERN = re.compile(
    r"^(?P<name>.+):target=(?P<target>[-+0-9.eE]+):"
    r"observed=(?P<observed>[-+0-9.eE]+):error=(?P<error>[-+0-9.eE]+)$"
)


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_best_json(path):
    path = Path(path)
    if path.is_dir():
        candidate = path / "staged_best.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"staged_best.json not found in {path}")
    return path


def parse_target_errors(text):
    rows = []
    for item in str(text).split("|"):
        item = item.strip()
        if not item:
            continue
        match = TARGET_PATTERN.match(item)
        if match is None:
            raise ValueError(f"could not parse target error entry: {item}")
        target = float(match.group("target"))
        observed = float(match.group("observed"))
        error = float(match.group("error"))
        rows.append(
            {
                "target_name": match.group("name"),
                "target_logp": target,
                "simulated_logp": observed,
                "error": error,
                "abs_error": abs(error),
            }
        )
    if not rows:
        raise ValueError("no target logP values found in best stage")
    return pd.DataFrame(rows)


def load_best_stage(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    stages = data.get("stages", [])
    if not stages:
        raise ValueError(f"no stages found in {path}")
    best = min(stages, key=lambda row: float(row.get("rmse", math.inf)))
    targets = parse_target_errors(best["target_errors"])
    for key in ["stage", "evaluation", "rmse", "max_abs_error", "status"]:
        targets[key] = best.get(key, "")
    return best, targets


def r2_score(target, observed):
    target = np.asarray(target, dtype=float)
    observed = np.asarray(observed, dtype=float)
    sse = float(np.sum(np.square(target - observed)))
    sst = float(np.sum(np.square(target - np.mean(target))))
    return float(1.0 - sse / sst), sse, sst


def plot_best_r2(targets, outdir, stem):
    r2, sse, sst = r2_score(targets["target_logp"], targets["simulated_logp"])
    rmse = float(np.sqrt(np.mean(np.square(targets["target_logp"] - targets["simulated_logp"]))))
    max_abs = float(targets["abs_error"].max())

    summary = {
        "r2": r2,
        "rmse": rmse,
        "max_abs_error": max_abs,
        "sse": sse,
        "sst": sst,
        "n_targets": int(len(targets)),
        "stage": targets["stage"].iloc[0],
        "evaluation": int(targets["evaluation"].iloc[0]),
        "status": targets["status"].iloc[0],
    }
    for key, value in summary.items():
        targets[key] = value
    targets.to_csv(outdir / f"{stem}.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.45, 3.35))
    ax.scatter(
        targets["target_logp"],
        targets["simulated_logp"],
        s=48,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for _, row in targets.iterrows():
        ax.annotate(
            row["target_name"],
            (row["target_logp"], row["simulated_logp"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )

    min_value = float(min(targets["target_logp"].min(), targets["simulated_logp"].min()))
    max_value = float(max(targets["target_logp"].max(), targets["simulated_logp"].max()))
    pad = max(0.12, 0.08 * (max_value - min_value if max_value > min_value else 1.0))
    lo = min_value - pad
    hi = max_value + pad
    ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1.0, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target logP")
    ax.set_ylabel("Simulated logP")
    ax.set_title("Best BO logP fit")
    ax.text(
        0.05,
        0.95,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nEval. = {summary['evaluation']}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Plot best BO target-vs-simulated logP R2")
    parser.add_argument("path", help="BO output directory or staged_best.json")
    parser.add_argument("--outdir", help="output directory; default is <BO dir>/figures")
    parser.add_argument("--stem", default="best_logp_r2", help="output file stem")
    args = parser.parse_args()

    configure_style()
    best_json = resolve_best_json(args.path)
    outdir = Path(args.outdir) if args.outdir else best_json.parent / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    _, targets = load_best_stage(best_json)
    summary = plot_best_r2(targets, outdir, args.stem)
    print(
        "wrote "
        f"{outdir / (args.stem + '.png')}, "
        f"{outdir / (args.stem + '.pdf')}, and "
        f"{outdir / (args.stem + '.csv')}"
    )
    print(
        f"best evaluation={summary['evaluation']}, "
        f"R2={summary['r2']:.6f}, "
        f"RMSE={summary['rmse']:.6f}"
    )


if __name__ == "__main__":
    main()
