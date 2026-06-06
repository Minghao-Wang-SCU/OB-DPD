#!/usr/bin/env python3
"""Create dual-axis BO history data and plot for target errors and RMSE."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_history_path(path):
    path = Path(path)
    if path.is_dir():
        for name in ("staged_fit_history.csv", "fit_history.csv"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"no staged_fit_history.csv or fit_history.csv found in {path}")
    return path


def read_history(path):
    path = resolve_history_path(path)
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            columns = next(reader)
            rows = [dict(zip(columns, row)) for row in reader if row]
        return pd.DataFrame(rows)


def parse_target_errors(history):
    pattern = re.compile(
        r"^(?P<name>.+):target=(?P<target>[-+0-9.eE]+):"
        r"observed=(?P<observed>[-+0-9.eE]+):error=(?P<error>[-+0-9.eE]+)$"
    )
    rows = []
    for idx, row in history.iterrows():
        iteration = int(row.get("evaluation", row.get("iteration", idx + 1)))
        for item in str(row.get("target_errors", "")).split("|"):
            match = pattern.match(item.strip())
            if match is None:
                continue
            error = float(match.group("error"))
            rows.append(
                {
                    "iteration": iteration,
                    "stage": row.get("stage", ""),
                    "status": row.get("status", ""),
                    "target_name": match.group("name"),
                    "target_logp": float(match.group("target")),
                    "observed_logp": float(match.group("observed")),
                    "error": error,
                    "abs_error": abs(error),
                    "rmse": float(row.get("rmse", "nan")),
                }
            )
    if not rows:
        raise ValueError("no target_errors entries found in history file")
    return pd.DataFrame(rows)


def make_wide_table(long_df):
    base = (
        long_df[["iteration", "stage", "status", "rmse"]]
        .drop_duplicates(subset=["iteration"])
        .sort_values("iteration")
        .reset_index(drop=True)
    )
    for value_col, prefix in (("error", "error"), ("abs_error", "abs_error"), ("observed_logp", "observed_logp")):
        pivot = long_df.pivot(index="iteration", columns="target_name", values=value_col)
        pivot = pivot.rename(columns={col: f"{prefix}_{sanitize_column(col)}" for col in pivot.columns})
        base = base.merge(pivot.reset_index(), on="iteration", how="left")
    return base


def sanitize_column(text):
    return str(text).replace(",", "").replace(" ", "_").replace("-", "_").replace(".", "_")


def plot_dual_axis(long_df, outbase, use_abs_error=True):
    configure_style()
    value_col = "abs_error" if use_abs_error else "error"
    ylabel = "Absolute logP error" if use_abs_error else "logP error (target - DPD)"

    fig, ax_left = plt.subplots(figsize=(7.0, 3.6))
    targets = list(dict.fromkeys(long_df["target_name"].tolist()))
    for idx, target in enumerate(targets):
        sub = long_df[long_df["target_name"] == target].sort_values("iteration")
        ax_left.plot(
            sub["iteration"],
            sub[value_col],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=PALETTE[idx % len(PALETTE)],
            label=target,
        )

    rmse = (
        long_df[["iteration", "rmse"]]
        .drop_duplicates(subset=["iteration"])
        .sort_values("iteration")
    )
    ax_right = ax_left.twinx()
    ax_right.plot(
        rmse["iteration"],
        rmse["rmse"],
        color="#000000",
        linestyle="--",
        linewidth=1.6,
        label="RMSE",
    )

    ax_left.set_xlabel("BO iteration")
    ax_left.set_ylabel(ylabel)
    ax_right.set_ylabel("RMSE")
    ax_left.set_xlim(float(rmse["iteration"].min()) - 0.5, float(rmse["iteration"].max()) + 0.5)
    ax_left.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
    )

    fig.tight_layout()
    fig.savefig(f"{outbase}.png")
    fig.savefig(f"{outbase}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_or_outdir", help="BO output directory or staged_fit_history.csv")
    parser.add_argument("--outdir", help="output directory; default is <run>/figures")
    parser.add_argument("--signed", action="store_true", help="plot signed error instead of absolute error")
    args = parser.parse_args()

    history_path = resolve_history_path(args.history_or_outdir)
    run_dir = history_path.parent
    outdir = Path(args.outdir) if args.outdir else run_dir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    history = read_history(history_path)
    long_df = parse_target_errors(history)
    wide_df = make_wide_table(long_df)

    long_path = outdir / "dual_axis_error_rmse_long.csv"
    wide_path = outdir / "dual_axis_error_rmse_data.csv"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    suffix = "signed_error" if args.signed else "abs_error"
    outbase = outdir / f"dual_axis_{suffix}_rmse"
    plot_dual_axis(long_df, outbase, use_abs_error=not args.signed)
    print(f"wrote {wide_path}")
    print(f"wrote {long_path}")
    print(f"wrote {outbase}.png")
    print(f"wrote {outbase}.pdf")


if __name__ == "__main__":
    main()
