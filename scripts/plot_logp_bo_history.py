#!/usr/bin/env python3
"""Plot Bayesian logP fitting history from OB-DPD outputs."""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
    "gray": "#666666",
}


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
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_history_path(path):
    path = Path(path)
    if path.is_dir():
        for name in ("fit_history.csv", "staged_fit_history.csv"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"no fit_history.csv or staged_fit_history.csv found in {path}")
    return path


def is_number(text):
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def parse_legacy_unquoted_staged(fields, columns):
    if len(fields) == len(columns):
        return dict(zip(columns, fields))
    if len(fields) < len(columns):
        raise ValueError("too few fields")
    prefix = {
        "stage": fields[0],
        "evaluation": fields[1],
        "optimizer": fields[2],
        "status": fields[3],
    }
    metric_index = None
    for idx in range(4, len(fields) - 6):
        if is_number(fields[idx]) and is_number(fields[idx + 1]) and is_number(fields[idx + 2]):
            metric_index = idx
            break
    if metric_index is None:
        raise ValueError("could not locate rmse/max_abs_error/effective_tolerance fields")
    prefix["failure_reason"] = ",".join(fields[4:metric_index])
    prefix["rmse"] = fields[metric_index]
    prefix["max_abs_error"] = fields[metric_index + 1]
    prefix["effective_tolerance"] = fields[metric_index + 2]
    prefix["fit_pairs"] = fields[metric_index + 3]
    prefix["fit_aij"] = fields[metric_index + 4]
    prefix["target_errors"] = ",".join(fields[metric_index + 5 : -1])
    prefix["run_dir"] = fields[-1]
    return prefix


def parse_legacy_unquoted_single(fields, columns):
    if len(fields) == len(columns):
        return dict(zip(columns, fields))
    if len(fields) < len(columns):
        raise ValueError("too few fields")
    prefix = {
        "iteration": fields[0],
        "optimizer": fields[1],
        "status": fields[2],
    }
    target_index = None
    for idx in range(3, len(fields) - 10):
        numeric_window = fields[idx : idx + 8]
        if all(is_number(item) or item in {"", "None", "nan", "NaN"} for item in numeric_window):
            target_index = idx
            break
    if target_index is None:
        raise ValueError("could not locate target/objective fields")
    prefix["failure_reason"] = ",".join(fields[3:target_index])
    for offset, name in enumerate(
        [
            "target_logp",
            "target_std",
            "effective_tolerance",
            "observed_logp",
            "error",
            "abs_error",
            "objective",
        ]
    ):
        prefix[name] = fields[target_index + offset]
    prefix["fit_pairs"] = fields[-3]
    prefix["fit_aij"] = fields[-2]
    prefix["run_dir"] = fields[-1]
    return prefix


def read_history_csv(path):
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            columns = next(reader)
            rows = []
            for fields in reader:
                if not fields:
                    continue
                if "rmse" in columns:
                    rows.append(parse_legacy_unquoted_staged(fields, columns))
                else:
                    rows.append(parse_legacy_unquoted_single(fields, columns))
        return pd.DataFrame(rows, columns=columns)


def parse_aij_series(df):
    rows = []
    if "fit_aij" not in df.columns:
        return pd.DataFrame(columns=["iteration", "pair", "Aij"])
    for _, row in df.iterrows():
        iteration = int(row["iteration"]) if "iteration" in row and not pd.isna(row["iteration"]) else int(row.name + 1)
        text = str(row.get("fit_aij", ""))
        for item in text.split(";"):
            item = item.strip()
            if not item or "=" not in item:
                continue
            pair, value = item.rsplit("=", 1)
            try:
                rows.append({"iteration": iteration, "pair": pair, "Aij": float(value)})
            except ValueError:
                continue
    return pd.DataFrame(rows)


def parse_target_errors(df):
    rows = []
    if "target_errors" not in df.columns:
        return pd.DataFrame(
            columns=[
                "stage",
                "evaluation",
                "target_name",
                "target",
                "observed",
                "error",
                "abs_error",
            ]
        )
    pattern = re.compile(
        r"^(?P<name>.+):target=(?P<target>[-+0-9.eE]+):"
        r"observed=(?P<observed>[-+0-9.eE]+):error=(?P<error>[-+0-9.eE]+)$"
    )
    for _, row in df.iterrows():
        text = str(row.get("target_errors", ""))
        for item in text.split("|"):
            item = item.strip()
            if not item:
                continue
            match = pattern.match(item)
            if match is None:
                continue
            error = float(match.group("error"))
            rows.append(
                {
                    "stage": row.get("stage", ""),
                    "evaluation": int(row.get("evaluation", row.get("iteration", len(rows) + 1))),
                    "status": row.get("status", ""),
                    "target_name": match.group("name"),
                    "target": float(match.group("target")),
                    "observed": float(match.group("observed")),
                    "error": error,
                    "abs_error": abs(error),
                }
            )
    return pd.DataFrame(rows)


def calculate_staged_r2(targets):
    columns = [
        "stage",
        "evaluation",
        "status",
        "n_targets",
        "r2",
        "sse",
        "sst",
        "rmse",
        "max_abs_error",
    ]
    if targets.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    group_cols = ["stage", "evaluation"]
    for (stage, evaluation), sub in targets.groupby(group_cols, dropna=False):
        target = pd.to_numeric(sub["target"], errors="coerce").to_numpy(dtype=float)
        observed = pd.to_numeric(sub["observed"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(target) & np.isfinite(observed)
        target = target[mask]
        observed = observed[mask]
        n_targets = int(len(target))
        if n_targets == 0:
            sse = sst = rmse = max_abs = r2 = math.nan
        else:
            errors = target - observed
            sse = float(np.sum(np.square(errors)))
            sst = float(np.sum(np.square(target - np.mean(target))))
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            max_abs = float(np.max(np.abs(errors)))
            r2 = float(1.0 - sse / sst) if n_targets > 1 and sst > 0.0 else math.nan
        statuses = sorted(set(str(item) for item in sub.get("status", pd.Series(dtype=str)).dropna()))
        rows.append(
            {
                "stage": stage,
                "evaluation": int(evaluation),
                "status": ";".join(statuses),
                "n_targets": n_targets,
                "r2": r2,
                "sse": sse,
                "sst": sst,
                "rmse": rmse,
                "max_abs_error": max_abs,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["stage", "evaluation"])


def metric_columns(df):
    if "observed_logp" in df.columns:
        metric = "abs_error" if "abs_error" in df.columns else "objective"
        return "single", metric
    if "rmse" in df.columns:
        return "staged", "rmse"
    raise ValueError("history file must contain observed_logp or rmse columns")


def add_status_markers(ax, df, y_col):
    if "status" not in df.columns or y_col not in df.columns:
        return
    failed = df["status"].astype(str).str.lower() != "ok"
    if failed.any():
        ax.scatter(
            df.loc[failed, "iteration"],
            df.loc[failed, y_col],
            s=36,
            marker="x",
            color=PALETTE["red"],
            label="failed/penalized",
            zorder=5,
        )


def save_figure(fig, stem, outdir):
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def slugify(value):
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value).strip())
    return text.strip("_") or "item"


def save_panel(fig, stem, outdir, data=None):
    if data is not None:
        data.to_csv(outdir / f"{stem}_data.csv", index=False)
    return save_figure(fig, stem, outdir)


def clean_iteration(series, n_rows):
    values = pd.to_numeric(series, errors="coerce")
    fallback = pd.Series(np.arange(1, n_rows + 1), index=values.index)
    return values.fillna(fallback).astype(int)


def plot_single_target(df, outdir):
    df = df.copy()
    df["iteration"] = clean_iteration(df["iteration"], len(df))
    df["observed_logp"] = pd.to_numeric(df["observed_logp"], errors="coerce")
    if "target_logp" in df.columns:
        df["target_logp"] = pd.to_numeric(df["target_logp"], errors="coerce")
    if "abs_error" in df.columns:
        df["abs_error"] = pd.to_numeric(df["abs_error"], errors="coerce")
    if "objective" in df.columns:
        df["objective"] = pd.to_numeric(df["objective"], errors="coerce")

    target = float(df["target_logp"].dropna().iloc[0]) if "target_logp" in df.columns and df["target_logp"].notna().any() else None
    best_abs = df["abs_error"].cummin() if "abs_error" in df.columns else None

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8))
    ax = axes[0, 0]
    ax.plot(df["iteration"], df["observed_logp"], "-o", color=PALETTE["blue"], lw=1.6, ms=4, label="simulated logP")
    if target is not None:
        ax.axhline(target, color=PALETTE["black"], lw=1.2, ls="--", label=f"target = {target:.4g}")
    add_status_markers(ax, df, "observed_logp")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("logP")
    ax.set_title("Simulated logP")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    if "abs_error" in df.columns:
        ax.plot(df["iteration"], df["abs_error"], "-o", color=PALETTE["orange"], lw=1.6, ms=4, label="absolute error")
        ax.plot(df["iteration"], best_abs, "-", color=PALETTE["green"], lw=1.6, label="best so far")
        if "effective_tolerance" in df.columns and df["effective_tolerance"].notna().any():
            tol = float(pd.to_numeric(df["effective_tolerance"], errors="coerce").dropna().iloc[0])
            ax.axhline(tol, color=PALETTE["black"], lw=1.0, ls=":", label=f"tolerance = {tol:.3g}")
        ax.set_ylabel("|target - simulated|")
        ax.legend(frameon=False)
    ax.set_xlabel("Iteration")
    ax.set_title("Error")

    ax = axes[1, 0]
    if "objective" in df.columns:
        ax.plot(df["iteration"], df["objective"], "-o", color=PALETTE["purple"], lw=1.6, ms=4)
        ax.set_ylabel("Objective")
    ax.set_xlabel("Iteration")
    ax.set_title("Loss")

    ax = axes[1, 1]
    if "error" in df.columns:
        err = pd.to_numeric(df["error"], errors="coerce")
        ax.axhline(0.0, color=PALETTE["black"], lw=1.0)
        ax.bar(df["iteration"], err, color=np.where(err >= 0.0, PALETTE["sky"], PALETTE["red"]), alpha=0.85)
        ax.set_ylabel("target - simulated")
    ax.set_xlabel("Iteration")
    ax.set_title("Signed error")

    fig.tight_layout()
    save_figure(fig, "logp_bo_overview", outdir)

    sim_data = df[["iteration", "observed_logp", "status"]].copy()
    if "target_logp" in df.columns:
        sim_data["target_logp"] = df["target_logp"]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot(df["iteration"], df["observed_logp"], "-o", color=PALETTE["blue"], lw=1.6, ms=4, label="simulated logP")
    if target is not None:
        ax.axhline(target, color=PALETTE["black"], lw=1.2, ls="--", label=f"target = {target:.4g}")
    add_status_markers(ax, df, "observed_logp")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("logP")
    ax.set_title("Simulated logP")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_panel(fig, "single_simulated_logp", outdir, sim_data)

    if "abs_error" in df.columns:
        error_data = df[["iteration", "abs_error", "status"]].copy()
        error_data["best_abs_error_so_far"] = best_abs
        if "effective_tolerance" in df.columns:
            error_data["effective_tolerance"] = df["effective_tolerance"]
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ax.plot(df["iteration"], df["abs_error"], "-o", color=PALETTE["orange"], lw=1.6, ms=4, label="absolute error")
        ax.plot(df["iteration"], best_abs, "-", color=PALETTE["green"], lw=1.6, label="best so far")
        if "effective_tolerance" in df.columns and df["effective_tolerance"].notna().any():
            tol = float(pd.to_numeric(df["effective_tolerance"], errors="coerce").dropna().iloc[0])
            ax.axhline(tol, color=PALETTE["black"], lw=1.0, ls=":", label=f"tolerance = {tol:.3g}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("|target - simulated|")
        ax.set_title("Absolute error")
        ax.legend(frameon=False)
        fig.tight_layout()
        save_panel(fig, "single_absolute_error", outdir, error_data)

    if "objective" in df.columns:
        loss_data = df[["iteration", "objective", "status"]].copy()
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ax.plot(df["iteration"], df["objective"], "-o", color=PALETTE["purple"], lw=1.6, ms=4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective")
        ax.set_title("Loss")
        fig.tight_layout()
        save_panel(fig, "single_loss", outdir, loss_data)

    if "error" in df.columns:
        signed_data = df[["iteration", "error", "status"]].copy()
        err = pd.to_numeric(df["error"], errors="coerce")
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ax.axhline(0.0, color=PALETTE["black"], lw=1.0)
        ax.bar(df["iteration"], err, color=np.where(err >= 0.0, PALETTE["sky"], PALETTE["red"]), alpha=0.85)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("target - simulated")
        ax.set_title("Signed error")
        fig.tight_layout()
        save_panel(fig, "single_signed_error", outdir, signed_data)


def plot_staged(df, outdir):
    df = df.copy()
    df["iteration"] = clean_iteration(df["evaluation"], len(df))
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    if "max_abs_error" in df.columns:
        df["max_abs_error"] = pd.to_numeric(df["max_abs_error"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    ax = axes[0]
    for stage, sub in df.groupby("stage", dropna=False):
        ax.plot(sub["iteration"], sub["rmse"], "-o", lw=1.6, ms=4, label=str(stage))
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("RMSE")
    ax.set_title("Stage RMSE")
    ax.legend(frameon=False)

    ax = axes[1]
    if "max_abs_error" in df.columns:
        for stage, sub in df.groupby("stage", dropna=False):
            ax.plot(sub["iteration"], sub["max_abs_error"], "-o", lw=1.6, ms=4, label=str(stage))
        ax.set_ylabel("Max absolute error")
    ax.set_xlabel("Evaluation")
    ax.set_title("Worst target error")
    fig.tight_layout()
    save_figure(fig, "logp_bo_staged_overview", outdir)

    rmse_data = df[["stage", "iteration", "rmse", "status"]].copy()
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for stage, sub in df.groupby("stage", dropna=False):
        ax.plot(sub["iteration"], sub["rmse"], "-o", lw=1.6, ms=4, label=str(stage))
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("RMSE")
    ax.set_title("Stage RMSE")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_panel(fig, "stage_rmse", outdir, rmse_data)

    if "max_abs_error" in df.columns:
        max_data = df[["stage", "iteration", "max_abs_error", "status"]].copy()
        fig, ax = plt.subplots(figsize=(3.8, 3.0))
        for stage, sub in df.groupby("stage", dropna=False):
            ax.plot(sub["iteration"], sub["max_abs_error"], "-o", lw=1.6, ms=4, label=str(stage))
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Max absolute error")
        ax.set_title("Worst target error")
        ax.legend(frameon=False)
        fig.tight_layout()
        save_panel(fig, "stage_max_abs_error", outdir, max_data)


def plot_staged_targets(df, outdir):
    targets = parse_target_errors(df)
    if targets.empty:
        return
    targets.to_csv(outdir / "staged_target_errors_long.csv", index=False)

    names = list(targets["target_name"].drop_duplicates())
    colors = list(PALETTE.values())
    fig_height = max(3.2, min(8.0, 0.25 * len(names) + 3.0))
    fig, axes = plt.subplots(2, 1, figsize=(8.2, fig_height), sharex=True)

    for idx, name in enumerate(names):
        sub = targets[targets["target_name"] == name].sort_values("evaluation")
        color = colors[idx % len(colors)]
        axes[0].plot(sub["evaluation"], sub["observed"], "-o", lw=1.3, ms=3, label=name, color=color)
        axes[0].plot(sub["evaluation"], sub["target"], "--", lw=0.9, color=color, alpha=0.55)
        axes[1].plot(sub["evaluation"], sub["abs_error"], "-o", lw=1.3, ms=3, label=name, color=color)

    axes[0].set_ylabel("Observed value")
    axes[0].set_title("Per-target observed values")
    axes[1].set_xlabel("Evaluation")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title("Per-target absolute errors")
    for ax in axes:
        ax.legend(frameon=False, ncol=2 if len(names) <= 8 else 3, fontsize=7)
    fig.tight_layout()
    save_figure(fig, "staged_target_error_overview", outdir)

    observed_data = targets[["stage", "evaluation", "status", "target_name", "target", "observed"]].copy()
    fig, ax = plt.subplots(figsize=(4.2, fig_height / 2.0 + 0.8))
    for idx, name in enumerate(names):
        sub = targets[targets["target_name"] == name].sort_values("evaluation")
        color = colors[idx % len(colors)]
        ax.plot(sub["evaluation"], sub["observed"], "-o", lw=1.3, ms=3, label=name, color=color)
        ax.plot(sub["evaluation"], sub["target"], "--", lw=0.9, color=color, alpha=0.55)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Observed logP")
    ax.set_title("Per-target observed values")
    ax.legend(frameon=False, ncol=2 if len(names) <= 8 else 3, fontsize=7)
    fig.tight_layout()
    save_panel(fig, "staged_target_observed_values", outdir, observed_data)

    abs_error_data = targets[["stage", "evaluation", "status", "target_name", "abs_error"]].copy()
    fig, ax = plt.subplots(figsize=(4.2, fig_height / 2.0 + 0.8))
    for idx, name in enumerate(names):
        sub = targets[targets["target_name"] == name].sort_values("evaluation")
        color = colors[idx % len(colors)]
        ax.plot(sub["evaluation"], sub["abs_error"], "-o", lw=1.3, ms=3, label=name, color=color)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Absolute error")
    ax.set_title("Per-target absolute errors")
    ax.legend(frameon=False, ncol=2 if len(names) <= 8 else 3, fontsize=7)
    fig.tight_layout()
    save_panel(fig, "staged_target_abs_errors", outdir, abs_error_data)
    plot_staged_r2(targets, outdir)


def plot_staged_r2(targets, outdir):
    r2_df = calculate_staged_r2(targets)
    if r2_df.empty:
        return
    r2_df.to_csv(outdir / "staged_r2_by_evaluation.csv", index=False)

    ok_mask = r2_df["status"].astype(str).str.lower().eq("ok")
    finite_mask = np.isfinite(pd.to_numeric(r2_df["r2"], errors="coerce"))
    valid = r2_df[ok_mask & finite_mask].copy()
    all_finite = r2_df[finite_mask].copy()

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ax = axes[0]
    colors = list(PALETTE.values())
    for idx, (stage, sub) in enumerate(r2_df.groupby("stage", dropna=False)):
        sub = sub.sort_values("evaluation")
        y = pd.to_numeric(sub["r2"], errors="coerce")
        display_y = y.clip(lower=-2.0, upper=1.05)
        ax.plot(
            sub["evaluation"],
            display_y,
            "-o",
            lw=1.4,
            ms=3.5,
            label=str(stage),
            color=colors[idx % len(colors)],
        )
        penalized = sub["status"].astype(str).str.lower() != "ok"
        if penalized.any():
            ax.scatter(
                sub.loc[penalized, "evaluation"],
                display_y.loc[penalized],
                marker="x",
                s=36,
                color=PALETTE["red"],
                label="failed/penalized" if idx == 0 else None,
                zorder=5,
            )
    if not valid.empty:
        best_rows = []
        for _, sub in valid.groupby("stage", dropna=False):
            sub = sub.sort_values("evaluation").copy()
            sub["best_r2_so_far"] = sub["r2"].cummax()
            best_rows.append(sub)
        best_line = pd.concat(best_rows, ignore_index=True)
        for stage, sub in best_line.groupby("stage", dropna=False):
            ax.plot(
                sub["evaluation"],
                sub["best_r2_so_far"].clip(lower=-2.0, upper=1.05),
                "--",
                lw=1.2,
                color=PALETTE["black"],
                alpha=0.65,
                label="best so far" if stage == best_line["stage"].iloc[0] else None,
            )
    ax.axhline(1.0, color=PALETTE["black"], lw=0.9, ls=":")
    ax.axhline(0.0, color=PALETTE["gray"], lw=0.8, ls=":")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("R²")
    ax.set_ylim(-2.05, 1.08)
    ax.set_title("R² over optimization")
    ax.text(
        0.02,
        0.03,
        "values < -2 clipped",
        transform=ax.transAxes,
        fontsize=7,
        color=PALETTE["gray"],
    )
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    best_source = valid if not valid.empty else all_finite
    if not best_source.empty:
        best_row = best_source.loc[best_source["r2"].idxmax()]
        best_targets = targets[
            (targets["stage"] == best_row["stage"])
            & (targets["evaluation"] == int(best_row["evaluation"]))
        ].copy()
        best_targets = best_targets[np.isfinite(best_targets["target"]) & np.isfinite(best_targets["observed"])]
        ax.scatter(
            best_targets["target"],
            best_targets["observed"],
            s=42,
            color=PALETTE["blue"],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for _, row in best_targets.iterrows():
            ax.annotate(
                str(row["target_name"]),
                (row["target"], row["observed"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.5,
            )
        min_val = float(min(best_targets["target"].min(), best_targets["observed"].min()))
        max_val = float(max(best_targets["target"].max(), best_targets["observed"].max()))
        pad = max(0.1, 0.08 * (max_val - min_val if max_val > min_val else 1.0))
        ax.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], "--", color=PALETTE["black"], lw=1.0)
        ax.set_xlim(min_val - pad, max_val + pad)
        ax.set_ylim(min_val - pad, max_val + pad)
        ax.text(
            0.04,
            0.96,
            f"best eval = {int(best_row['evaluation'])}\nR² = {best_row['r2']:.3f}\nRMSE = {best_row['rmse']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    ax.set_xlabel("Target logP")
    ax.set_ylabel("Simulated logP")
    ax.set_title("Best R² fit")
    fig.tight_layout()
    save_figure(fig, "staged_r2_overview", outdir)

    r2_plot_data = r2_df.copy()
    r2_plot_data["display_r2"] = pd.to_numeric(r2_plot_data["r2"], errors="coerce").clip(lower=-2.0, upper=1.05)
    r2_plot_data["is_failed_or_penalized"] = r2_plot_data["status"].astype(str).str.lower() != "ok"
    if not valid.empty:
        best_rows = []
        for _, sub in valid.groupby("stage", dropna=False):
            sub = sub.sort_values("evaluation").copy()
            sub["best_r2_so_far"] = sub["r2"].cummax()
            best_rows.append(sub[["stage", "evaluation", "best_r2_so_far"]])
        best_line_data = pd.concat(best_rows, ignore_index=True)
        r2_plot_data = r2_plot_data.merge(best_line_data, on=["stage", "evaluation"], how="left")
    else:
        r2_plot_data["best_r2_so_far"] = np.nan

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for idx, (stage, sub) in enumerate(r2_plot_data.groupby("stage", dropna=False)):
        sub = sub.sort_values("evaluation")
        ax.plot(
            sub["evaluation"],
            sub["display_r2"],
            "-o",
            lw=1.4,
            ms=3.5,
            label=str(stage),
            color=colors[idx % len(colors)],
        )
        penalized = sub["is_failed_or_penalized"]
        if penalized.any():
            ax.scatter(
                sub.loc[penalized, "evaluation"],
                sub.loc[penalized, "display_r2"],
                marker="x",
                s=36,
                color=PALETTE["red"],
                label="failed/penalized" if idx == 0 else None,
                zorder=5,
            )
        if sub["best_r2_so_far"].notna().any():
            ax.plot(
                sub["evaluation"],
                sub["best_r2_so_far"].clip(lower=-2.0, upper=1.05),
                "--",
                lw=1.2,
                color=PALETTE["black"],
                alpha=0.65,
                label="best so far" if idx == 0 else None,
            )
    ax.axhline(1.0, color=PALETTE["black"], lw=0.9, ls=":")
    ax.axhline(0.0, color=PALETTE["gray"], lw=0.8, ls=":")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("R²")
    ax.set_ylim(-2.05, 1.08)
    ax.set_title("R² over optimization")
    ax.text(0.02, 0.03, "values < -2 clipped", transform=ax.transAxes, fontsize=7, color=PALETTE["gray"])
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_panel(fig, "staged_r2_over_time", outdir, r2_plot_data)

    best_source = valid if not valid.empty else all_finite
    if not best_source.empty:
        best_row = best_source.loc[best_source["r2"].idxmax()]
        best_targets = targets[
            (targets["stage"] == best_row["stage"])
            & (targets["evaluation"] == int(best_row["evaluation"]))
        ].copy()
        best_targets = best_targets[np.isfinite(best_targets["target"]) & np.isfinite(best_targets["observed"])]
        for column in ["r2", "rmse", "max_abs_error", "sse", "sst", "n_targets"]:
            if column in best_row:
                best_targets[column] = best_row[column]
        fig, ax = plt.subplots(figsize=(3.45, 3.35))
        ax.scatter(
            best_targets["target"],
            best_targets["observed"],
            s=42,
            color=PALETTE["blue"],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for _, row in best_targets.iterrows():
            ax.annotate(
                str(row["target_name"]),
                (row["target"], row["observed"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.5,
            )
        min_val = float(min(best_targets["target"].min(), best_targets["observed"].min()))
        max_val = float(max(best_targets["target"].max(), best_targets["observed"].max()))
        pad = max(0.1, 0.08 * (max_val - min_val if max_val > min_val else 1.0))
        ax.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], "--", color=PALETTE["black"], lw=1.0)
        ax.set_xlim(min_val - pad, max_val + pad)
        ax.set_ylim(min_val - pad, max_val + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.04,
            0.96,
            f"best eval = {int(best_row['evaluation'])}\nR² = {best_row['r2']:.3f}\nRMSE = {best_row['rmse']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.set_xlabel("Target logP")
        ax.set_ylabel("Simulated logP")
        ax.set_title("Best R² fit")
        fig.tight_layout()
        save_panel(fig, "staged_best_r2_fit", outdir, best_targets)


def plot_aij(df, outdir):
    aij = parse_aij_series(df)
    if aij.empty:
        return
    aij.to_csv(outdir / "aij_trajectory_long.csv", index=False)
    pairs = list(aij["pair"].drop_duplicates())
    fig_height = max(3.2, min(10.0, 0.32 * len(pairs) + 2.0))
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    colors = list(PALETTE.values())
    for idx, pair in enumerate(pairs):
        sub = aij[aij["pair"] == pair]
        ax.plot(sub["iteration"], sub["Aij"], "-o", lw=1.3, ms=3, label=pair, color=colors[idx % len(colors)])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Aij")
    ax.set_title("Optimized pair parameters")
    if len(pairs) <= 12:
        ax.legend(frameon=False, ncol=2)
    else:
        ax.legend(frameon=False, ncol=2, fontsize=6)
    fig.tight_layout()
    save_figure(fig, "aij_trajectories", outdir)

    pair_dir = outdir / "aij_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    for idx, pair in enumerate(pairs):
        sub = aij[aij["pair"] == pair].sort_values("iteration").copy()
        fig, ax = plt.subplots(figsize=(3.8, 3.0))
        ax.plot(sub["iteration"], sub["Aij"], "-o", lw=1.4, ms=3.5, color=colors[idx % len(colors)])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Aij")
        ax.set_title(pair)
        fig.tight_layout()
        save_panel(fig, f"aij_{slugify(pair)}", pair_dir, sub)


def write_summary(df, kind, metric, outdir):
    work = df.copy()
    if "iteration" not in work.columns:
        if "evaluation" in work.columns:
            work["iteration"] = work["evaluation"]
        else:
            work["iteration"] = np.arange(1, len(work) + 1)
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    best_idx = work[metric].idxmin()
    best = work.loc[[best_idx]].copy()
    best.to_csv(outdir / "best_iteration.csv", index=False)
    summary = {
        "history_kind": kind,
        "evaluations": int(len(work)),
        "metric": metric,
        "best_iteration": int(best.iloc[0]["iteration"]),
        "best_metric": float(best.iloc[0][metric]) if not pd.isna(best.iloc[0][metric]) else math.nan,
    }
    if "observed_logp" in work.columns:
        summary["best_observed_logp"] = float(pd.to_numeric(best.iloc[0:1]["observed_logp"], errors="coerce").iloc[0])
        if "target_logp" in work.columns:
            summary["target_logp"] = float(pd.to_numeric(best.iloc[0:1]["target_logp"], errors="coerce").iloc[0])
    pd.DataFrame([summary]).to_csv(outdir / "optimization_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Plot OB-DPD Bayesian logP optimization history")
    parser.add_argument("history", help="fit_history.csv, staged_fit_history.csv, or an output directory")
    parser.add_argument("--outdir", help="figure output directory; default is <history_dir>/figures")
    args = parser.parse_args()

    configure_style()
    history_path = resolve_history_path(args.history)
    outdir = Path(args.outdir) if args.outdir else history_path.parent / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    df = read_history_csv(history_path)
    if df.empty:
        raise ValueError(f"history file is empty: {history_path}")
    kind, metric = metric_columns(df)
    if "iteration" not in df.columns:
        if "evaluation" in df.columns:
            df["iteration"] = df["evaluation"]
        else:
            df["iteration"] = np.arange(1, len(df) + 1)

    if kind == "single":
        plot_single_target(df, outdir)
    else:
        plot_staged(df, outdir)
        plot_staged_targets(df, outdir)
    plot_aij(df, outdir)
    write_summary(df, kind, metric, outdir)
    print(f"wrote Bayesian logP plots to {outdir}")


if __name__ == "__main__":
    main()
