#!/usr/bin/env python3
"""Plot water/organic interface detection and bulk sampling regions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 9,
            "axes.labelsize": 9,
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


def contiguous_regions(indices):
    indices = sorted(int(item) for item in indices)
    if not indices:
        return []
    regions = []
    start = previous = indices[0]
    for item in indices[1:]:
        if item == previous + 1:
            previous = item
            continue
        regions.append((start, previous))
        start = previous = item
    regions.append((start, previous))
    return regions


def shade_regions(ax, regions, n_slabs, color="#F0E442", alpha=0.28, label=None):
    used_label = False
    for start, end in regions:
        x0 = start / n_slabs
        x1 = (end + 1) / n_slabs
        ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0, label=label if not used_label else None)
        used_label = True


def load_inputs(run_dir):
    run_dir = Path(run_dir)
    profile_path = run_dir / "partition_slab_profiles.csv"
    summary_path = run_dir / "partition_summary.json"
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    profile = pd.read_csv(profile_path)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return profile, summary


def write_plot_data(profile, summary, outpath):
    data = profile.copy()
    data["interface_gradient_fraction"] = float(summary.get("interface_gradient_fraction", 0.3))
    data["threshold"] = (
        float(summary.get("interface_gradient_fraction", 0.3))
        * (float(data["gradient"].max()) - float(data["gradient"].min()))
    )
    data["C_water"] = float(summary.get("C_water", "nan"))
    data["C_organic"] = float(summary.get("C_organic", summary.get("C_octanol", "nan")))
    data["logP"] = float(summary.get("log10_partition_ratio", "nan"))
    data.to_csv(outpath, index=False)


def plot_interface(run_dir, outbase, title=None):
    profile, summary = load_inputs(run_dir)
    configure_style()

    n_slabs = int(summary.get("slabs", len(profile)))
    water_regions = contiguous_regions(summary.get("water_bulk_slabs", []))
    organic_regions = contiguous_regions(summary.get("organic_bulk_slabs", []))
    interface_regions = contiguous_regions([int(v) for v in profile.loc[profile["is_interface"] == 1, "slab"]])
    threshold = float(summary.get("interface_gradient_fraction", 0.3)) * (
        float(profile["gradient"].max()) - float(profile["gradient"].min())
    )

    outbase = Path(outbase)
    outbase.parent.mkdir(parents=True, exist_ok=True)
    write_plot_data(profile, summary, outbase.with_name(outbase.name + "_data.csv"))

    fig, (ax_profile, ax_gradient) = plt.subplots(1, 2, figsize=(7.3, 3.0), sharex=True)
    ax_solute = ax_profile.twinx()

    for ax in (ax_profile, ax_gradient):
        shade_regions(ax, interface_regions, n_slabs, color="#BDBDBD", alpha=0.18, label="Interface")
        shade_regions(ax, water_regions, n_slabs, color="#F0E442", alpha=0.32, label="Bulk sampling region")
        shade_regions(ax, organic_regions, n_slabs, color="#F0E442", alpha=0.32)

    ax_profile.plot(
        profile["z_center"],
        profile["water_concentration"],
        color="#0072B2",
        lw=1.8,
        label="Water",
    )
    ax_profile.plot(
        profile["z_center"],
        profile["organic_concentration"],
        color="#009E73",
        lw=1.8,
        label="Organic solvent",
    )
    ax_solute.plot(
        profile["z_center"],
        profile["solute_concentration"],
        color="#D55E00",
        lw=1.5,
        ls="--",
        label="Solute",
    )
    ax_profile.set_xlabel("Position along z / Lz")
    ax_profile.set_ylabel("Solvent concentration")
    ax_solute.set_ylabel("Solute concentration")
    ax_profile.set_title("Concentration profile")
    lines_1, labels_1 = ax_profile.get_legend_handles_labels()
    lines_2, labels_2 = ax_solute.get_legend_handles_labels()
    ax_profile.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="upper right")

    ax_gradient.plot(profile["z_center"], profile["gradient"], color="#000000", lw=1.6, label="Gradient")
    ax_gradient.axhline(threshold, color="#D55E00", lw=1.0, ls="--", label="+threshold")
    ax_gradient.axhline(-threshold, color="#D55E00", lw=1.0, ls=":", label="-threshold")
    ax_gradient.axhline(0.0, color="#666666", lw=0.8)
    ax_gradient.set_xlabel("Position along z / Lz")
    ax_gradient.set_ylabel("Gradient")
    ax_gradient.set_title("Gradient profile")
    ax_gradient.legend(frameon=False, loc="upper right")

    for ax in (ax_profile, ax_gradient):
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="y", color="#DDDDDD", lw=0.5, alpha=0.8)
    ax_solute.spines["top"].set_visible(False)

    if title:
        fig.suptitle(title, y=1.03, fontsize=10)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"))
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="directory containing partition_slab_profiles.csv")
    parser.add_argument("--outbase", help="output path without extension")
    parser.add_argument("--title", help="optional figure title")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    outbase = (
        Path(args.outbase)
        if args.outbase
        else run_dir / "interface_bulk_region_evaluation"
    )
    plot_interface(run_dir, outbase, args.title)
    print(f"wrote {Path(outbase).with_suffix('.png')}")
    print(f"wrote {Path(outbase).with_suffix('.pdf')}")
    print(f"wrote {Path(outbase).with_name(Path(outbase).name + '_data.csv')}")


if __name__ == "__main__":
    main()
