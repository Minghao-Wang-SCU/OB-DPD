#!/usr/bin/env python3
"""Analyze SDS morphology from OB-DPD LAMMPS trajectories.

Outputs are written as raw CSV files and publication-ready figures. The script
assumes the current SDS topology used in OB-DPD:
  type 1: SDS hydrophobic tail beads
  type 2: SDS anionic head bead
  type 3: water bead
  type 4: Na+ counterion
Each SDS molecule is represented by 4 beads in packed_polymer_and_solution.data.
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


MW_SDS_NA = 288.38
MW_SDS_ANION = 265.39
MW_WATER = 18.01528
_WORKER_CONTEXT = {}


class UnionFind:
    def __init__(self, items):
        self.parent = {int(item): int(item) for item in items}
        self.rank = {int(item): 0 for item in items}

    def find(self, item):
        item = int(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def progress(iterable, total=None, desc="frames", disable=False):
    if tqdm is not None and not disable:
        return tqdm(iterable, total=total, desc=desc, unit="frame")
    return iterable


def parse_input_count(run_dir):
    path = run_dir / "input_data.txt"
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) >= 3 and lines[0].lower() == "smiles":
        return int(lines[2])
    return None


def parse_lammps_input(run_dir):
    path = run_dir / "lammps.in"
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "timestep":
            out["timestep"] = float(parts[1])
        elif len(parts) >= 5 and parts[0] == "dump":
            out["dump_every"] = int(parts[4])
        elif len(parts) >= 2 and parts[0] == "run":
            out["run_steps"] = int(parts[1])
    return out


def read_system_summary(run_dir):
    path = run_dir / "system_bead_summary.csv"
    values = {}
    if path.exists():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            values[str(row["item"])] = float(row["value"])
    return values


def parse_box_from_data(run_dir):
    path = run_dir / "packed_polymer_and_solution.data"
    bounds = {}
    with path.open(errors="ignore") as handle:
        for line in handle:
            if "xlo xhi" in line:
                fields = line.split()
                bounds["x"] = (float(fields[0]), float(fields[1]))
            elif "ylo yhi" in line:
                fields = line.split()
                bounds["y"] = (float(fields[0]), float(fields[1]))
            elif "zlo zhi" in line:
                fields = line.split()
                bounds["z"] = (float(fields[0]), float(fields[1]))
            if len(bounds) == 3:
                break
    return np.array([bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z")], dtype=float)


def infer_sds_topology(run_dir):
    n_sds = parse_input_count(run_dir)
    if n_sds is None:
        raise ValueError(f"cannot infer SDS count from {run_dir / 'input_data.txt'}")
    molecule_atoms = {}
    atom_to_mol = {}
    tail_atoms = []
    head_atoms = []
    for mol in range(1, n_sds + 1):
        atoms = list(range((mol - 1) * 4 + 1, mol * 4 + 1))
        molecule_atoms[mol] = atoms
        for atom in atoms:
            atom_to_mol[atom] = mol
        tail_atoms.extend(atoms[:3])
        head_atoms.append(atoms[3])
    return n_sds, molecule_atoms, atom_to_mol, set(tail_atoms), set(head_atoms)


def minimum_image(delta, box_lengths):
    return delta - box_lengths * np.round(delta / box_lengths)


def read_dump_frame(handle):
    line = handle.readline()
    while line and not line.startswith("ITEM: TIMESTEP"):
        line = handle.readline()
    if not line:
        return None
    timestep = int(handle.readline().strip())
    if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
        raise ValueError("unexpected dump format: missing NUMBER OF ATOMS")
    n_atoms = int(handle.readline().strip())
    if not handle.readline().startswith("ITEM: BOX BOUNDS"):
        raise ValueError("unexpected dump format: missing BOX BOUNDS")
    bounds = []
    for _ in range(3):
        lo, hi, *_ = handle.readline().split()
        bounds.append((float(lo), float(hi)))
    columns = handle.readline().split()[2:]
    col = {name: idx for idx, name in enumerate(columns)}
    rows = []
    for _ in range(n_atoms):
        rows.append(handle.readline().split())
    return timestep, bounds, col, rows


def iter_tail_frames(dump_path, start_timestep=0, stride=1, max_frames=None, final_timestep=3000000):
    selected = []
    with dump_path.open(errors="ignore") as handle:
        if start_timestep > 0 and final_timestep > 0:
            size = os.path.getsize(dump_path)
            offset = int(max(0, size * min(start_timestep / final_timestep, 0.98) - 0.02 * size))
            handle.seek(offset)
            handle.readline()
        frame_i = 0
        while True:
            frame = read_dump_frame(handle)
            if frame is None:
                break
            timestep = frame[0]
            if timestep < start_timestep:
                continue
            if frame_i % max(1, stride) == 0:
                selected.append(frame)
                if max_frames and len(selected) >= max_frames:
                    break
            frame_i += 1
    return selected


def parse_frame_positions(frame, atom_to_mol, tail_atoms, head_atoms):
    timestep, bounds, col, rows = frame
    box_lengths = np.array([hi - lo for lo, hi in bounds], dtype=float)
    positions = {}
    atom_types = {}
    tail_positions = []
    tail_mols = []
    water_positions = []
    sodium_positions = []
    head_positions = []
    for parts in rows:
        atom_id = int(parts[col["id"]])
        atom_type = int(parts[col["type"]])
        pos = np.array(
            [float(parts[col["x"]]), float(parts[col["y"]]), float(parts[col["z"]])],
            dtype=float,
        )
        atom_types[atom_id] = atom_type
        if atom_id in atom_to_mol:
            positions[atom_id] = pos
            if atom_id in tail_atoms:
                tail_positions.append(pos)
                tail_mols.append(atom_to_mol[atom_id])
            elif atom_id in head_atoms:
                head_positions.append(pos)
        elif atom_type == 3:
            water_positions.append(pos)
        elif atom_type == 4:
            sodium_positions.append(pos)
    return {
        "timestep": timestep,
        "box_lengths": box_lengths,
        "positions": positions,
        "tail_positions": np.asarray(tail_positions, dtype=float),
        "tail_mols": np.asarray(tail_mols, dtype=np.int32),
        "head_positions": np.asarray(head_positions, dtype=float),
        "water_positions": np.asarray(water_positions, dtype=float),
        "sodium_positions": np.asarray(sodium_positions, dtype=float),
    }


def build_clusters_from_tails(tail_positions, tail_mols, molecule_ids, cutoff, box_lengths):
    uf = UnionFind(molecule_ids)
    if len(tail_positions) == 0:
        return {mol: [mol] for mol in molecule_ids}
    cell_size = cutoff
    ncell = np.maximum(np.floor(box_lengths / cell_size).astype(int), 1)
    cell_ids = np.floor(tail_positions / box_lengths * ncell).astype(int) % ncell
    cells = defaultdict(list)
    for idx, cell in enumerate(cell_ids):
        cells[tuple(cell)].append(idx)
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for cell, indices in cells.items():
        cell_vec = np.array(cell)
        for off in offsets:
            neigh = tuple((cell_vec + np.array(off)) % ncell)
            if neigh not in cells or neigh < cell:
                continue
            other = cells[neigh]
            for local_i, i in enumerate(indices):
                start = local_i + 1 if neigh == cell else 0
                for j in other[start:]:
                    if tail_mols[i] == tail_mols[j]:
                        continue
                    delta = minimum_image(tail_positions[i] - tail_positions[j], box_lengths)
                    if float(np.dot(delta, delta)) <= cutoff * cutoff:
                        uf.union(int(tail_mols[i]), int(tail_mols[j]))
    clusters = defaultdict(list)
    for mol in molecule_ids:
        clusters[uf.find(mol)].append(mol)
    return dict(clusters)


def unwrap_points(points, box_lengths):
    if len(points) == 0:
        return points
    ref = points[0]
    return ref + minimum_image(points - ref, box_lengths)


def shape_descriptors(cluster_mols, positions, molecule_atoms, box_lengths):
    atom_ids = [atom for mol in cluster_mols for atom in molecule_atoms[mol] if atom in positions]
    pts = np.asarray([positions[atom] for atom in atom_ids], dtype=float)
    if len(pts) < 2:
        return dict(rg=0.0, kappa2=0.0, asphericity=0.0, eig1=0.0, eig2=0.0, eig3=0.0, eig1_over_eig3=math.nan)
    unwrapped = unwrap_points(pts, box_lengths)
    centered = unwrapped - unwrapped.mean(axis=0)
    gyr = centered.T @ centered / len(centered)
    eig = np.sort(np.linalg.eigvalsh(gyr))[::-1]
    l1, l2, l3 = [float(x) for x in eig]
    rg2 = l1 + l2 + l3
    if rg2 <= 0:
        return dict(rg=0.0, kappa2=0.0, asphericity=0.0, eig1=l1, eig2=l2, eig3=l3, eig1_over_eig3=math.nan)
    kappa2 = 1.0 - 3.0 * (l1 * l2 + l2 * l3 + l3 * l1) / (rg2 * rg2)
    asphericity = l1 - 0.5 * (l2 + l3)
    return {
        "rg": math.sqrt(max(rg2, 0.0)),
        "kappa2": float(kappa2),
        "asphericity": float(asphericity),
        "eig1": l1,
        "eig2": l2,
        "eig3": l3,
        "eig1_over_eig3": float(l1 / l3) if l3 > 0 else math.inf,
    }


def circular_span(points, box_lengths):
    spans = []
    for axis, length in enumerate(box_lengths):
        coords = np.sort((points[:, axis] % length) / length)
        if len(coords) <= 1:
            spans.append(0.0)
            continue
        gaps = np.diff(np.r_[coords, coords[0] + 1.0])
        spans.append(float((1.0 - np.max(gaps)) * length))
    return np.asarray(spans, dtype=float)


def radial_density(center, positions, box_lengths, bins):
    if positions is None or len(positions) == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    deltas = minimum_image(positions - center, box_lengths)
    distances = np.linalg.norm(deltas, axis=1)
    hist, _ = np.histogram(distances, bins=bins)
    volumes = 4.0 / 3.0 * math.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    return hist / volumes


def init_frame_worker(context):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def analyze_frame_worker(frame):
    ctx = _WORKER_CONTEXT
    parsed = parse_frame_positions(frame, ctx["atom_to_mol"], ctx["tail_atoms"], ctx["head_atoms"])
    clusters = build_clusters_from_tails(
        parsed["tail_positions"],
        parsed["tail_mols"],
        ctx["molecule_ids"],
        ctx["cluster_cutoff"],
        parsed["box_lengths"],
    )
    clusters_sorted = sorted(clusters.values(), key=len, reverse=True)
    size_counts = Counter(len(mols) for mols in clusters_sorted)
    micelle_sizes = [len(mols) for mols in clusters_sorted if len(mols) >= ctx["micelle_min_size"]]
    monomers = size_counts.get(1, 0)
    largest = clusters_sorted[0] if clusters_sorted else []
    largest_shape = shape_descriptors(largest, parsed["positions"], ctx["molecule_atoms"], parsed["box_lengths"])
    all_micelle_shapes = [
        shape_descriptors(mols, parsed["positions"], ctx["molecule_atoms"], parsed["box_lengths"])
        for mols in clusters_sorted
        if len(mols) >= ctx["micelle_min_size"]
    ]
    largest_atoms = [atom for mol in largest for atom in ctx["molecule_atoms"][mol] if atom in parsed["positions"]]
    largest_pts = np.asarray([parsed["positions"][atom] for atom in largest_atoms], dtype=float)
    spans = circular_span(largest_pts, parsed["box_lengths"]) if len(largest_pts) else np.zeros(3)
    percolated = spans > (ctx["percolation_fraction"] * parsed["box_lengths"])
    percolation_dimension = int(np.count_nonzero(percolated))
    time = int(parsed["timestep"]) * ctx["timestep"]

    cluster_rows = []
    for cid, mols in enumerate(clusters_sorted, start=1):
        shape = shape_descriptors(mols, parsed["positions"], ctx["molecule_atoms"], parsed["box_lengths"])
        cluster_rows.append(
            {
                "system": ctx["system"],
                "timestep": parsed["timestep"],
                "time": time,
                "cluster_id": cid,
                "cluster_size": len(mols),
                **shape,
            }
        )

    mic_kappa = np.mean([s["kappa2"] for s in all_micelle_shapes]) if all_micelle_shapes else math.nan
    mic_rg = np.mean([s["rg"] for s in all_micelle_shapes]) if all_micelle_shapes else math.nan
    frame_row = {
        "system": ctx["system"],
        "timestep": parsed["timestep"],
        "time": time,
        "n_sds": ctx["n_sds"],
        "n_water": ctx["n_water"],
        "wt_percent_real_sds_na": ctx["wt_real"],
        "wt_percent_equal_bead_mass": ctx["wt_bead"],
        "n_clusters": len(clusters_sorted),
        "monomer_fraction": monomers / ctx["n_sds"],
        "largest_cluster_size": len(largest),
        "largest_cluster_fraction": len(largest) / ctx["n_sds"],
        "nagg_number_average": float(np.mean(micelle_sizes)) if micelle_sizes else 0.0,
        "nagg_weight_average": float(np.sum(np.square(micelle_sizes)) / np.sum(micelle_sizes)) if micelle_sizes else 0.0,
        "micelle_count": len(micelle_sizes),
        "rg_largest": largest_shape["rg"],
        "kappa2_largest": largest_shape["kappa2"],
        "eig1_over_eig3_largest": largest_shape["eig1_over_eig3"],
        "rg_micelle_mean": mic_rg,
        "kappa2_micelle_mean": mic_kappa,
        "span_x": spans[0],
        "span_y": spans[1],
        "span_z": spans[2],
        "percolation_x": int(percolated[0]),
        "percolation_y": int(percolated[1]),
        "percolation_z": int(percolated[2]),
        "percolation_dimension": percolation_dimension,
    }

    density = None
    if len(largest_pts) and ctx["radial_density"]:
        bins = ctx["radial_bins"]
        center = unwrap_points(largest_pts, parsed["box_lengths"]).mean(axis=0) % parsed["box_lengths"]
        tail_positions = np.asarray(
            [parsed["positions"][atom] for mol in largest for atom in ctx["molecule_atoms"][mol][:3] if atom in parsed["positions"]],
            dtype=float,
        )
        head_positions = np.asarray(
            [parsed["positions"][ctx["molecule_atoms"][mol][3]] for mol in largest if ctx["molecule_atoms"][mol][3] in parsed["positions"]],
            dtype=float,
        )
        density = {
            "tail": radial_density(center, tail_positions, parsed["box_lengths"], bins),
            "head": radial_density(center, head_positions, parsed["box_lengths"], bins),
            "water": radial_density(center, parsed["water_positions"], parsed["box_lengths"], bins),
            "na": radial_density(center, parsed["sodium_positions"], parsed["box_lengths"], bins),
        }
    return cluster_rows, frame_row, density


def read_last_frames(run_dir, args, metadata):
    dump_path = run_dir / args.dump_file
    if args.start_timestep is None:
        run_steps = int(metadata.get("run_steps", args.final_timestep))
        dump_every = int(metadata.get("dump_every", 5000))
        start = max(0, run_steps - dump_every * args.max_frames * args.frame_stride)
    else:
        start = args.start_timestep
    frames = iter_tail_frames(
        dump_path,
        start_timestep=start,
        stride=args.frame_stride,
        max_frames=args.max_frames,
        final_timestep=int(metadata.get("run_steps", args.final_timestep)),
    )
    if not frames:
        raise ValueError(f"no frames read from {dump_path}")
    return frames


def classify_morphology(row):
    if row["percolation_dimension_mean"] >= 2.0:
        return "percolated/concentrated phase"
    if row["largest_cluster_fraction_mean"] > 0.8:
        if row["kappa2_largest_mean"] > 0.45:
            return "wormlike or cylindrical aggregate"
        if row["kappa2_largest_mean"] < 0.20:
            return "single spherical aggregate"
        return "large connected aggregate"
    if row["nagg_weight_average_mean"] < 10:
        return "monomers/small aggregates"
    micelle_kappa = row.get("kappa2_micelle_mean")
    if micelle_kappa is None:
        micelle_kappa = row.get("kappa2_micelle_mean_mean", math.nan)
    if micelle_kappa < 0.20:
        return "spherical micelles"
    if micelle_kappa < 0.45:
        return "ellipsoidal micelles"
    return "rod-like/wormlike micelles"


def analyze_run(run_dir, args, outroot):
    run_dir = Path(run_dir).resolve()
    system = run_dir.name
    outdir = outroot / system
    rawdir = outdir / "raw"
    vizdir = outdir / "visualization_data"
    figdir = outdir / "figures"
    rawdir.mkdir(parents=True, exist_ok=True)
    vizdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    n_sds, molecule_atoms, atom_to_mol, tail_atoms, head_atoms = infer_sds_topology(run_dir)
    molecule_ids = list(range(1, n_sds + 1))
    metadata = parse_lammps_input(run_dir)
    timestep = float(metadata.get("timestep", 0.02))
    summary = read_system_summary(run_dir)
    n_water = int(summary.get("water_bead_count", 0))
    box_volume = float(summary.get("box_volume", np.prod(parse_box_from_data(run_dir))))
    wt_real = 100.0 * n_sds * MW_SDS_NA / (n_sds * MW_SDS_NA + n_water * MW_WATER) if n_water else math.nan
    wt_bead = 100.0 * (5 * n_sds) / (5 * n_sds + n_water) if n_water else math.nan

    frames = read_last_frames(run_dir, args, metadata)
    cluster_rows = []
    frame_rows = []
    density_acc = None
    density_frames = 0
    bins = np.linspace(0.0, args.radial_rmax, args.radial_bins + 1)

    context = {
        "system": system,
        "n_sds": n_sds,
        "n_water": n_water,
        "wt_real": wt_real,
        "wt_bead": wt_bead,
        "molecule_atoms": molecule_atoms,
        "atom_to_mol": atom_to_mol,
        "tail_atoms": tail_atoms,
        "head_atoms": head_atoms,
        "molecule_ids": molecule_ids,
        "cluster_cutoff": args.cluster_cutoff,
        "micelle_min_size": args.micelle_min_size,
        "percolation_fraction": args.percolation_fraction,
        "timestep": timestep,
        "radial_density": args.radial_density,
        "radial_bins": bins,
    }
    if args.jobs > 1 and len(frames) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=init_frame_worker, initargs=(context,)) as pool:
            results = progress(pool.map(analyze_frame_worker, frames), total=len(frames), desc=system, disable=args.no_progress)
            for frame_cluster_rows, frame_row, density in results:
                cluster_rows.extend(frame_cluster_rows)
                frame_rows.append(frame_row)
                if density is not None:
                    if density_acc is None:
                        density_acc = {key: np.zeros(args.radial_bins, dtype=float) for key in density}
                    for key in density:
                        density_acc[key] += density[key]
                    density_frames += 1
    else:
        init_frame_worker(context)
        for frame in progress(frames, total=len(frames), desc=system, disable=args.no_progress):
            frame_cluster_rows, frame_row, density = analyze_frame_worker(frame)
            cluster_rows.extend(frame_cluster_rows)
            frame_rows.append(frame_row)
            if density is not None:
                if density_acc is None:
                    density_acc = {key: np.zeros(args.radial_bins, dtype=float) for key in density}
                for key in density:
                    density_acc[key] += density[key]
                density_frames += 1

    cluster_df = pd.DataFrame(cluster_rows)
    frame_df = pd.DataFrame(frame_rows)
    cluster_df.to_csv(rawdir / "cluster_observations.csv", index=False)
    frame_df.to_csv(rawdir / "frame_morphology_timeseries.csv", index=False)
    dist = cluster_df.groupby("cluster_size").size().reset_index(name="count")
    dist["probability"] = dist["count"] / dist["count"].sum()
    dist.to_csv(rawdir / "cluster_size_distribution.csv", index=False)
    dist.to_csv(vizdir / "cluster_size_distribution_visualization_data.csv", index=False)
    frame_df.to_csv(vizdir / "fraction_shape_timeseries_visualization_data.csv", index=False)

    if density_acc is not None and density_frames > 0:
        dens_df = pd.DataFrame(
            {
                "r_lower": bins[:-1],
                "r_upper": bins[1:],
                "r_center": 0.5 * (bins[:-1] + bins[1:]),
                "tail_density": density_acc["tail"] / density_frames,
                "head_density": density_acc["head"] / density_frames,
                "water_density": density_acc["water"] / density_frames,
                "na_density": density_acc["na"] / density_frames,
            }
        )
        dens_df.to_csv(rawdir / "radial_density_largest_cluster.csv", index=False)
        dens_df.to_csv(vizdir / "radial_density_largest_cluster_visualization_data.csv", index=False)
    else:
        dens_df = pd.DataFrame()

    summary_row = summarize_system(system, frame_df, n_sds, n_water, wt_real, wt_bead, args)
    pd.DataFrame([summary_row]).to_csv(outdir / "sds_morphology_summary.csv", index=False)
    plot_system_outputs(system, cluster_df, frame_df, dens_df, figdir)
    return summary_row


def mean_std(frame_df, column):
    return float(frame_df[column].mean()), float(frame_df[column].std(ddof=1)) if len(frame_df) > 1 else 0.0


def summarize_system(system, frame_df, n_sds, n_water, wt_real, wt_bead, args):
    row = {
        "system": system,
        "n_sds": n_sds,
        "n_water": n_water,
        "wt_percent_real_sds_na": wt_real,
        "wt_percent_equal_bead_mass": wt_bead,
        "frames_analyzed": len(frame_df),
        "cluster_cutoff": args.cluster_cutoff,
        "micelle_min_size": args.micelle_min_size,
    }
    for column in [
        "n_clusters",
        "monomer_fraction",
        "largest_cluster_size",
        "largest_cluster_fraction",
        "nagg_number_average",
        "nagg_weight_average",
        "micelle_count",
        "rg_largest",
        "kappa2_largest",
        "eig1_over_eig3_largest",
        "rg_micelle_mean",
        "kappa2_micelle_mean",
        "percolation_dimension",
    ]:
        mean, std = mean_std(frame_df, column)
        row[f"{column}_mean"] = mean
        row[f"{column}_std"] = std
    row["assigned_morphology"] = classify_morphology(row)
    return row


def plot_system_outputs(system, cluster_df, frame_df, dens_df, figdir):
    figdir.mkdir(parents=True, exist_ok=True)
    dist = cluster_df.groupby("cluster_size").size().reset_index(name="count")
    dist["probability"] = dist["count"] / dist["count"].sum()

    plt.figure(figsize=(4.0, 3.0))
    plt.plot(dist["cluster_size"], dist["probability"], marker="o", lw=1.2)
    plt.yscale("log")
    plt.xlabel("Aggregation number")
    plt.ylabel("Probability")
    plt.title(system)
    plt.tight_layout()
    savefig(figdir / "cluster_size_distribution")

    plt.figure(figsize=(4.4, 3.0))
    plt.plot(frame_df["time"], frame_df["largest_cluster_fraction"], label="Largest fraction")
    plt.plot(frame_df["time"], frame_df["monomer_fraction"], label="Monomer fraction")
    plt.xlabel("DPD time")
    plt.ylabel("Fraction")
    plt.title(system)
    plt.legend(frameon=False)
    plt.tight_layout()
    savefig(figdir / "fraction_timeseries")

    plt.figure(figsize=(4.4, 3.0))
    plt.plot(frame_df["time"], frame_df["nagg_weight_average"], label="Weight-avg Nagg")
    plt.plot(frame_df["time"], frame_df["kappa2_largest"], label="Largest kappa2")
    plt.xlabel("DPD time")
    plt.ylabel("Value")
    plt.title(system)
    plt.legend(frameon=False)
    plt.tight_layout()
    savefig(figdir / "shape_timeseries")

    if not dens_df.empty:
        plt.figure(figsize=(4.4, 3.2))
        plt.plot(dens_df["r_center"], dens_df["tail_density"], label="Tail", color="#0072B2")
        plt.plot(dens_df["r_center"], dens_df["head_density"], label="Head", color="#D55E00")
        plt.plot(dens_df["r_center"], dens_df["water_density"], label="Water", color="#56B4E9")
        plt.plot(dens_df["r_center"], dens_df["na_density"], label="Na+", color="#009E73")
        plt.xlabel("r from largest aggregate COM")
        plt.ylabel("Number density")
        plt.title(system)
        plt.legend(frameon=False)
        plt.tight_layout()
        savefig(figdir / "radial_density_largest_cluster")


def savefig(base):
    plt.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(base.with_suffix(".svg"), dpi=600, bbox_inches="tight")
    plt.close()


def plot_all_summary(summary_df, outroot):
    figdir = outroot / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plotdata = summary_df.sort_values("wt_percent_real_sds_na")
    plotdata.to_csv(outroot / "visualization_data_summary_vs_concentration.csv", index=False)

    def lineplot(y, ylabel, name):
        plt.figure(figsize=(4.5, 3.2))
        plt.plot(plotdata["wt_percent_real_sds_na"], plotdata[y], marker="o", lw=1.5)
        plt.xlabel("SDS concentration (wt%, SDS-Na)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        savefig(figdir / name)

    lineplot("nagg_weight_average_mean", "Weight-average aggregation number", "nagg_vs_concentration")
    lineplot("largest_cluster_fraction_mean", "Largest-cluster fraction", "largest_fraction_vs_concentration")
    lineplot("monomer_fraction_mean", "Monomer fraction", "monomer_fraction_vs_concentration")
    lineplot("kappa2_largest_mean", "Largest-cluster shape anisotropy", "kappa2_vs_concentration")
    lineplot("percolation_dimension_mean", "Percolation dimension", "percolation_vs_concentration")

    overview_cols = [
        "system",
        "wt_percent_real_sds_na",
        "nagg_weight_average_mean",
        "nagg_weight_average_std",
        "largest_cluster_fraction_mean",
        "largest_cluster_fraction_std",
        "kappa2_largest_mean",
        "kappa2_largest_std",
        "percolation_dimension_mean",
        "percolation_dimension_std",
        "assigned_morphology",
    ]
    plotdata[overview_cols].to_csv(outroot / "visualization_data_sds_morphology_overview.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    x = plotdata["wt_percent_real_sds_na"]
    panels = [
        ("nagg_weight_average_mean", "nagg_weight_average_std", "Weight-average Nagg", axes[0, 0]),
        ("largest_cluster_fraction_mean", "largest_cluster_fraction_std", "Largest-cluster fraction", axes[0, 1]),
        ("kappa2_largest_mean", "kappa2_largest_std", "Largest-cluster kappa2", axes[1, 0]),
        ("percolation_dimension_mean", "percolation_dimension_std", "Percolation dimension", axes[1, 1]),
    ]
    for y, yerr, ylabel, ax in panels:
        ax.errorbar(x, plotdata[y], yerr=plotdata[yerr], marker="o", lw=1.3, capsize=2.5, color="#0072B2")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("SDS concentration (wt%, SDS-Na)")
    fig.tight_layout()
    savefig(figdir / "sds_morphology_overview")

    fig, ax1 = plt.subplots(figsize=(5.2, 3.5))
    ax1.plot(plotdata["wt_percent_real_sds_na"], plotdata["nagg_weight_average_mean"], marker="o", color="#0072B2", label="Nagg")
    ax1.set_xlabel("SDS concentration (wt%, SDS-Na)")
    ax1.set_ylabel("Weight-average Nagg", color="#0072B2")
    ax2 = ax1.twinx()
    ax2.plot(plotdata["wt_percent_real_sds_na"], plotdata["kappa2_largest_mean"], marker="s", color="#D55E00", label="kappa2")
    ax2.set_ylabel("Largest-cluster kappa2", color="#D55E00")
    fig.tight_layout()
    savefig(figdir / "nagg_kappa2_dual_axis")


def default_completed_sds_runs(root):
    candidates = [
        "SDS_gpu_full_100",
        "SDS_gpu_full_300",
        "SDS_gpu_full_500",
        "SDS_gpu_full_700",
        "SDS_gpu_full_900",
        "SDS_wt30_box30_full_mpi8",
        "SDS_wt40_box30_full_mpi8",
        "SDS_wt50_box30_full_mpi8",
        "SDS_wt60_box30_full_mpi8",
        "SDS_wt75_box30_full_mpi8",
        "SDS_wt80_box30_full_mpi8",
    ]
    return [root / name for name in candidates if (root / name / "dump.lammpstrj").exists()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="*", help="SDS run directories")
    parser.add_argument("--outdir", default="analysis/sds_morphology")
    parser.add_argument("--validation-root", default="validation_runs")
    parser.add_argument("--dump-file", default="dump.lammpstrj")
    parser.add_argument("--cluster-cutoff", type=float, default=1.2)
    parser.add_argument("--micelle-min-size", type=int, default=10)
    parser.add_argument("--percolation-fraction", type=float, default=0.75)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-timestep", type=int)
    parser.add_argument("--final-timestep", type=int, default=3000000)
    parser.add_argument("--radial-density", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--radial-rmax", type=float, default=10.0)
    parser.add_argument("--radial-bins", type=int, default=80)
    parser.add_argument("--jobs", type=int, default=1, help="parallel worker processes per SDS system")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    root = Path(args.validation_root)
    run_dirs = [Path(item) for item in args.run_dirs] if args.run_dirs else default_completed_sds_runs(root)
    outroot = Path(args.outdir).resolve()
    outroot.mkdir(parents=True, exist_ok=True)
    summaries = []
    for run_dir in run_dirs:
        print(f"Analyzing {run_dir}", file=sys.stderr)
        summaries.append(analyze_run(run_dir, args, outroot))
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(outroot / "sds_morphology_summary_all.csv", index=False)
    plot_all_summary(summary_df, outroot)
    print(outroot / "sds_morphology_summary_all.csv")


if __name__ == "__main__":
    main()
