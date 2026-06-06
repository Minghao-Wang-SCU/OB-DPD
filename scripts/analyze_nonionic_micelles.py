#!/usr/bin/env python3
"""Analyze CnEm nonionic surfactant micellization from LAMMPS DPD output.

The analysis follows the literature workflow used for alkyl ethoxylates:
cluster-size distributions, CMC from the monomer/sub-micelle peak,
weight-averaged aggregation numbers from the micelle peak, convergence
traces, and micelle shape descriptors.
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except Exception:  # pragma: no cover - optional at import time
    Chem = None
    Descriptors = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


TAIL_TYPES_DEFAULT = {"CH3", "CH2", "CH2CH2", "CH3CH2"}
WATER_MW = 18.01528


class SimpleProgress:
    def __init__(self, total=None, desc="progress", disable=False):
        self.total = total
        self.desc = desc
        self.disable = disable
        self.count = 0

    def __enter__(self):
        if not self.disable:
            self._write()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.disable:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def update(self, step=1):
        self.count += step
        self._write()

    def _write(self):
        if self.disable:
            return
        if self.total:
            pct = 100.0 * self.count / self.total
            text = f"\r{self.desc}: {self.count}/{self.total} frames ({pct:5.1f}%)"
        else:
            text = f"\r{self.desc}: {self.count} frames"
        sys.stderr.write(text)
        sys.stderr.flush()


def progress_bar(total=None, desc="progress", disable=False):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit="frame", disable=disable)
    return SimpleProgress(total=total, desc=desc, disable=disable)


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def parse_lammps_input(path):
    result = {}
    variables = {}
    if not path.exists():
        return result
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "timestep":
            result["timestep"] = float(parts[1])
        elif len(parts) >= 4 and parts[0] == "variable" and parts[2] == "equal":
            try:
                variables[parts[1]] = float(parts[3])
            except ValueError:
                pass
        elif len(parts) >= 5 and parts[0] == "dump":
            result["dump_every"] = int(parts[4])
        elif len(parts) >= 2 and parts[0] == "run":
            result["run_steps"] = int(parts[1])
    if variables:
        result["variables"] = variables
    return result


def parse_pair_cutoffs(path, fallback_cutoff):
    """Read pair-specific DPD cutoffs from LAMMPS pair_coeff lines."""
    cutoffs = {}
    if not path.exists():
        return cutoffs
    variables = parse_lammps_input(path).get("variables", {})
    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.split("#", 1)[0].strip()
        parts = stripped.split()
        if len(parts) < 6 or parts[0] != "pair_coeff":
            continue
        try:
            i = int(parts[1])
            j = int(parts[2])
        except ValueError:
            continue
        token = parts[-1]
        if token.startswith("${") and token.endswith("}"):
            cutoff = variables.get(token[2:-1], fallback_cutoff)
        else:
            try:
                cutoff = float(token)
            except ValueError:
                cutoff = fallback_cutoff
        cutoffs[(i, j)] = cutoff
        cutoffs[(j, i)] = cutoff
    return cutoffs


def parse_input_smiles(path):
    if not path.exists():
        return ""
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) >= 2 and lines[0].lower() == "smiles":
        return lines[1]
    return ""


def parse_input_count(path):
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) >= 3 and lines[0].lower() == "smiles":
        return int(lines[2])
    return None


def mol_weight_from_smiles(smiles):
    if not smiles or Chem is None:
        return math.nan
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return math.nan
    return float(Descriptors.MolWt(mol))


def read_system_summary(path):
    values = {}
    if not path.exists():
        return values
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        values[str(row["item"])] = float(row["value"])
    return values


def read_type_names(path):
    df = pd.read_csv(path)
    return {idx + 1: str(row["assigned_type"]) for idx, row in df.iterrows()}


def read_data_atoms(path, type_names, contact_types, molecule_count=None):
    atom_mol = {}
    atom_type = {}
    molecule_atoms = defaultdict(list)
    contact_atoms = []
    box = None
    in_atoms = False
    with path.open(errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.endswith("xlo xhi"):
                fields = stripped.split()
                xlo, xhi = float(fields[0]), float(fields[1])
                box = [xlo, xhi]
            elif stripped == "Atoms":
                in_atoms = True
                next(handle, None)
                continue
            elif in_atoms and not stripped:
                continue
            elif in_atoms and re.match(r"^[A-Za-z]", stripped):
                break
            elif in_atoms:
                parts = stripped.split()
                if len(parts) < 6:
                    continue
                atom_id = int(parts[0])
                mol_id = int(parts[1])
                type_id = int(parts[2])
                atom_mol[atom_id] = mol_id
                atom_type[atom_id] = type_id
                molecule_atoms[mol_id].append(atom_id)
                if type_names.get(type_id) in contact_types:
                    contact_atoms.append(atom_id)
    if molecule_count and molecule_count > 1 and len(set(atom_mol.values())) <= 1:
        atom_ids = sorted(atom_mol)
        if len(atom_ids) % molecule_count != 0:
            raise ValueError(
                f"cannot infer molecule ids: {len(atom_ids)} solute atoms is not divisible by {molecule_count}"
            )
        beads_per_molecule = len(atom_ids) // molecule_count
        molecule_atoms = defaultdict(list)
        for index, atom_id in enumerate(atom_ids):
            mol_id = index // beads_per_molecule + 1
            atom_mol[atom_id] = mol_id
            molecule_atoms[mol_id].append(atom_id)

    if box is None:
        raise ValueError(f"could not read box bounds from {path}")
    return atom_mol, atom_type, dict(molecule_atoms), contact_atoms


def first_timestep_at_or_after(handle, start_timestep):
    while True:
        pos = handle.tell()
        line = handle.readline()
        if not line:
            return None
        if line.startswith("ITEM: TIMESTEP"):
            timestep_line = handle.readline()
            if not timestep_line:
                return None
            timestep = int(timestep_line.strip())
            if timestep >= start_timestep:
                handle.seek(pos)
                return timestep


def seek_near_timestep(handle, file_size, start_timestep, final_timestep):
    if start_timestep <= 0 or final_timestep <= 0:
        handle.seek(0)
        return
    fraction = min(max(start_timestep / final_timestep, 0.0), 0.98)
    # Move back by 1% of the file to reduce the chance of starting too late.
    offset = int(max(0, file_size * fraction - file_size * 0.01))
    handle.seek(offset)
    handle.readline()
    found = first_timestep_at_or_after(handle, start_timestep)
    if found is None:
        handle.seek(0)


def read_frame(handle):
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
    atom_header = handle.readline().split()[2:]
    col = {name: idx for idx, name in enumerate(atom_header)}
    required = {"id", "x", "y", "z"}
    if not required.issubset(col):
        raise ValueError(f"dump atom columns must contain {required}, got {atom_header}")
    return timestep, n_atoms, bounds, col


def minimum_image(delta, box_lengths):
    return delta - box_lengths * np.round(delta / box_lengths)


def build_clusters(
    contact_positions,
    contact_mols,
    contact_type_ids,
    molecule_ids,
    cutoff,
    box_lengths,
    pair_cutoffs=None,
):
    uf = UnionFind(molecule_ids)
    if len(contact_positions) == 0:
        return {mol: [mol] for mol in molecule_ids}
    max_cutoff = max(pair_cutoffs.values()) if pair_cutoffs else cutoff
    cell_size = max_cutoff
    ncell = np.maximum(np.floor(box_lengths / cell_size).astype(int), 1)
    cells = defaultdict(list)
    scaled = np.floor(contact_positions / box_lengths * ncell).astype(int) % ncell
    for idx, cell in enumerate(scaled):
        cells[tuple(cell)].append(idx)
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for cell, indices in cells.items():
        cell_arr = np.array(cell)
        for off in offsets:
            neigh = tuple((cell_arr + np.array(off)) % ncell)
            if neigh not in cells or neigh < cell:
                continue
            other = cells[neigh]
            for local_i, idx_i in enumerate(indices):
                start = local_i + 1 if neigh == cell else 0
                for idx_j in other[start:]:
                    if contact_mols[idx_i] == contact_mols[idx_j]:
                        continue
                    pair_cutoff = cutoff
                    if pair_cutoffs:
                        pair = (int(contact_type_ids[idx_i]), int(contact_type_ids[idx_j]))
                        pair_cutoff = pair_cutoffs.get(pair, cutoff)
                    delta = minimum_image(contact_positions[idx_i] - contact_positions[idx_j], box_lengths)
                    if float(np.dot(delta, delta)) <= pair_cutoff * pair_cutoff:
                        uf.union(contact_mols[idx_i], contact_mols[idx_j])
    clusters = defaultdict(list)
    for mol in molecule_ids:
        clusters[uf.find(mol)].append(mol)
    return dict(clusters)


def shape_for_cluster(cluster_mols, positions, molecule_atoms, box_lengths):
    atom_ids = [atom for mol in cluster_mols for atom in molecule_atoms[mol]]
    pts = np.array([positions[atom] for atom in atom_ids if atom in positions], dtype=float)
    if len(pts) < 2:
        return 0.0, 0.0
    ref = pts[0]
    unwrapped = ref + minimum_image(pts - ref, box_lengths)
    centered = unwrapped - unwrapped.mean(axis=0)
    gyr = centered.T @ centered / len(centered)
    eig = np.sort(np.linalg.eigvalsh(gyr))
    rg2 = float(eig.sum())
    rg = math.sqrt(max(rg2, 0.0))
    if rg2 <= 0:
        return rg, 0.0
    l1, l2, l3 = eig
    asphericity = ((l3 - l2) ** 2 + (l3 - l1) ** 2 + (l2 - l1) ** 2) / (2.0 * rg2 * rg2)
    return rg, float(asphericity)


def analyze_frame_task(task):
    (
        timestep,
        time,
        box_lengths,
        atom_ids,
        atom_positions,
        contact_mask,
        atom_mol_values,
        atom_type_values,
        molecule_ids,
        molecule_atoms,
        cutoff,
        pair_cutoffs,
    ) = task
    positions = {
        int(atom_id): atom_positions[idx]
        for idx, atom_id in enumerate(atom_ids)
    }
    contact_positions = atom_positions[contact_mask]
    contact_mols = atom_mol_values[contact_mask]
    contact_type_ids = atom_type_values[contact_mask]
    clusters = build_clusters(
        contact_positions,
        contact_mols,
        contact_type_ids,
        molecule_ids,
        cutoff,
        box_lengths,
        pair_cutoffs=pair_cutoffs,
    )
    rows = []
    size_counts = Counter()
    for cid, mols in enumerate(clusters.values(), start=1):
        size = len(mols)
        rg, asph = shape_for_cluster(mols, positions, molecule_atoms, box_lengths)
        size_counts[size] += 1
        rows.append(
            {
                "timestep": timestep,
                "time": time,
                "cluster_id": cid,
                "cluster_size": size,
                "rg": rg,
                "asphericity": asph,
            }
        )
    return rows, size_counts


def choose_cluster_threshold(size_counts, minimum=2):
    sizes = sorted(size_counts)
    if not sizes:
        return minimum
    max_size = max(sizes)
    hist = np.zeros(max_size + 1, dtype=float)
    for size, count in size_counts.items():
        hist[size] = count
    # Smooth lightly to avoid selecting a one-frame hole.
    smooth = hist.copy()
    for i in range(1, max_size):
        smooth[i] = 0.25 * hist[i - 1] + 0.5 * hist[i] + 0.25 * hist[i + 1]
    peaks = [i for i in range(1, max_size) if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]]
    peaks = [p for p in peaks if p >= 1 and smooth[p] > 0]
    if len(peaks) < 2:
        return max(minimum, 10)
    first = min(peaks)
    large_peaks = [p for p in peaks if p > max(first + 3, minimum)]
    if not large_peaks:
        return max(minimum, 10)
    second = max(large_peaks, key=lambda p: smooth[p])
    lo, hi = sorted((first, second))
    valley = min(range(lo + 1, hi), key=lambda i: smooth[i]) if hi > lo + 1 else lo + 1
    return max(minimum, int(valley + 1))


def frame_timeseries_from_clusters(cluster_df, nmono, nmic, box_volume, water_count, surf_mw):
    frame_acc = {}
    for row in cluster_df.itertuples(index=False):
        size = int(row.cluster_size)
        key = (int(row.timestep), float(row.time))
        acc = frame_acc.setdefault(key, {"sizes": [], "micelle_sizes": [], "free": 0})
        acc["sizes"].append(size)
        if size > nmic:
            acc["micelle_sizes"].append(size)
        if size < nmono:
            acc["free"] += size
    rows = []
    for (timestep, time), acc in sorted(frame_acc.items()):
        mic = np.array(acc["micelle_sizes"], dtype=float)
        n_free = acc["free"]
        cmc_reduced = n_free / box_volume if box_volume else math.nan
        cmc_wt = math.nan
        if not math.isnan(surf_mw) and water_count > 0:
            cmc_wt = 100.0 * (n_free * surf_mw) / (n_free * surf_mw + water_count * WATER_MW)
        rows.append(
            {
                "timestep": timestep,
                "time": time,
                "nmono": nmono,
                "nmic": nmic,
                "free_surfactants": n_free,
                "cmc_reduced_number_density": cmc_reduced,
                "cmc_wt_percent": cmc_wt,
                "micelle_count": int(len(mic)),
                "max_cluster_size": int(max(acc["sizes"]) if acc["sizes"] else 0),
                "nagg_number_average": float(mic.mean()) if len(mic) else 0.0,
                "nagg_weight_average": float((mic * mic).sum() / mic.sum()) if len(mic) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def classify_clusters(cluster_df, nmono, nmic):
    result = cluster_df.copy()
    result["is_monomeric"] = (result["cluster_size"] < nmono).astype(int)
    result["is_micelle"] = (result["cluster_size"] > nmic).astype(int)
    return result


def write_threshold_sensitivity(cluster_df, rawdir, nmono_values, nmic_values, box_volume, water_count, surf_mw):
    summary_rows = []
    timeseries_rows = []
    for nmono in nmono_values:
        for nmic in nmic_values:
            frame_df = frame_timeseries_from_clusters(cluster_df, nmono, nmic, box_volume, water_count, surf_mw)
            frame_df.insert(0, "sensitivity_nmono", nmono)
            frame_df.insert(1, "sensitivity_nmic", nmic)
            timeseries_rows.extend(frame_df.to_dict("records"))
            summary_rows.append(summarize_frame_df(frame_df, nmono, nmic))
    pd.DataFrame(summary_rows).to_csv(rawdir / "threshold_sensitivity_summary.csv", index=False)
    pd.DataFrame(timeseries_rows).to_csv(rawdir / "threshold_sensitivity_timeseries.csv", index=False)


def summarize_frame_df(frame_df, nmono, nmic):
    return {
        "nmono": nmono,
        "nmic": nmic,
        "frames_analyzed": len(frame_df),
        "cmc_wt_percent_mean": frame_df["cmc_wt_percent"].mean(),
        "cmc_wt_percent_std": frame_df["cmc_wt_percent"].std(ddof=1),
        "cmc_reduced_number_density_mean": frame_df["cmc_reduced_number_density"].mean(),
        "cmc_reduced_number_density_std": frame_df["cmc_reduced_number_density"].std(ddof=1),
        "nagg_number_average_mean": frame_df["nagg_number_average"].mean(),
        "nagg_number_average_std": frame_df["nagg_number_average"].std(ddof=1),
        "nagg_weight_average_mean": frame_df["nagg_weight_average"].mean(),
        "nagg_weight_average_std": frame_df["nagg_weight_average"].std(ddof=1),
        "micelle_count_mean": frame_df["micelle_count"].mean(),
        "micelle_count_std": frame_df["micelle_count"].std(ddof=1),
        "max_cluster_size_mean": frame_df["max_cluster_size"].mean(),
        "max_cluster_size_std": frame_df["max_cluster_size"].std(ddof=1),
    }


def parse_int_list(value):
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_run_file(run_dir, file_name):
    path = Path(file_name)
    if path.is_absolute():
        return path
    return run_dir / path


def estimate_frame_total(start_timestep, run_steps, dump_every, timestep_size, frame_stride, max_frames):
    if max_frames:
        return max_frames
    if not run_steps or run_steps < start_timestep:
        return None
    dump_interval_steps = int(dump_every or (round(10.0 / timestep_size) if timestep_size else 1000))
    frames = int((run_steps - start_timestep) / dump_interval_steps) + 1
    return max(1, math.ceil(frames / max(1, frame_stride)))


def plot_outputs(cluster_df, frame_df, outdir, system_name):
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    hist = cluster_df.groupby("cluster_size").size().reset_index(name="count")
    hist["probability"] = hist["count"] / hist["count"].sum()
    plt.figure(figsize=(6, 4))
    plt.plot(hist["cluster_size"], hist["probability"], marker="o", lw=1)
    plt.yscale("log")
    plt.xlabel("Cluster size")
    plt.ylabel("Probability")
    plt.title(f"{system_name} cluster-size distribution")
    plt.tight_layout()
    plt.savefig(figdir / "cluster_size_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(frame_df["time"], frame_df["free_surfactants"], label="free surfactants")
    plt.plot(frame_df["time"], frame_df["nagg_weight_average"], label="weight-avg Nagg")
    plt.plot(frame_df["time"], frame_df["max_cluster_size"], label="max cluster")
    plt.xlabel("DPD time")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "convergence_timeseries.png", dpi=200)
    plt.close()

    micelles = cluster_df[cluster_df["is_micelle"] == 1]
    if not micelles.empty:
        plt.figure(figsize=(6, 4))
        plt.hist(micelles["asphericity"], bins=40)
        plt.xlabel("Asphericity")
        plt.ylabel("Micelle observations")
        plt.tight_layout()
        plt.savefig(figdir / "asphericity_distribution.png", dpi=200)
        plt.close()


def analyze_run(args, run_dir):
    run_dir = Path(run_dir).resolve()
    system = run_dir.name
    outdir = Path(args.outdir).resolve() / system
    rawdir = outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)
    type_names = read_type_names(run_dir / "bead_type_assignment.csv")
    contact_types = set(args.contact_types.split(","))
    molecule_count = parse_input_count(run_dir / "input_data.txt")
    atom_mol, atom_type, molecule_atoms, contact_atoms = read_data_atoms(
        run_dir / "packed_polymer_and_solution.data", type_names, contact_types, molecule_count=molecule_count
    )
    molecule_ids = sorted(molecule_atoms)
    metadata = parse_lammps_input(run_dir / "lammps.in")
    timestep_size = args.timestep or metadata.get("timestep", 0.01)
    dump_every = metadata.get("dump_every", 1000)
    run_steps = args.final_timestep or metadata.get("run_steps", 0)
    pair_cutoffs = None
    if args.overlap_mode == "pair-cutoff":
        pair_cutoffs = parse_pair_cutoffs(run_dir / "lammps.in", args.cutoff)
    start_timestep = args.start_timestep
    if start_timestep is None:
        start_timestep = max(0, int(run_steps - args.last_time * (1.0 / timestep_size)))

    summary = read_system_summary(run_dir / "system_bead_summary.csv")
    box_volume = summary.get("box_volume", math.nan)
    water_count = summary.get("water_bead_count", math.nan)
    surf_mw = mol_weight_from_smiles(parse_input_smiles(run_dir / "input_data.txt"))

    raw_cluster_path = rawdir / "cluster_observations_unclassified.csv"
    size_counts = Counter()
    processed = 0
    reuse_clusters = args.reuse_clusters and raw_cluster_path.exists()
    pool = Pool(processes=args.workers) if (not reuse_clusters and args.workers and args.workers > 1) else None
    pending = []
    contact_set = set(contact_atoms)
    dump_path = resolve_run_file(run_dir, args.dump_file)

    def consume_result(result, writer):
        nonlocal processed
        rows, counts = result
        size_counts.update(counts)
        for row in rows:
            writer.writerow(row)
        processed += 1

    if not reuse_clusters:
        progress_total = estimate_frame_total(
            start_timestep,
            run_steps,
            dump_every,
            timestep_size,
            args.frame_stride,
            args.max_frames,
        )
        with dump_path.open(errors="ignore") as handle, raw_cluster_path.open("w", newline="") as raw, progress_bar(
            total=progress_total,
            desc=f"{system} frames",
            disable=args.no_progress,
        ) as pbar:
            try:
                if args.tail_seek:
                    seek_near_timestep(handle, os.path.getsize(dump_path), start_timestep, run_steps)
                writer = csv.DictWriter(
                    raw,
                    fieldnames=["timestep", "time", "cluster_id", "cluster_size", "rg", "asphericity"],
                )
                writer.writeheader()
                frame_index = 0
                while True:
                    frame = read_frame(handle)
                    if frame is None:
                        break
                    timestep, n_atoms, bounds, col = frame
                    if timestep < start_timestep:
                        for _ in range(n_atoms):
                            handle.readline()
                        continue
                    if (frame_index % args.frame_stride) != 0:
                        for _ in range(n_atoms):
                            handle.readline()
                        frame_index += 1
                        continue
                    box_lengths = np.array([hi - lo for lo, hi in bounds], dtype=float)
                    atom_ids = []
                    atom_positions = []
                    atom_mol_values = []
                    atom_type_values = []
                    contact_flags = []
                    for _ in range(n_atoms):
                        parts = handle.readline().split()
                        atom_id = int(parts[col["id"]])
                        mol_id = atom_mol.get(atom_id)
                        if mol_id is None:
                            continue
                        atom_ids.append(atom_id)
                        atom_positions.append(
                            [float(parts[col["x"]]), float(parts[col["y"]]), float(parts[col["z"]])]
                        )
                        atom_mol_values.append(mol_id)
                        atom_type_values.append(atom_type[atom_id])
                        contact_flags.append(atom_id in contact_set)
                    task = (
                        timestep,
                        timestep * timestep_size,
                        box_lengths,
                        np.asarray(atom_ids, dtype=np.int32),
                        np.asarray(atom_positions, dtype=np.float32),
                        np.asarray(contact_flags, dtype=bool),
                        np.asarray(atom_mol_values, dtype=np.int32),
                        np.asarray(atom_type_values, dtype=np.int32),
                        molecule_ids,
                        molecule_atoms,
                        args.cutoff,
                        pair_cutoffs,
                    )
                    if pool is None:
                        consume_result(analyze_frame_task(task), writer)
                        pbar.update(1)
                    else:
                        pending.append(pool.apply_async(analyze_frame_task, (task,)))
                        if len(pending) >= args.workers * args.prefetch:
                            consume_result(pending.pop(0).get(), writer)
                            pbar.update(1)
                    frame_index += 1
                    if args.max_frames and (processed + len(pending)) >= args.max_frames:
                        break
                while pending:
                    consume_result(pending.pop(0).get(), writer)
                    pbar.update(1)
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()

    if args.cluster_threshold:
        args.nmono = args.cluster_threshold
        args.nmic = args.cluster_threshold
    cluster_path = rawdir / "cluster_observations.csv"
    frame_path = rawdir / "frame_timeseries.csv"
    cluster_df = classify_clusters(pd.read_csv(raw_cluster_path), args.nmono, args.nmic)
    cluster_df.to_csv(cluster_path, index=False)
    frame_df = frame_timeseries_from_clusters(cluster_df, args.nmono, args.nmic, box_volume, water_count, surf_mw)
    frame_df.to_csv(frame_path, index=False)
    nmono_values = parse_int_list(args.sensitivity_nmono)
    nmic_values = parse_int_list(args.sensitivity_nmic)
    if nmono_values and nmic_values:
        write_threshold_sensitivity(cluster_df, rawdir, nmono_values, nmic_values, box_volume, water_count, surf_mw)
    dist = cluster_df.groupby("cluster_size").size().reset_index(name="count")
    dist["probability"] = dist["count"] / dist["count"].sum()
    dist.to_csv(rawdir / "cluster_size_distribution.csv", index=False)

    equil = frame_df
    frame_summary = summarize_frame_df(equil, args.nmono, args.nmic)
    summary_row = {
        "system": system,
        "frames_analyzed": len(frame_df),
        "start_timestep": start_timestep,
        "nmono": args.nmono,
        "nmic": args.nmic,
        "overlap_mode": args.overlap_mode,
        "contact_types": ",".join(sorted(contact_types)),
        "contact_cutoff": args.cutoff,
        "surfactant_mw": surf_mw,
        "water_bead_count": water_count,
        "box_volume": box_volume,
    }
    summary_row.update(frame_summary)
    pd.DataFrame([summary_row]).to_csv(outdir / "micelle_summary.csv", index=False)
    plot_outputs(cluster_df, frame_df, outdir, system)
    return summary_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--outdir", default="analysis/nonionic_micelles")
    parser.add_argument("--contact-types", default=",".join(sorted(TAIL_TYPES_DEFAULT)))
    parser.add_argument("--overlap-mode", choices=["pair-cutoff", "fixed"], default="pair-cutoff")
    parser.add_argument("--cutoff", type=float, default=1.2, help="fallback fixed overlap cutoff")
    parser.add_argument("--nmono", type=int, default=10, help="clusters smaller than nmono are monomeric for CMC")
    parser.add_argument("--nmic", type=int, default=10, help="clusters larger than nmic are micelles for Nagg")
    parser.add_argument("--sensitivity-nmono", default="5,10,15")
    parser.add_argument("--sensitivity-nmic", default="10,15,20,25,30")
    parser.add_argument("--min-micelle-size", type=int, default=10)
    parser.add_argument("--cluster-threshold", type=int, help="legacy single threshold; sets both nmono and nmic")
    parser.add_argument("--last-time", type=float, default=30000.0)
    parser.add_argument("--start-timestep", type=int)
    parser.add_argument("--final-timestep", type=int)
    parser.add_argument("--timestep", type=float)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--dump-file", default="dump.lammpstrj", help="trajectory file, relative to each run directory")
    parser.add_argument("--reuse-clusters", action="store_true", help="reuse raw/cluster_observations_unclassified.csv")
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes per sample for frame analysis")
    parser.add_argument("--prefetch", type=int, default=2, help="queued frame batches per worker")
    parser.add_argument("--tail-seek", action="store_true", default=True)
    parser.add_argument("--no-tail-seek", dest="tail_seek", action="store_false")
    args = parser.parse_args()

    summaries = [analyze_run(args, run_dir) for run_dir in args.run_dirs]
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(outdir / "micelle_summary_all.csv", index=False)


if __name__ == "__main__":
    main()
