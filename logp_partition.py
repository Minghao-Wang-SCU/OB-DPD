#!/usr/bin/env python3
"""Build, run, and analyze a water/octanol DPD partition simulation.

This script is intentionally independent from main.py. It creates a slab
system with water on one side, octanol on the other side, multiple solute
molecules, and then estimates C_octanol / C_water from solute molecule centers
in the production trajectory.
"""

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import subprocess
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from logp_param import ANDERSON_TYPES, DEFAULT_LOGP_BEAD_SOLUBILITY, load_logp_table, lookup_pair, solubility_aij
from logp_targets import collect_logp_values, robust_consensus, write_logp_report

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None


BEAD_RECIPES = {
    "water": ["H2O"],
    "octanol": ["CH3", "CH2CH2", "CH2CH2", "CH2CH2", "CH2OH"],
    "ethanol": ["CH3", "CH2OH"],
    "butanol": ["CH3", "CH2CH2", "CH2OH"],
    "hexane": ["CH3", "CH2CH2", "CH2CH2", "CH3"],
    "heptane": ["CH3", "CH2CH2", "CH2CH2", "CH2CH2"],
    "benzene": ["aCHCH", "aCHCH", "aCHCH"],
}

PHASE_BEADS = {"H2O", "CH3", "CH2CH2", "CH2OH"}
BOND_FORCE_CONSTANT = 150.0
ANGLE_FORCE_CONSTANT = 5.0
ARTICLE_ANGLE_FORCE_CONSTANT = 2.5
HEURISTIC_ANGLE_FORCE_CONSTANT = 2.0
WEAK_ANGLE_FORCE_CONSTANT = 1.0
DEFAULT_BOND_R0 = 0.39
DEFAULT_BEAD_SOLUBILITY = DEFAULT_LOGP_BEAD_SOLUBILITY
DEFAULT_HEAVY_ATOM_RADIUS_SCALE = 1.0
DEFAULT_BONDED_R0_FACTOR = 0.4
KNOWN_BEAD_HEAVY_ATOMS = {
    "H2O": 1,
    "2H2O": 2,
    "CH3": 1,
    "CH2": 1,
    "CH3CH2": 2,
    "CH2CH2": 2,
    "CH2OH": 2,
    "CH2NH2": 2,
    "CH2OCH2": 3,
    "CH3OCH2": 3,
    "aCHCH": 2,
    "CH2OSO3-": 6,
    "COO-": 3,
    "Na+": 1,
    "Cl-": 1,
}
ALKYL_BEADS = {"CH3", "CH2", "CH3CH2", "CH2CH2"}
ARTICLE_BACKBONE_BEADS = {"CH2", "CH2CH2"}
POLAR_HEAD_BEADS = {"CH2OH", "CH2NH2", "CH2OCH2", "CH3OCH2"}
AROMATIC_BEADS = {"aCHCH"}
IONIC_BEADS = {"CH2OSO3-", "COO-", "Na+", "Cl-"}
COUNTERION_BEADS = {"Na+": 1.0, "Cl-": -1.0}


class SimpleProgress:
    def __init__(self, total=None, desc="progress", unit="it", disable=False):
        self.total = total
        self.desc = desc
        self.unit = unit
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

    def set_postfix(self, values=None, **kwargs):
        return

    def _write(self):
        if self.disable:
            return
        if self.total:
            pct = 100.0 * self.count / self.total
            text = f"\r{self.desc}: {self.count}/{self.total} {self.unit} ({pct:5.1f}%)"
        else:
            text = f"\r{self.desc}: {self.count} {self.unit}"
        sys.stderr.write(text)
        sys.stderr.flush()


def progress_bar(total=None, desc="progress", unit="it", disable=False):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit=unit, disable=disable)
    return SimpleProgress(total=total, desc=desc, unit=unit, disable=disable)


def normal_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def default_lammps_executable():
    env_value = os.environ.get("LAMMPS_BIN")
    if env_value:
        return env_value
    return shutil.which("lmp_mpi") or shutil.which("lmp") or "lmp_mpi"


@dataclass
class MoleculeTemplate:
    name: str
    beads: list[str]
    coords: object = None

    @property
    def bonds(self):
        if self.name == "benzene" and len(self.beads) == 3:
            return [(1, 2), (2, 3), (3, 1)]
        return [(idx + 1, idx + 2) for idx in range(len(self.beads) - 1)]

    @property
    def angles(self):
        if self.name == "benzene" and len(self.beads) == 3:
            return [(1, 2, 3), (2, 3, 1), (3, 1, 2)]
        return [(idx + 1, idx + 2, idx + 3) for idx in range(len(self.beads) - 2)]


def parse_recipe(value, allow_custom=False):
    if value in BEAD_RECIPES:
        beads = BEAD_RECIPES[value]
        name = value
    else:
        beads = [part.strip() for part in value.split(",") if part.strip()]
        name = "custom"
    unknown = [bead for bead in beads if bead not in ANDERSON_TYPES]
    if unknown:
        if not allow_custom:
            raise ValueError(f"unsupported Anderson bead type(s): {', '.join(sorted(set(unknown)))}")
        for bead in unknown:
            if not re.fullmatch(r"[A-Za-z0-9_+\-]+", bead):
                raise ValueError(f"custom bead names must be alphanumeric/underscore: {bead}")
    return MoleculeTemplate(name=name, beads=beads)


def geometry_angle_params(template, left, center, right):
    coords = getattr(template, "coords", None)
    if coords is None:
        return None
    coords = np.asarray(coords, dtype=float)
    if len(coords) < max(left, center, right):
        return None
    vec_left = coords[left - 1] - coords[center - 1]
    vec_right = coords[right - 1] - coords[center - 1]
    norm = np.linalg.norm(vec_left) * np.linalg.norm(vec_right)
    if norm <= 0.0:
        return None
    cos_theta = float(np.dot(vec_left, vec_right) / norm)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta0 = math.degrees(math.acos(cos_theta))
    return HEURISTIC_ANGLE_FORCE_CONSTANT, theta0, "geometry"


def article_angle_params(bead_left, bead_center, bead_right):
    sides = {bead_left, bead_right}
    if bead_center in ARTICLE_BACKBONE_BEADS:
        if bead_left in ALKYL_BEADS and bead_right in ALKYL_BEADS:
            return ARTICLE_ANGLE_FORCE_CONSTANT, 105.0, "article_alkyl"
        if "CH2OH" in sides and ((sides - {"CH2OH"}) & ALKYL_BEADS):
            return ARTICLE_ANGLE_FORCE_CONSTANT, 125.0, "article_alcohol"
    return None


def heuristic_angle_params(bead_left, bead_center, bead_right):
    article = article_angle_params(bead_left, bead_center, bead_right)
    if article is not None:
        return article
    beads = {bead_left, bead_center, bead_right}
    if beads <= AROMATIC_BEADS:
        return HEURISTIC_ANGLE_FORCE_CONSTANT, 60.0, "heuristic_aromatic_ring"
    if bead_center in AROMATIC_BEADS:
        return HEURISTIC_ANGLE_FORCE_CONSTANT, 120.0, "heuristic_aromatic_center"
    if beads & IONIC_BEADS:
        return WEAK_ANGLE_FORCE_CONSTANT, 120.0, "heuristic_ionic_weak"
    if bead_center in ALKYL_BEADS and beads & POLAR_HEAD_BEADS:
        return HEURISTIC_ANGLE_FORCE_CONSTANT, 120.0, "heuristic_polar_alkyl"
    if beads <= ALKYL_BEADS:
        return HEURISTIC_ANGLE_FORCE_CONSTANT, 105.0, "heuristic_alkyl"
    return WEAK_ANGLE_FORCE_CONSTANT, 120.0, "heuristic_unknown_weak"


def angle_params_for_triplet(template, left, center, right, mode):
    if mode == "none":
        return None
    bead_left = template.beads[left - 1]
    bead_center = template.beads[center - 1]
    bead_right = template.beads[right - 1]
    if mode == "geometry":
        params = geometry_angle_params(template, left, center, right)
        if params is not None:
            return params
        return heuristic_angle_params(bead_left, bead_center, bead_right)
    if mode == "article":
        return article_angle_params(bead_left, bead_center, bead_right)
    if mode == "heuristic":
        return heuristic_angle_params(bead_left, bead_center, bead_right)
    raise ValueError(f"unsupported angle parameter mode: {mode}")


def angle_type_for_params(angle_type_ids, angle_coeffs, force_constant, theta0, label):
    key = (round(float(force_constant), 8), round(float(theta0), 8), str(label))
    if key in angle_type_ids:
        return angle_type_ids[key]
    angle_type = len(angle_type_ids) + 1
    angle_type_ids[key] = angle_type
    angle_coeffs.append(
        {
            "type": angle_type,
            "force_constant": float(force_constant),
            "theta0": float(theta0),
            "label": str(label),
        }
    )
    return angle_type


def add_molecule(
    atoms,
    bonds,
    angles,
    template,
    type_ids,
    bond_type_ids,
    angle_type_ids,
    angle_coeffs,
    angle_param_mode,
    mol_id,
    origin,
    spacing=0.45,
):
    first_atom = len(atoms) + 1
    direction = np.random.normal(size=3)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
    for idx, bead in enumerate(template.beads):
        pos = np.array(origin) + (idx - (len(template.beads) - 1) / 2.0) * spacing * direction
        atoms.append(
            {
                "id": len(atoms) + 1,
                "mol": mol_id,
                "type": type_ids[bead],
                "bead": bead,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            }
        )
    for left, right in template.bonds:
        key = pair_key(template.beads[left - 1], template.beads[right - 1])
        bonds.append((len(bonds) + 1, bond_type_ids[key], first_atom + left - 1, first_atom + right - 1))
    for left, center, right in template.angles:
        params = angle_params_for_triplet(template, left, center, right, angle_param_mode)
        if params is None:
            continue
        force_constant, theta0, label = params
        angle_type = angle_type_for_params(angle_type_ids, angle_coeffs, force_constant, theta0, label)
        angles.append((len(angles) + 1, angle_type, first_atom + left - 1, first_atom + center - 1, first_atom + right - 1))


def random_point(box, zlo, zhi, margin=0.8):
    lx, ly, _ = box
    return (
        random.uniform(margin, lx - margin),
        random.uniform(margin, ly - margin),
        random.uniform(zlo + margin, zhi - margin),
    )


def pair_key(bead_i, bead_j):
    return tuple(sorted((bead_i, bead_j)))


def cutoff_from_self_radii(table, bead_i, bead_j, default_cutoff):
    pair = lookup_pair(table, bead_i, bead_j)
    if pair is not None and not np.isnan(pair["R_ij"]):
        return float(pair["R_ij"])
    self_i = lookup_pair(table, bead_i, bead_i)
    self_j = lookup_pair(table, bead_j, bead_j)
    if self_i is not None and self_j is not None and not np.isnan(self_i["R_ij"]) and not np.isnan(self_j["R_ij"]):
        return 0.5 * (float(self_i["R_ij"]) + float(self_j["R_ij"]))
    return float(default_cutoff)


def corrected_cutoff(
    table,
    bead_i,
    bead_j,
    default_cutoff,
    correction_mode="missing",
    heavy_counts=None,
    heavy_radius_scale=DEFAULT_HEAVY_ATOM_RADIUS_SCALE,
):
    pair = lookup_pair(table, bead_i, bead_j)
    if pair is not None and not np.isnan(pair["R_ij"]):
        table_cutoff = float(pair["R_ij"])
    else:
        self_i = lookup_pair(table, bead_i, bead_i)
        self_j = lookup_pair(table, bead_j, bead_j)
        if self_i is not None and self_j is not None and not np.isnan(self_i["R_ij"]) and not np.isnan(self_j["R_ij"]):
            table_cutoff = 0.5 * (float(self_i["R_ij"]) + float(self_j["R_ij"]))
        else:
            table_cutoff = None
    if should_apply_heavy_correction(correction_mode, table_cutoff):
        value = heavy_atom_rij(table, bead_i, bead_j, counts=heavy_counts, scale=heavy_radius_scale)
        if value is not None:
            return value
    return float(table_cutoff) if table_cutoff is not None else float(default_cutoff)


def parse_pair_overrides(values):
    overrides = {}
    for value in values or []:
        try:
            pair_part, aij_part = value.split("=")
            bead_i, bead_j = pair_part.split(":")
            overrides[pair_key(bead_i.strip(), bead_j.strip())] = float(aij_part)
        except ValueError as exc:
            raise ValueError(f"pair override must look like BEAD1:BEAD2=Aij, got {value}") from exc
    return overrides


def parse_bead_solubility(values):
    solubility = dict(DEFAULT_BEAD_SOLUBILITY)
    for value in values or []:
        try:
            bead, sigma = value.split("=", 1)
            solubility[bead.strip()] = float(sigma)
        except ValueError as exc:
            raise ValueError(f"bead solubility must look like BEAD=VALUE, got {value}") from exc
    return solubility


def parse_bead_heavy_atoms(values):
    counts = dict(KNOWN_BEAD_HEAVY_ATOMS)
    for value in values or []:
        try:
            bead, count = value.split("=", 1)
            count = int(count)
        except ValueError as exc:
            raise ValueError(f"bead heavy atom count must look like BEAD=COUNT, got {value}") from exc
        if count <= 0:
            raise ValueError(f"heavy atom count must be positive for {bead}")
        counts[bead.strip()] = count
    return counts


def rdkit_heavy_atom_count(value):
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return None
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def bead_heavy_atom_count(bead, counts=None):
    counts = counts or KNOWN_BEAD_HEAVY_ATOMS
    if bead in counts:
        return int(counts[bead])
    parsed = rdkit_heavy_atom_count(bead)
    if parsed is not None and parsed > 0:
        return int(parsed)
    return None


def self_radius_from_table(table, bead):
    pair = lookup_pair(table, bead, bead)
    if pair is not None and not np.isnan(pair["R_ij"]):
        return 0.5 * float(pair["R_ij"])
    return None


def heavy_atom_radius(table, bead, counts=None, scale=DEFAULT_HEAVY_ATOM_RADIUS_SCALE, reference_bead="H2O"):
    table_radius = self_radius_from_table(table, bead)
    if table_radius is not None:
        return table_radius
    count = bead_heavy_atom_count(bead, counts)
    if count is None:
        return None
    ref_radius = self_radius_from_table(table, reference_bead)
    if ref_radius is None:
        ref_radius = 0.5
    ref_count = bead_heavy_atom_count(reference_bead, counts) or 1
    return float(scale) * ref_radius * (float(count) / float(ref_count)) ** (1.0 / 3.0)


def heavy_atom_rij(table, bead_i, bead_j, counts=None, scale=DEFAULT_HEAVY_ATOM_RADIUS_SCALE):
    radius_i = heavy_atom_radius(table, bead_i, counts=counts, scale=scale)
    radius_j = heavy_atom_radius(table, bead_j, counts=counts, scale=scale)
    if radius_i is None or radius_j is None:
        return None
    return float(radius_i + radius_j)


def should_apply_heavy_correction(mode, existing_value):
    if mode == "all":
        return True
    if mode == "missing":
        return existing_value is None or np.isnan(existing_value)
    return False


def solubility_pair_aij(bead_i, bead_j, solubility_values):
    if bead_i not in solubility_values or bead_j not in solubility_values:
        return None
    return solubility_aij(solubility_values[bead_i], solubility_values[bead_j], bead_i == bead_j)


def clamp_aij(value, min_aij=None, max_aij=None):
    if min_aij is not None:
        value = max(float(min_aij), value)
    if max_aij is not None:
        value = min(float(max_aij), value)
    return value


def pair_coeff_rows(
    type_names,
    table_path,
    sigma,
    overrides=None,
    default_missing_aij=None,
    bead_solubility=None,
    min_aij=None,
    max_aij=None,
    heavy_counts=None,
    heavy_correction="missing",
    heavy_radius_scale=DEFAULT_HEAVY_ATOM_RADIUS_SCALE,
):
    overrides = overrides or {}
    table = load_logp_table(table_path)
    rows = []
    for i, bead_i in enumerate(type_names, start=1):
        for j, bead_j in enumerate(type_names[i - 1 :], start=i):
            key = pair_key(bead_i, bead_j)
            pair = lookup_pair(table, bead_i, bead_j)
            cutoff = corrected_cutoff(
                table,
                bead_i,
                bead_j,
                sigma,
                correction_mode=heavy_correction,
                heavy_counts=heavy_counts,
                heavy_radius_scale=heavy_radius_scale,
            )
            if key in overrides:
                rows.append((i, j, float(overrides[key]), cutoff))
            elif pair is None and bead_solubility is not None:
                value = solubility_pair_aij(bead_i, bead_j, bead_solubility)
                if value is None and default_missing_aij is not None:
                    value = float(default_missing_aij)
                if value is None:
                    raise ValueError(
                        f"missing logP table parameter and solubility value for {bead_i}-{bead_j}"
                    )
                rows.append((i, j, clamp_aij(float(value), min_aij, max_aij), cutoff))
            elif pair is None and default_missing_aij is not None:
                rows.append((i, j, float(default_missing_aij), cutoff))
            elif pair is None:
                raise ValueError(f"missing logP table parameter for {bead_i}-{bead_j}")
            else:
                rows.append((i, j, float(pair["A_ij"]), cutoff))
    return rows


def infer_logp_bead_charge(bead):
    bead = str(bead).strip()
    if bead in COUNTERION_BEADS:
        return COUNTERION_BEADS[bead], "matched", "counterion bead"
    if bead in {"H2O", "2H2O"}:
        return 0.0, "matched", "water bead"
    if re.search(r"(^|[^A-Za-z0-9])Na\+$", bead) or bead.endswith("+"):
        return 1.0, "heuristic", "bead name ends with +"
    if bead.endswith("-"):
        return -1.0, "heuristic", "bead name ends with -"
    if Chem is not None and any(char in bead for char in "[]()=#@"):
        mol = Chem.MolFromSmiles(bead)
        if mol is not None:
            return float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())), "rdkit", "formal charge from bead SMILES"
    return 0.0, "neutral", "no formal charge marker"


def parse_bead_charges(values):
    charges = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError("--bead-charge must use BEAD=CHARGE")
        bead, value = item.split("=", 1)
        charges[bead.strip()] = float(value)
    return charges


def assign_type_charges(type_names, explicit_charges=None, charge_unit_scale=1.0):
    explicit_charges = explicit_charges or {}
    rows = []
    charges = []
    for type_id, bead in enumerate(type_names, start=1):
        if bead in explicit_charges:
            charge = float(explicit_charges[bead])
            status = "override"
            reason = "--bead-charge"
        else:
            charge, status, reason = infer_logp_bead_charge(bead)
        charge *= float(charge_unit_scale)
        charges.append(charge)
        rows.append(
            {
                "type_id": type_id,
                "bead": bead,
                "charge": charge,
                "status": status,
                "reason": reason,
            }
        )
    return charges, rows


def template_charge(template, charge_by_bead):
    return float(sum(charge_by_bead.get(bead, 0.0) for bead in template.beads))


def counterions_for_net_charge(net_charge):
    rounded = int(round(float(net_charge)))
    if abs(float(net_charge) - rounded) > 1.0e-6:
        raise ValueError(f"net charge must be close to an integer to add counterions, got {net_charge}")
    if rounded > 0:
        return {"Cl-": rounded}
    if rounded < 0:
        return {"Na+": -rounded}
    return {}


def write_logp_charge_assignment(path, rows, total_charge, counterion_counts):
    columns = ["type_id", "bead", "charge", "status", "reason"]
    write_csv(path / "charge_assignment.csv", rows, columns)
    summary_rows = [
        {"item": "net_charge_before_counterions", "value": total_charge},
        {"item": "Na+_counterions", "value": int(counterion_counts.get("Na+", 0))},
        {"item": "Cl-_counterions", "value": int(counterion_counts.get("Cl-", 0))},
        {"item": "net_charge_after_counterions", "value": 0.0},
    ]
    write_csv(path / "charge_summary.csv", summary_rows, ["item", "value"])


def ion_pair_overrides(type_names, table_path, existing_overrides, bead_solubility, type_charges):
    table = load_logp_table(table_path)
    existing_overrides = existing_overrides or {}
    water_key = "H2O" if "H2O" in type_names else None
    overrides = {}
    for i, bead_i in enumerate(type_names, start=1):
        for j, bead_j in enumerate(type_names[i - 1 :], start=i):
            key = pair_key(bead_i, bead_j)
            if lookup_pair(table, bead_i, bead_j) is not None or key in existing_overrides:
                continue
            charge_i = float(type_charges[i - 1])
            charge_j = float(type_charges[j - 1])
            if abs(charge_i) == 0.0 and abs(charge_j) == 0.0:
                continue
            if abs(charge_i) > 0.0 and abs(charge_j) > 0.0:
                overrides[key] = 25.0
                continue
            neutral = bead_j if abs(charge_i) > 0.0 else bead_i
            if water_key is not None:
                ref_pair = lookup_pair(table, water_key, neutral)
                ref_key = pair_key(water_key, neutral)
                if ref_key in existing_overrides:
                    overrides[key] = float(existing_overrides[ref_key])
                elif ref_pair is not None:
                    overrides[key] = float(ref_pair["A_ij"])
                elif bead_solubility is not None:
                    value = solubility_pair_aij(water_key, neutral, bead_solubility)
                    overrides[key] = 25.0 if value is None else float(value)
                else:
                    overrides[key] = 25.0
            else:
                overrides[key] = 25.0
    return overrides


def bond_type_data(
    templates,
    table_path,
    heavy_counts=None,
    heavy_correction="missing",
    heavy_radius_scale=DEFAULT_HEAVY_ATOM_RADIUS_SCALE,
    bonded_r0_factor=DEFAULT_BONDED_R0_FACTOR,
):
    table = load_logp_table(table_path)
    bond_type_ids = {}
    bond_coeffs = []
    for template in templates:
        for left, right in template.bonds:
            key = pair_key(template.beads[left - 1], template.beads[right - 1])
            if key in bond_type_ids:
                continue
            pair = lookup_pair(table, *key)
            r0 = DEFAULT_BOND_R0
            if pair is not None and not np.isnan(pair.get("r0", np.nan)):
                r0 = float(pair["r0"])
            elif heavy_correction in {"missing", "all"}:
                rij = heavy_atom_rij(table, key[0], key[1], counts=heavy_counts, scale=heavy_radius_scale)
                if rij is not None:
                    r0 = float(bonded_r0_factor) * float(rij)
            if heavy_correction == "all":
                rij = heavy_atom_rij(table, key[0], key[1], counts=heavy_counts, scale=heavy_radius_scale)
                if rij is not None:
                    r0 = float(bonded_r0_factor) * float(rij)
            bond_type_ids[key] = len(bond_type_ids) + 1
            bond_coeffs.append((bond_type_ids[key], BOND_FORCE_CONSTANT, r0, key[0], key[1]))
    return bond_type_ids, bond_coeffs


def write_lammps_data(
    path,
    atoms,
    bonds,
    angles,
    box,
    type_names,
    bond_coeffs,
    angle_coeffs,
    charge_enabled=False,
    type_charges=None,
):
    type_charges = type_charges or [0.0] * len(type_names)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("LAMMPS water/octanol partition data\n\n")
        handle.write(f"{len(atoms)} atoms\n")
        handle.write(f"{len(bonds)} bonds\n")
        handle.write(f"{len(angles)} angles\n\n")
        handle.write(f"{len(type_names)} atom types\n")
        if bonds:
            handle.write(f"{len(bond_coeffs)} bond types\n")
        if angles:
            handle.write(f"{len(angle_coeffs)} angle types\n")
        handle.write("\n")
        handle.write(f"0.0 {box[0]:.6f} xlo xhi\n")
        handle.write(f"0.0 {box[1]:.6f} ylo yhi\n")
        handle.write(f"0.0 {box[2]:.6f} zlo zhi\n\n")
        handle.write(f"Atoms # {'full' if charge_enabled else 'molecular'}\n\n")
        for atom in atoms:
            if charge_enabled:
                charge = float(type_charges[int(atom["type"]) - 1])
                handle.write(
                    f"{atom['id']} {atom['mol']} {atom['type']} {charge:.12e} "
                    f"{atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n"
                )
            else:
                handle.write(
                    f"{atom['id']} {atom['mol']} {atom['type']} "
                    f"{atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n"
                )
        if bonds:
            handle.write("\nBonds\n\n")
            for bond_id, bond_type, atom_i, atom_j in bonds:
                handle.write(f"{bond_id} {bond_type} {atom_i} {atom_j}\n")
        if angles:
            handle.write("\nAngles\n\n")
            for angle_id, angle_type, atom_i, atom_j, atom_k in angles:
                handle.write(f"{angle_id} {angle_type} {atom_i} {atom_j} {atom_k}\n")


def write_lammps_input(
    path,
    pair_rows,
    bond_coeffs,
    angle_coeffs,
    steps,
    dump_every,
    temperature,
    sigma,
    use_bonds,
    use_angles,
    timestep=0.01,
    ensemble="nve",
    pressure=23.7,
    pressure_damp=2.0,
    charge_enabled=False,
    type_charges=None,
    charge_lambda=0.25,
    coul_cutoff=3.0,
    kspace_accuracy=1.0e-4,
):
    global_cutoff = max([float(row[3]) for row in pair_rows] or [1.0])
    type_charges = type_charges or []
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            f"""# Water/octanol logP partition sampling
dimension       3
units           lj
boundary        p p p
atom_style      {'full' if charge_enabled else 'molecular'}
read_data       logp_partition.data

mass * 1.0
variable        gamma equal {sigma}
velocity        all create {temperature} 92894 mom yes rot yes dist gaussian

neighbor        0.5 bin
neigh_modify    every 1 delay 0 one 4000
comm_modify     vel yes

"""
        )
        if use_bonds:
            handle.write("bond_style      harmonic\n")
            for bond_type, force_constant, r0, bead_i, bead_j in bond_coeffs:
                handle.write(f"bond_coeff      {bond_type} {force_constant:.6f} {r0:.6f} # {bead_i}:{bead_j}\n")
            handle.write("\n")
        if use_angles and angle_coeffs:
            handle.write("angle_style     harmonic\n")
            for coeff in sorted(angle_coeffs, key=lambda item: item["type"]):
                label = coeff.get("label", "")
                comment = f" # {label}" if label else ""
                handle.write(
                    f"angle_coeff     {coeff['type']} {coeff['force_constant']:.6f} "
                    f"{coeff['theta0']:.6f}{comment}\n"
                )
            handle.write("\n")
        if charge_enabled:
            handle.write(
                f"pair_style      dpd/coul/slater/long {temperature} {global_cutoff:.6f} "
                f"92894 {float(charge_lambda):.6g} {float(coul_cutoff):.6g}\n"
            )
            handle.write(f"kspace_style   pppm {float(kspace_accuracy):.1e}\n")
            for type_id, charge in enumerate(type_charges, start=1):
                handle.write(f"set type {type_id} charge {float(charge):.12e}\n")
        else:
            handle.write(f"pair_style      dpd {temperature} {global_cutoff:.6f} 92894\n")
        for i, j, aij, cutoff in pair_rows:
            if charge_enabled:
                charged = abs(float(type_charges[i - 1])) > 0.0 and abs(float(type_charges[j - 1])) > 0.0
                handle.write(f"pair_coeff      {i} {j} {aij:.6f} ${{gamma}}{' yes' if charged else ''}\n")
            else:
                handle.write(f"pair_coeff      {i} {j} {aij:.6f} ${{gamma}} {cutoff:.6f}\n")
        if ensemble == "nph":
            fix_line = f"fix             1 all nph iso {float(pressure):.6g} {float(pressure):.6g} {float(pressure_damp):.6g}"
        elif ensemble == "nve":
            fix_line = "fix             1 all nve"
        else:
            raise ValueError(f"unsupported partition ensemble: {ensemble}")
        handle.write(
            f"""
{fix_line}
timestep        {float(timestep):.6g}
thermo          {dump_every}
thermo_modify   lost ignore flush yes lost/bond ignore
	dump            1 all custom {dump_every} dump.lammpstrj id mol type {'q ' if charge_enabled else ''}x y z

run             {steps}
"""
        )


def write_density_lammps_input(
    path,
    pair_rows,
    bond_coeffs,
    angle_coeffs,
    steps,
    dump_every,
    temperature,
    pressure,
    sigma,
    use_bonds,
    use_angles,
    charge_enabled=False,
    type_charges=None,
    charge_lambda=0.25,
    coul_cutoff=3.0,
    kspace_accuracy=1.0e-4,
):
    global_cutoff = max([float(row[3]) for row in pair_rows] or [1.0])
    type_charges = type_charges or []
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            f"""# Pure liquid density fitting
dimension       3
units           lj
boundary        p p p
atom_style      {'full' if charge_enabled else 'molecular'}
read_data       pure_density.data

mass * 1.0
variable        gamma equal {sigma}
velocity        all create {temperature} 92894 mom yes rot yes dist gaussian

neighbor        0.5 bin
neigh_modify    every 1 delay 0 one 4000
comm_modify     vel yes

"""
        )
        if use_bonds:
            handle.write("bond_style      harmonic\n")
            for bond_type, force_constant, r0, bead_i, bead_j in bond_coeffs:
                handle.write(f"bond_coeff      {bond_type} {force_constant:.6f} {r0:.6f} # {bead_i}:{bead_j}\n")
            handle.write("\n")
        if use_angles and angle_coeffs:
            handle.write("angle_style     harmonic\n")
            for coeff in sorted(angle_coeffs, key=lambda item: item["type"]):
                label = coeff.get("label", "")
                comment = f" # {label}" if label else ""
                handle.write(
                    f"angle_coeff     {coeff['type']} {coeff['force_constant']:.6f} "
                    f"{coeff['theta0']:.6f}{comment}\n"
                )
            handle.write("\n")
        if charge_enabled:
            handle.write(
                f"pair_style      dpd/coul/slater/long {temperature} {global_cutoff:.6f} "
                f"92894 {float(charge_lambda):.6g} {float(coul_cutoff):.6g}\n"
            )
            handle.write(f"kspace_style   pppm {float(kspace_accuracy):.1e}\n")
            for type_id, charge in enumerate(type_charges, start=1):
                handle.write(f"set type {type_id} charge {float(charge):.12e}\n")
        else:
            handle.write(f"pair_style      dpd {temperature} {global_cutoff:.6f} 92894\n")
        for i, j, aij, cutoff in pair_rows:
            if charge_enabled:
                charged = abs(float(type_charges[i - 1])) > 0.0 and abs(float(type_charges[j - 1])) > 0.0
                handle.write(f"pair_coeff      {i} {j} {aij:.6f} ${{gamma}}{' yes' if charged else ''}\n")
            else:
                handle.write(f"pair_coeff      {i} {j} {aij:.6f} ${{gamma}} {cutoff:.6f}\n")
        handle.write(
            f"""
fix             1 all npt temp {temperature} {temperature} 1.0 iso {pressure} {pressure} 10.0
timestep        0.01
thermo          {dump_every}
thermo_modify   lost ignore flush yes lost/bond ignore
dump            1 all custom {dump_every} dump.lammpstrj id mol type {'q ' if charge_enabled else ''}x y z

run             {steps}
"""
        )


def build_system(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    allow_custom = getattr(args, "allow_custom_beads", False)
    heavy_counts = getattr(args, "bead_heavy_atom_counts", None)
    heavy_correction = getattr(args, "heavy_atom_correction", "missing")
    heavy_radius_scale = float(getattr(args, "heavy_radius_scale", DEFAULT_HEAVY_ATOM_RADIUS_SCALE))
    bonded_r0_factor = float(getattr(args, "bonded_r0_factor", DEFAULT_BONDED_R0_FACTOR))
    angle_param_mode = getattr(args, "angle_param_mode", None) or "heuristic"
    solute = parse_recipe(args.solute, allow_custom=allow_custom)
    water = parse_recipe("water")
    organic_name = getattr(args, "organic_solvent", "octanol")
    organic = parse_recipe(organic_name, allow_custom=allow_custom)
    box = tuple(args.box)
    half_z = box[2] / 2.0

    beads_per_water = len(water.beads)
    beads_per_organic = len(organic.beads)
    half_volume = box[0] * box[1] * half_z
    water_density = float(getattr(args, "water_density", None) or args.density)
    organic_density = float(getattr(args, "organic_density", None) or getattr(args, "octanol_density", None) or args.density)
    water_count = int(getattr(args, "water_count", 0) or 0)
    organic_count = int(getattr(args, "organic_count", 0) or 0)
    if water_count <= 0:
        water_count = max(1, int(water_density * half_volume / beads_per_water))
    if organic_count <= 0:
        organic_count = max(1, int(organic_density * half_volume / beads_per_organic))
    n_solute = int(args.n_solute)
    solute_bead_fraction = getattr(args, "solute_bead_fraction", None)
    if solute_bead_fraction is not None:
        solvent_beads = water_count * beads_per_water + organic_count * beads_per_organic
        n_solute = max(1, int(round(float(solute_bead_fraction) * solvent_beads / len(solute.beads))))

    explicit_charges = parse_bead_charges(getattr(args, "bead_charge", []))
    charge_method = getattr(args, "charge_method", "auto")
    initial_type_names = sorted(set(water.beads + organic.beads + solute.beads))
    initial_type_charges, _ = assign_type_charges(
        initial_type_names,
        explicit_charges=explicit_charges,
        charge_unit_scale=getattr(args, "charge_unit_scale", 1.0),
    )
    initial_charge_by_bead = dict(zip(initial_type_names, initial_type_charges))
    charge_enabled = charge_method == "explicit" or (
        charge_method == "auto" and any(abs(charge) > 1.0e-12 for charge in initial_type_charges)
    )
    net_charge = 0.0
    counterion_counts = {}
    if charge_enabled:
        net_charge = (
            water_count * template_charge(water, initial_charge_by_bead)
            + organic_count * template_charge(organic, initial_charge_by_bead)
            + n_solute * template_charge(solute, initial_charge_by_bead)
        )
        counterion_counts = counterions_for_net_charge(net_charge)

    counterion_templates = [MoleculeTemplate(name=bead, beads=[bead]) for bead, count in counterion_counts.items() if count > 0]
    type_names = sorted(set(water.beads + organic.beads + solute.beads + list(counterion_counts.keys())))
    type_charges, charge_rows = assign_type_charges(
        type_names,
        explicit_charges=explicit_charges,
        charge_unit_scale=getattr(args, "charge_unit_scale", 1.0),
    )
    type_ids = {bead: idx + 1 for idx, bead in enumerate(type_names)}
    bond_type_ids, bond_coeffs = bond_type_data(
        [water, organic, solute] + counterion_templates,
        args.table,
        heavy_counts=heavy_counts,
        heavy_correction=heavy_correction,
        heavy_radius_scale=heavy_radius_scale,
        bonded_r0_factor=bonded_r0_factor,
    )
    atoms = []
    bonds = []
    angles = []
    angle_type_ids = {}
    angle_coeffs = []
    water_mol_ids = []
    organic_mol_ids = []
    solute_mol_ids = []
    counterion_mol_ids = []
    mol_id = 1

    for _ in range(water_count):
        add_molecule(
            atoms,
            bonds,
            angles,
            water,
            type_ids,
            bond_type_ids,
            angle_type_ids,
            angle_coeffs,
            angle_param_mode,
            mol_id,
            random_point(box, 0.0, half_z),
        )
        water_mol_ids.append(mol_id)
        mol_id += 1

    for _ in range(organic_count):
        add_molecule(
            atoms,
            bonds,
            angles,
            organic,
            type_ids,
            bond_type_ids,
            angle_type_ids,
            angle_coeffs,
            angle_param_mode,
            mol_id,
            random_point(box, half_z, box[2]),
        )
        organic_mol_ids.append(mol_id)
        mol_id += 1

    for idx in range(n_solute):
        if args.solute_init == "water":
            zlo, zhi = 0.0, half_z
        elif args.solute_init == "octanol":
            zlo, zhi = half_z, box[2]
        elif args.solute_init == "interface":
            zlo, zhi = half_z - args.interface_width / 2.0, half_z + args.interface_width / 2.0
        else:
            zlo, zhi = (0.0, half_z) if idx % 2 == 0 else (half_z, box[2])
        add_molecule(
            atoms,
            bonds,
            angles,
            solute,
            type_ids,
            bond_type_ids,
            angle_type_ids,
            angle_coeffs,
            angle_param_mode,
            mol_id,
            random_point(box, zlo, zhi),
        )
        solute_mol_ids.append(mol_id)
        mol_id += 1

    for bead, count in counterion_counts.items():
        ion = MoleculeTemplate(name=bead, beads=[bead])
        for _ in range(int(count)):
            add_molecule(
                atoms,
                bonds,
                angles,
                ion,
                type_ids,
                bond_type_ids,
                angle_type_ids,
                angle_coeffs,
                angle_param_mode,
                mol_id,
                random_point(box, 0.0, half_z),
            )
            counterion_mol_ids.append(mol_id)
            mol_id += 1

    overrides = getattr(args, "pair_overrides", {})
    default_missing_aij = getattr(args, "default_missing_aij", None)
    bead_solubility = getattr(args, "bead_solubility_values", None)
    if charge_enabled:
        overrides = dict(overrides)
        overrides.update(ion_pair_overrides(type_names, args.table, overrides, bead_solubility, type_charges))
    pair_rows = pair_coeff_rows(
        type_names,
        args.table,
        args.sigma,
        overrides,
        default_missing_aij,
        bead_solubility,
        getattr(args, "min_aij", None),
        getattr(args, "max_aij", None),
        heavy_counts=heavy_counts,
        heavy_correction=heavy_correction,
        heavy_radius_scale=heavy_radius_scale,
    )
    write_lammps_data(
        outdir / "logp_partition.data",
        atoms,
        bonds,
        angles,
        box,
        type_names,
        bond_coeffs,
        angle_coeffs,
        charge_enabled=charge_enabled,
        type_charges=type_charges,
    )
    write_lammps_input(
        outdir / "logp_partition.in",
        pair_rows,
        bond_coeffs,
        angle_coeffs,
        args.steps,
        args.dump_every,
        args.temperature,
        args.sigma,
        bool(bonds),
        bool(angles),
        getattr(args, "timestep", 0.01),
        getattr(args, "ensemble", "nve"),
        getattr(args, "pressure", 23.7),
        getattr(args, "pressure_damp", 2.0),
        charge_enabled=charge_enabled,
        type_charges=type_charges,
        charge_lambda=getattr(args, "charge_lambda", 0.25),
        coul_cutoff=getattr(args, "coul_cutoff", 3.0),
        kspace_accuracy=getattr(args, "kspace_accuracy", 1.0e-4),
    )
    if charge_enabled:
        write_logp_charge_assignment(outdir, charge_rows, net_charge, counterion_counts)

    metadata = {
        "solute": args.solute,
        "solute_beads": solute.beads,
        "organic_solvent": organic_name,
        "organic_beads": organic.beads,
        "water_mol_ids": water_mol_ids,
        "organic_mol_ids": organic_mol_ids,
        "solute_mol_ids": solute_mol_ids,
        "counterion_mol_ids": counterion_mol_ids,
        "counterion_counts": counterion_counts,
        "water_molecules": water_count,
        "organic_molecules": organic_count,
        "octanol_molecules": organic_count,
        "n_solute": n_solute,
        "box": list(box),
        "density": args.density,
        "water_density": water_density,
        "organic_density": organic_density,
        "octanol_density": organic_density,
        "solute_bead_fraction": solute_bead_fraction,
        "interface_z": half_z,
        "interface_width": args.interface_width,
        "analysis_method": getattr(args, "analysis_method", "fixed"),
        "ensemble": getattr(args, "ensemble", "nve"),
        "pressure": float(getattr(args, "pressure", 23.7)),
        "pressure_damp": float(getattr(args, "pressure_damp", 2.0)),
        "heavy_atom_correction": heavy_correction,
        "heavy_radius_scale": heavy_radius_scale,
        "bonded_r0_factor": bonded_r0_factor,
        "angle_param_mode": angle_param_mode,
        "charge_method": charge_method,
        "charge_enabled": charge_enabled,
        "type_charges": type_charges,
        "charge_lambda": float(getattr(args, "charge_lambda", 0.25)),
        "coul_cutoff": float(getattr(args, "coul_cutoff", 3.0)),
        "kspace_accuracy": float(getattr(args, "kspace_accuracy", 1.0e-4)),
        "net_charge_before_counterions": net_charge,
        "bead_heavy_atom_counts": {
            bead: bead_heavy_atom_count(bead, heavy_counts)
            for bead in type_names
        },
        "type_ids": type_ids,
        "type_names": type_names,
        "bond_coeffs": [
            {"type": bond_type, "force_constant": force_constant, "r0": r0, "bead_i": bead_i, "bead_j": bead_j}
            for bond_type, force_constant, r0, bead_i, bead_j in bond_coeffs
        ],
        "angle_coeffs": angle_coeffs if angles else [],
        "pair_coeffs": [
            {"i": i, "j": j, "bead_i": type_names[i - 1], "bead_j": type_names[j - 1], "A_ij": a, "R_ij": r}
            for i, j, a, r in pair_rows
        ],
    }
    with open(outdir / "logp_partition_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return outdir


def arithmetic_self_mixing_overrides(beads, overrides):
    mixed = dict(overrides)
    unique_beads = sorted(set(beads))
    for idx, bead_i in enumerate(unique_beads):
        for bead_j in unique_beads[idx + 1 :]:
            self_i = pair_key(bead_i, bead_i)
            self_j = pair_key(bead_j, bead_j)
            if self_i in mixed and self_j in mixed:
                mixed[pair_key(bead_i, bead_j)] = 0.5 * (mixed[self_i] + mixed[self_j])
    return mixed


def build_density_system(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    allow_custom = getattr(args, "allow_custom_beads", False)
    heavy_counts = getattr(args, "bead_heavy_atom_counts", None)
    heavy_correction = getattr(args, "heavy_atom_correction", "missing")
    heavy_radius_scale = float(getattr(args, "heavy_radius_scale", DEFAULT_HEAVY_ATOM_RADIUS_SCALE))
    bonded_r0_factor = float(getattr(args, "bonded_r0_factor", DEFAULT_BONDED_R0_FACTOR))
    angle_param_mode = getattr(args, "angle_param_mode", None) or "heuristic"
    molecule = parse_recipe(args.solute, allow_custom=allow_custom)
    n_molecules = int(getattr(args, "n_molecules", 100))
    initial_density = float(getattr(args, "initial_density", 3.0))
    total_beads = max(1, n_molecules * len(molecule.beads))
    box_length = (total_beads / initial_density) ** (1.0 / 3.0)
    box = (box_length, box_length, box_length)

    explicit_charges = parse_bead_charges(getattr(args, "bead_charge", []))
    charge_method = getattr(args, "charge_method", "auto")
    initial_type_names = sorted(set(molecule.beads))
    initial_type_charges, _ = assign_type_charges(
        initial_type_names,
        explicit_charges=explicit_charges,
        charge_unit_scale=getattr(args, "charge_unit_scale", 1.0),
    )
    initial_charge_by_bead = dict(zip(initial_type_names, initial_type_charges))
    charge_enabled = charge_method == "explicit" or (
        charge_method == "auto" and any(abs(charge) > 1.0e-12 for charge in initial_type_charges)
    )
    net_charge = 0.0
    counterion_counts = {}
    if charge_enabled:
        net_charge = n_molecules * template_charge(molecule, initial_charge_by_bead)
        counterion_counts = counterions_for_net_charge(net_charge)
    counterion_templates = [MoleculeTemplate(name=bead, beads=[bead]) for bead, count in counterion_counts.items() if count > 0]

    type_names = sorted(set(molecule.beads + list(counterion_counts.keys())))
    type_charges, charge_rows = assign_type_charges(
        type_names,
        explicit_charges=explicit_charges,
        charge_unit_scale=getattr(args, "charge_unit_scale", 1.0),
    )
    type_ids = {bead: idx + 1 for idx, bead in enumerate(type_names)}
    bond_type_ids, bond_coeffs = bond_type_data(
        [molecule] + counterion_templates,
        args.table,
        heavy_counts=heavy_counts,
        heavy_correction=heavy_correction,
        heavy_radius_scale=heavy_radius_scale,
        bonded_r0_factor=bonded_r0_factor,
    )
    atoms = []
    bonds = []
    angles = []
    angle_type_ids = {}
    angle_coeffs = []
    for mol_id in range(1, n_molecules + 1):
        add_molecule(
            atoms,
            bonds,
            angles,
            molecule,
            type_ids,
            bond_type_ids,
            angle_type_ids,
            angle_coeffs,
            angle_param_mode,
            mol_id,
            random_point(box, 0.0, box[2], margin=0.2),
        )
    next_mol_id = n_molecules + 1
    for bead, count in counterion_counts.items():
        ion = MoleculeTemplate(name=bead, beads=[bead])
        for _ in range(int(count)):
            add_molecule(
                atoms,
                bonds,
                angles,
                ion,
                type_ids,
                bond_type_ids,
                angle_type_ids,
                angle_coeffs,
                angle_param_mode,
                next_mol_id,
                random_point(box, 0.0, box[2], margin=0.2),
            )
            next_mol_id += 1

    overrides = getattr(args, "pair_overrides", {})
    if getattr(args, "density_mixing_rule", "self_arithmetic") == "self_arithmetic":
        overrides = arithmetic_self_mixing_overrides(molecule.beads, overrides)
    default_missing_aij = getattr(args, "default_missing_aij", None)
    bead_solubility = getattr(args, "bead_solubility_values", None)
    if charge_enabled:
        overrides = dict(overrides)
        overrides.update(ion_pair_overrides(type_names, args.table, overrides, bead_solubility, type_charges))
    pair_rows = pair_coeff_rows(
        type_names,
        args.table,
        args.sigma,
        overrides,
        default_missing_aij,
        bead_solubility,
        getattr(args, "min_aij", None),
        getattr(args, "max_aij", None),
        heavy_counts=heavy_counts,
        heavy_correction=heavy_correction,
        heavy_radius_scale=heavy_radius_scale,
    )
    write_lammps_data(
        outdir / "pure_density.data",
        atoms,
        bonds,
        angles,
        box,
        type_names,
        bond_coeffs,
        angle_coeffs,
        charge_enabled=charge_enabled,
        type_charges=type_charges,
    )
    write_density_lammps_input(
        outdir / "pure_density.in",
        pair_rows,
        bond_coeffs,
        angle_coeffs,
        int(getattr(args, "steps", 30000)),
        int(getattr(args, "dump_every", 1000)),
        float(getattr(args, "temperature", 1.0)),
        float(getattr(args, "pressure", 23.7)),
        float(getattr(args, "sigma", 1.0)),
        bool(bonds),
        bool(angles),
        charge_enabled=charge_enabled,
        type_charges=type_charges,
        charge_lambda=getattr(args, "charge_lambda", 0.25),
        coul_cutoff=getattr(args, "coul_cutoff", 3.0),
        kspace_accuracy=getattr(args, "kspace_accuracy", 1.0e-4),
    )
    if charge_enabled:
        write_logp_charge_assignment(outdir, charge_rows, net_charge, counterion_counts)
    metadata = {
        "molecule": args.solute,
        "molecule_beads": molecule.beads,
        "molecules": n_molecules,
        "atoms": len(atoms),
        "counterion_counts": counterion_counts,
        "initial_density": initial_density,
        "pressure": float(getattr(args, "pressure", 23.7)),
        "heavy_atom_correction": heavy_correction,
        "heavy_radius_scale": heavy_radius_scale,
        "bonded_r0_factor": bonded_r0_factor,
        "angle_param_mode": angle_param_mode,
        "charge_method": charge_method,
        "charge_enabled": charge_enabled,
        "type_charges": type_charges,
        "charge_lambda": float(getattr(args, "charge_lambda", 0.25)),
        "coul_cutoff": float(getattr(args, "coul_cutoff", 3.0)),
        "kspace_accuracy": float(getattr(args, "kspace_accuracy", 1.0e-4)),
        "net_charge_before_counterions": net_charge,
        "bead_heavy_atom_counts": {
            bead: bead_heavy_atom_count(bead, heavy_counts)
            for bead in type_names
        },
        "type_ids": type_ids,
        "type_names": type_names,
        "bond_coeffs": [
            {"type": bond_type, "force_constant": force_constant, "r0": r0, "bead_i": bead_i, "bead_j": bead_j}
            for bond_type, force_constant, r0, bead_i, bead_j in bond_coeffs
        ],
        "angle_coeffs": angle_coeffs if angles else [],
        "pair_coeffs": [
            {"i": i, "j": j, "bead_i": type_names[i - 1], "bead_j": type_names[j - 1], "A_ij": a, "R_ij": r}
            for i, j, a, r in pair_rows
        ],
    }
    with open(outdir / "pure_density_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return outdir


def read_dump_frames(dump_path, keep_fraction):
    frames = []
    with open(dump_path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            timestep = int(handle.readline().strip())
            handle.readline()
            natoms = int(handle.readline().strip())
            handle.readline()
            for _ in range(3):
                handle.readline()
            columns_line = handle.readline().strip().split()[2:]
            rows = []
            for _ in range(natoms):
                values = handle.readline().strip().split()
                rows.append(dict(zip(columns_line, values)))
            frames.append((timestep, rows))
    if not frames:
        raise ValueError(f"no frames found in {dump_path}")
    start = max(0, int(len(frames) * (1.0 - keep_fraction)))
    return frames[start:]


def read_partition_counts(dump_path, metadata, water_zmin, water_zmax, octanol_zmin, octanol_zmax):
    box_z = float(metadata["box"][2])
    solute_ids = set(int(item) for item in metadata["solute_mol_ids"])
    counts = []
    with open(dump_path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            timestep = int(handle.readline().strip())
            handle.readline()
            natoms = int(handle.readline().strip())
            bounds_header = handle.readline()
            if not bounds_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"unexpected dump format in {dump_path}: missing BOX BOUNDS")
            for _ in range(3):
                handle.readline()
            columns = handle.readline().strip().split()[2:]
            try:
                mol_index = columns.index("mol")
                z_index = columns.index("z")
            except ValueError as exc:
                raise ValueError(f"dump must contain mol and z columns: {dump_path}") from exc

            by_mol = {}
            for _ in range(natoms):
                values = handle.readline().strip().split()
                mol = int(values[mol_index])
                if mol in solute_ids:
                    by_mol.setdefault(mol, []).append(float(values[z_index]) % box_z)

            water = octanol = ignored = 0
            for zs in by_mol.values():
                center_z = float(np.mean(zs))
                if water_zmin < center_z < water_zmax:
                    water += 1
                elif octanol_zmin < center_z < octanol_zmax:
                    octanol += 1
                else:
                    ignored += 1
            counts.append((timestep, water, octanol, ignored))
    if not counts:
        raise ValueError(f"no frames found in {dump_path}")
    return counts


def read_dump_box_volumes(dump_path, keep_fraction):
    frames = []
    with open(dump_path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            timestep = int(handle.readline().strip())
            handle.readline()
            natoms = int(handle.readline().strip())
            bounds_header = handle.readline()
            if not bounds_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"unexpected dump format in {dump_path}: missing BOX BOUNDS")
            lengths = []
            for _ in range(3):
                parts = handle.readline().strip().split()
                lengths.append(float(parts[1]) - float(parts[0]))
            columns_line = handle.readline()
            if not columns_line.startswith("ITEM: ATOMS"):
                raise ValueError(f"unexpected dump format in {dump_path}: missing ATOMS")
            for _ in range(natoms):
                handle.readline()
            frames.append((timestep, lengths[0] * lengths[1] * lengths[2]))
    if not frames:
        raise ValueError(f"no frames found in {dump_path}")
    start = max(0, int(len(frames) * (1.0 - keep_fraction)))
    return frames[start:]


def circular_box_center(values, lo, hi):
    length = float(hi) - float(lo)
    if length <= 0.0:
        return float(np.mean(values))
    fractional = [((float(value) - float(lo)) % length) / length for value in values]
    angles = 2.0 * math.pi * np.asarray(fractional, dtype=float)
    sin_mean = float(np.mean(np.sin(angles)))
    cos_mean = float(np.mean(np.cos(angles)))
    if abs(sin_mean) < 1.0e-14 and abs(cos_mean) < 1.0e-14:
        frac = float(np.mean(fractional)) % 1.0
    else:
        frac = (math.atan2(sin_mean, cos_mean) / (2.0 * math.pi)) % 1.0
    return frac * length


def iter_partition_molecule_centers(dump_path, metadata, equilibration_steps=0):
    water_ids = set(int(item) for item in metadata.get("water_mol_ids", []))
    organic_ids = set(int(item) for item in metadata.get("organic_mol_ids", []))
    solute_ids = set(int(item) for item in metadata["solute_mol_ids"])
    group_by_mol = {}
    for mol in water_ids:
        group_by_mol[mol] = "water"
    for mol in organic_ids:
        group_by_mol[mol] = "organic"
    for mol in solute_ids:
        group_by_mol[mol] = "solute"

    with open(dump_path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            timestep = int(handle.readline().strip())
            handle.readline()
            natoms = int(handle.readline().strip())
            bounds_header = handle.readline()
            if not bounds_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"unexpected dump format in {dump_path}: missing BOX BOUNDS")
            bounds = []
            for _ in range(3):
                lo, hi = handle.readline().strip().split()[:2]
                bounds.append((float(lo), float(hi)))
            zlo, zhi = bounds[2]
            z_length = zhi - zlo
            if z_length <= 0.0:
                raise ValueError(f"invalid z box length in {dump_path}: {z_length}")
            columns = handle.readline().strip().split()[2:]
            try:
                mol_index = columns.index("mol")
                z_index = columns.index("z")
            except ValueError as exc:
                raise ValueError(f"dump must contain mol and z columns: {dump_path}") from exc

            by_mol = {}
            for _ in range(natoms):
                values = handle.readline().strip().split()
                mol = int(values[mol_index])
                group = group_by_mol.get(mol)
                if group is None:
                    continue
                by_mol.setdefault(mol, [group, []])[1].append(float(values[z_index]))
            if timestep < equilibration_steps:
                continue
            centers = {"water": [], "organic": [], "solute": []}
            for group, z_values in by_mol.values():
                centers[group].append(circular_box_center(z_values, zlo, zhi))
            lengths = tuple(float(hi - lo) for lo, hi in bounds)
            yield timestep, centers, lengths


def circular_interface_groups(indices, n_slabs):
    if not indices:
        return []
    groups = []
    current = [indices[0]]
    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == n_slabs - 1:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    return groups


def choose_analysis_frames(dump_path, metadata, args):
    equilibration_steps = int(getattr(args, "equilibration_steps", 0) or 0)
    analysis_frames = getattr(args, "analysis_frames", None)
    if analysis_frames is not None and int(analysis_frames) > 0:
        selected = deque(maxlen=int(analysis_frames))
        for item in iter_partition_molecule_centers(dump_path, metadata, equilibration_steps):
            selected.append(item)
        return list(selected)
    all_frames = list(iter_partition_molecule_centers(dump_path, metadata, equilibration_steps))
    if not all_frames:
        return []
    keep_fraction = float(getattr(args, "keep_fraction", 0.5))
    start = max(0, int(len(all_frames) * (1.0 - keep_fraction)))
    return all_frames[start:]


def analyze_ummap(args):
    outdir = Path(args.outdir).resolve()
    with open(outdir / "logp_partition_metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    n_slabs = int(getattr(args, "slabs", 30) or 30)
    if n_slabs < 3:
        raise ValueError("UMMAP-style analysis requires at least 3 slabs")
    frames = choose_analysis_frames(outdir / "dump.lammpstrj", metadata, args)
    if not frames:
        raise ValueError("no post-equilibration frames available for UMMAP-style analysis")

    slab_edges = np.linspace(0.0, 1.0, n_slabs + 1)
    slab_centers = 0.5 * (slab_edges[:-1] + slab_edges[1:])
    profiles = {
        "water": np.zeros(n_slabs, dtype=float),
        "organic": np.zeros(n_slabs, dtype=float),
        "solute": np.zeros(n_slabs, dtype=float),
    }

    box_lengths = []
    for _, centers, lengths in frames:
        x_length, y_length, z_length = (float(item) for item in lengths)
        slab_volume = x_length * y_length * (z_length / n_slabs)
        if slab_volume <= 0.0:
            raise ValueError("UMMAP-style analysis failed: nonpositive slab volume")
        box_lengths.append([x_length, y_length, z_length])
        for group in profiles:
            fractional_centers = [(float(value) / z_length) % 1.0 for value in centers[group]]
            hist, _ = np.histogram(fractional_centers, bins=slab_edges)
            profiles[group] += hist.astype(float) / slab_volume
    for group in profiles:
        profiles[group] = profiles[group] / len(frames)

    solvent_difference = profiles["organic"] - profiles["water"]
    gradient = np.gradient(solvent_difference, slab_centers)
    dmm = float(np.max(gradient) - np.min(gradient))
    if dmm <= 0.0 or not np.isfinite(dmm):
        raise ValueError("UMMAP-style interface detection failed: no solvent-gradient signal")
    threshold = float(getattr(args, "interface_gradient_fraction", 0.3) or 0.3) * dmm
    interface_indices = sorted(int(idx) for idx in np.where(np.abs(gradient) >= threshold)[0])
    interface_indices = sorted(set(interface_indices + [int(np.argmax(gradient)), int(np.argmin(gradient))]))
    interface_groups = circular_interface_groups(interface_indices, n_slabs)
    if not interface_groups or len(interface_groups) > 2:
        raise ValueError(
            "UMMAP-style interface detection failed: expected one or two interface regions, "
            f"found {len(interface_groups)}"
        )

    interface_mask = np.zeros(n_slabs, dtype=bool)
    pad = int(getattr(args, "interface_slab_padding", 1) or 0)
    for idx in interface_indices:
        for offset in range(-pad, pad + 1):
            interface_mask[(idx + offset) % n_slabs] = True
    bulk_mask = ~interface_mask
    water_bulk = bulk_mask & (profiles["water"] > profiles["organic"])
    organic_bulk = bulk_mask & (profiles["organic"] > profiles["water"])
    if not np.any(water_bulk) or not np.any(organic_bulk):
        raise ValueError("UMMAP-style interface detection failed: missing water or organic bulk slabs")
    water_dominant = profiles["water"] > profiles["organic"]
    organic_dominant = profiles["organic"] > profiles["water"]
    water_dominant_count = int(np.count_nonzero(water_dominant))
    organic_dominant_count = int(np.count_nonzero(organic_dominant))
    water_organic_dominant_ratio = (
        float(water_dominant_count / organic_dominant_count)
        if organic_dominant_count > 0
        else None
    )

    c_water = float(np.mean(profiles["solute"][water_bulk]))
    c_organic = float(np.mean(profiles["solute"][organic_bulk]))
    if c_water <= 0.0 and c_organic <= 0.0:
        raise ValueError("UMMAP-style logP failed: no solute in either bulk phase")
    if c_water <= 0.0:
        logp = 10.0
    elif c_organic <= 0.0:
        logp = -10.0
    else:
        logp = math.log10(c_organic / c_water)
        if logp > 10.0:
            logp = 10.0
        elif logp < -10.0:
            logp = -10.0

    with open(outdir / "partition_slab_profiles.csv", "w", encoding="utf-8") as handle:
        handle.write("slab,z_center,water_concentration,organic_concentration,solute_concentration,gradient,is_interface,bulk_label\n")
        for idx, z_center in enumerate(slab_centers):
            label = "interface"
            if water_bulk[idx]:
                label = "water"
            elif organic_bulk[idx]:
                label = "organic"
            handle.write(
                f"{idx},{z_center:.8f},{profiles['water'][idx]:.12g},"
                f"{profiles['organic'][idx]:.12g},{profiles['solute'][idx]:.12g},"
                f"{gradient[idx]:.12g},{int(interface_mask[idx])},{label}\n"
            )

    summary = {
        "analysis_method": "ummap",
        "frames_used": len(frames),
        "first_timestep": int(frames[0][0]),
        "last_timestep": int(frames[-1][0]),
        "equilibration_steps": int(getattr(args, "equilibration_steps", 0) or 0),
        "slabs": n_slabs,
        "mean_box": [float(item) for item in np.mean(np.asarray(box_lengths, dtype=float), axis=0)],
        "interface_gradient_fraction": float(getattr(args, "interface_gradient_fraction", 0.3) or 0.3),
        "interface_slab_padding": pad,
        "interface_groups": interface_groups,
        "water_bulk_slabs": [int(idx) for idx in np.where(water_bulk)[0]],
        "organic_bulk_slabs": [int(idx) for idx in np.where(organic_bulk)[0]],
        "water_dominant_slabs": [int(idx) for idx in np.where(water_dominant)[0]],
        "organic_dominant_slabs": [int(idx) for idx in np.where(organic_dominant)[0]],
        "water_dominant_fraction": float(water_dominant_count / n_slabs),
        "organic_dominant_fraction": float(organic_dominant_count / n_slabs),
        "water_to_organic_dominant_ratio": water_organic_dominant_ratio,
        "C_water": c_water,
        "C_organic": c_organic,
        "C_octanol": c_organic,
        "C_organic_over_C_water": c_organic / c_water if c_water > 0 else None,
        "C_octanol_over_C_water": c_organic / c_water if c_water > 0 else None,
        "log10_partition_ratio": logp,
    }
    with open(outdir / "partition_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def analyze(args):
    if getattr(args, "analysis_method", "fixed") == "ummap":
        return analyze_ummap(args)
    outdir = Path(args.outdir).resolve()
    with open(outdir / "logp_partition_metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    box = metadata["box"]
    interface_z = metadata["interface_z"]
    width = args.interface_width if args.interface_width is not None else metadata["interface_width"]
    water_zmin = width / 2.0
    water_zmax = interface_z - width / 2.0
    octanol_zmin = interface_z + width / 2.0
    octanol_zmax = box[2] - width / 2.0
    solute_ids = set(metadata["solute_mol_ids"])
    all_counts = read_partition_counts(outdir / "dump.lammpstrj", metadata, water_zmin, water_zmax, octanol_zmin, octanol_zmax)
    start = max(0, int(len(all_counts) * (1.0 - args.keep_fraction)))
    frames = all_counts[start:]
    water_counts = [water for _, water, _, _ in frames]
    octanol_counts = [octanol for _, _, octanol, _ in frames]
    ignored_counts = [ignored for _, _, _, ignored in frames]

    water_volume = box[0] * box[1] * (water_zmax - water_zmin)
    octanol_volume = box[0] * box[1] * (octanol_zmax - octanol_zmin)
    pseudo = args.pseudocount
    c_water = (float(np.mean(water_counts)) + pseudo) / water_volume
    c_octanol = (float(np.mean(octanol_counts)) + pseudo) / octanol_volume
    ratio = c_octanol / c_water
    summary = {
        "frames_used": len(frames),
        "keep_fraction": args.keep_fraction,
        "mean_solute_in_water": float(np.mean(water_counts)),
        "mean_solute_in_octanol": float(np.mean(octanol_counts)),
        "mean_solute_in_interface": float(np.mean(ignored_counts)),
        "C_water": c_water,
        "C_octanol": c_octanol,
        "C_octanol_over_C_water": ratio,
        "log10_partition_ratio": math.log10(ratio) if ratio > 0 else None,
        "pseudocount": pseudo,
        "water_bulk_zmin": water_zmin,
        "water_bulk_zmax": water_zmax,
        "octanol_bulk_zmin": octanol_zmin,
        "octanol_bulk_zmax": octanol_zmax,
    }
    with open(outdir / "partition_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(outdir / "partition_timeseries.csv", "w", encoding="utf-8") as handle:
        handle.write("frame_index,water,octanol,interface\n")
        for idx, (_, water, octanol, ignored) in enumerate(frames):
            handle.write(f"{idx},{water},{octanol},{ignored}\n")
    print(json.dumps(summary, indent=2))
    return summary


def analyze_density(args):
    outdir = Path(args.outdir).resolve()
    with open(outdir / "pure_density_metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    frames = read_dump_box_volumes(outdir / "dump.lammpstrj", getattr(args, "keep_fraction", 0.5))
    volumes = np.asarray([volume for _, volume in frames], dtype=float)
    mean_volume = float(np.mean(volumes))
    bead_density = float(metadata["atoms"] / mean_volume)
    molecule_density = float(metadata["molecules"] / mean_volume)
    summary = {
        "frames_used": len(frames),
        "keep_fraction": getattr(args, "keep_fraction", 0.5),
        "mean_volume": mean_volume,
        "std_volume": float(np.std(volumes)),
        "bead_number_density": bead_density,
        "molecule_number_density": molecule_density,
        "pressure": metadata.get("pressure"),
        "molecules": metadata["molecules"],
        "atoms": metadata["atoms"],
    }
    with open(outdir / "density_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def run_lammps(args):
    outdir = build_system(args)
    lammps = shutil.which(args.lammps)
    if not lammps:
        raise FileNotFoundError(f"LAMMPS executable not found: {args.lammps}")
    cmd = ["mpirun", "-np", str(args.job), lammps]
    if args.gpu:
        cmd.extend(["-sf", "gpu", "-pk", "gpu", "1"])
    cmd.extend(["-i", "logp_partition.in"])
    with open(outdir / "run_command.txt", "w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n")
    subprocess.run(cmd, cwd=outdir, check=True)
    return analyze(args)


def auto_fit_pairs(solute_beads):
    pairs = set()
    for bead in solute_beads:
        if bead != "H2O":
            pairs.add(pair_key(bead, "H2O"))
    return sorted(pairs)


def parse_fit_pairs(value, solute_beads):
    if value == "auto":
        return auto_fit_pairs(solute_beads)
    pairs = []
    for item in value.split(","):
        if not item.strip():
            continue
        bead_i, bead_j = item.split(":")
        pairs.append(pair_key(bead_i.strip(), bead_j.strip()))
    return sorted(set(pairs))


def write_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_fit_aij_text(text):
    overrides = {}
    for item in str(text or "").split(";"):
        item = item.strip()
        if not item or "=" not in item or ":" not in item:
            continue
        pair_text, value = item.rsplit("=", 1)
        bead_i, bead_j = pair_text.split(":", 1)
        overrides[pair_key(bead_i.strip(), bead_j.strip())] = float(value)
    return overrides


def candidate_from_history_row(row, fit_pairs):
    overrides = parse_fit_aij_text(row.get("fit_aij", ""))
    return np.asarray([overrides[key] for key in fit_pairs], dtype=float)


def restore_stage_history(stage_history, all_history, fit_pairs):
    if not stage_history:
        return None, [], []
    best = None
    observations = []
    losses = []
    for row in stage_history:
        row["evaluation"] = int(row["evaluation"])
        row["rmse"] = float(row["rmse"])
        row["max_abs_error"] = float(row["max_abs_error"])
        row["effective_tolerance"] = float(row.get("effective_tolerance", 0.0) or 0.0)
        overrides = parse_fit_aij_text(row.get("fit_aij", ""))
        if best is None or row["rmse"] < best["rmse"]:
            best = dict(row)
            best["overrides"] = dict(overrides)
        observations.append(tuple(float(item) for item in candidate_from_history_row(row, fit_pairs)))
        losses.append(float(row["rmse"]) ** 2)
    all_history.extend(stage_history)
    return best, observations, losses


def fitted_table_rows(type_names, pair_rows):
    rows = []
    for i, j, aij, rij in pair_rows:
        rows.append(
            {
                "beadi": type_names[i - 1],
                "beadj": type_names[j - 1],
                "A_ij": aij,
                "R_ij": rij,
                "r0": "fit",
            }
        )
    return rows


def merged_fitted_table_rows(table, overrides):
    rows = []
    seen = set()
    for _, row in table.iterrows():
        bead_i = str(row["beadi"]).strip()
        bead_j = str(row["beadj"]).strip()
        key = pair_key(bead_i, bead_j)
        fitted = key in overrides
        rows.append(
            {
                "beadi": bead_i,
                "beadj": bead_j,
                "A_ij": overrides[key] if fitted else row["A_ij"],
                "R_ij": row.get("R_ij", np.nan),
                "r0": "fit" if fitted else row.get("r0", np.nan),
            }
        )
        seen.add(key)
    for key, value in sorted(overrides.items()):
        if key in seen:
            continue
        rows.append({"beadi": key[0], "beadj": key[1], "A_ij": value, "R_ij": np.nan, "r0": "fit"})
    return rows


def serialize_overrides(overrides):
    return {f"{key[0]}:{key[1]}": value for key, value in overrides.items()}


def parse_manual_candidate(value, fit_pairs, current):
    value = value.strip()
    if not value:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) == len(fit_pairs) and all("=" not in part for part in parts):
        try:
            return np.asarray([float(part) for part in parts], dtype=float)
        except ValueError as exc:
            raise ValueError("manual Aij values must be numeric") from exc

    candidate = np.array(current, copy=True, dtype=float)
    by_key = {pair: idx for idx, pair in enumerate(fit_pairs)}
    for part in parts:
        if "=" not in part or ":" not in part:
            raise ValueError(
                "manual input must be comma-separated values or BEAD1:BEAD2=Aij entries"
            )
        pair_part, value_part = part.split("=", 1)
        bead_i, bead_j = pair_part.split(":", 1)
        key = pair_key(bead_i.strip(), bead_j.strip())
        if key not in by_key:
            allowed = ", ".join(f"{a}:{b}" for a, b in fit_pairs)
            raise ValueError(f"manual pair {key[0]}:{key[1]} is not fitted; allowed pairs: {allowed}")
        candidate[by_key[key]] = float(value_part)
    return candidate


def parse_bo_grid_specs(values):
    global_spec = None
    pair_specs = {}
    for value in values or []:
        if "=" in value:
            pair_part, spec_part = value.split("=", 1)
            bead_i, bead_j = pair_part.split(":", 1)
            key = pair_key(bead_i.strip(), bead_j.strip())
        else:
            key = None
            spec_part = value
        pieces = [float(item.strip()) for item in spec_part.split(":") if item.strip()]
        if len(pieces) != 3:
            raise ValueError(
                "Bayesian grid specs must look like LOW:HIGH:STEP or BEAD1:BEAD2=LOW:HIGH:STEP"
            )
        low, high, step = pieces
        if step <= 0 or high < low:
            raise ValueError(f"invalid Bayesian grid spec: {value}")
        if key is None:
            global_spec = (low, high, step)
        else:
            pair_specs[key] = (low, high, step)
    return global_spec, pair_specs


def grid_values_from_spec(low, high, step):
    count = int(math.floor((high - low) / step + 1e-9)) + 1
    values = low + step * np.arange(count, dtype=float)
    if len(values) == 0 or values[-1] < high - 1e-9:
        values = np.append(values, high)
    return np.asarray([round(float(item), 10) for item in values], dtype=float)


def build_bayesian_grid_axes(fit_pairs, args):
    global_spec, pair_specs = parse_bo_grid_specs(getattr(args, "bo_grid", []))
    if global_spec is None:
        global_spec = (args.min_aij, args.max_aij, args.bo_grid_step)
    axes = []
    for key in fit_pairs:
        low, high, step = pair_specs.get(key, global_spec)
        low = clamp_aij(low, args.min_aij, args.max_aij)
        high = clamp_aij(high, args.min_aij, args.max_aij)
        axes.append(grid_values_from_spec(low, high, step))
    return axes


def bayesian_grid_size(axes):
    size = 1
    for axis in axes:
        size *= len(axis)
    return int(size)


def sample_grid_points(axes, n_points, rng, excluded=None):
    if n_points <= 0:
        return []
    excluded = excluded or set()
    total = bayesian_grid_size(axes)
    available = total - len(excluded)
    if available <= 0:
        return []
    n_points = min(int(n_points), available)
    points = []
    seen = set(excluded)
    if total <= argsort_limit():
        candidates = enumerate_grid_points(axes, excluded=excluded)
        order = rng.permutation(len(candidates))
        return [candidates[int(idx)] for idx in order[:n_points]]
    max_attempts = max(1000, n_points * 200)
    while len(points) < n_points and len(seen) < total and max_attempts > 0:
        point = tuple(float(axis[int(rng.integers(0, len(axis)))]) for axis in axes)
        max_attempts -= 1
        if point in seen:
            continue
        seen.add(point)
        points.append(point)
    return points


def argsort_limit():
    return 200000


def enumerate_grid_points(axes, excluded=None, limit=None):
    excluded = excluded or set()
    if limit is not None and limit <= 0:
        return []
    points = []
    for point in itertools.product(*axes):
        key = tuple(float(item) for item in point)
        if key in excluded:
            continue
        points.append(key)
        if limit and len(points) >= limit:
            break
    return points


def iter_grid_point_chunks(axes, excluded=None, chunk_size=100000):
    excluded = excluded or set()
    numeric_axes = []
    for idx, axis in enumerate(axes, start=1):
        if isinstance(axis, (str, bytes)):
            raise TypeError(
                f"Bayesian grid axis {idx} is a string ({axis!r}); "
                "expected numeric Aij values. Check --bo-grid arguments."
            )
        numeric_axes.append(np.asarray(axis, dtype=float))
    chunk = []
    for point in itertools.product(*numeric_axes):
        key = tuple(float(item) for item in point)
        if key in excluded:
            continue
        chunk.append(key)
        if len(chunk) >= chunk_size:
            yield np.asarray(chunk, dtype=float)
            chunk = []
    if chunk:
        yield np.asarray(chunk, dtype=float)


def expected_improvement(mu, sigma, best_loss, epsilon):
    sigma = np.maximum(sigma, 1e-12)
    improvement = best_loss - mu - epsilon
    z = improvement / sigma
    return improvement * normal_cdf(z) + sigma * normal_pdf(z)


def choose_bayesian_candidate(axes, observations, losses, rng, args):
    if not observations:
        candidates = sample_grid_points(axes, 1, rng)
        if not candidates:
            return None
        return np.asarray(candidates[0], dtype=float)

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    x_train = np.asarray(observations, dtype=float)
    y_train = np.asarray(losses, dtype=float)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(x_train.shape[1])) + WhiteKernel(
        noise_level=args.bo_noise,
        noise_level_bounds=(1e-10, 1e1),
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=args.bo_gp_restarts,
        random_state=args.seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(x_train, y_train)
    evaluated = {tuple(float(v) for v in row) for row in observations}
    best_candidate = None
    best_ei = -float("inf")
    if getattr(args, "bo_full_grid_ei", True):
        for x_candidates in iter_grid_point_chunks(
            axes,
            excluded=evaluated,
            chunk_size=int(getattr(args, "bo_grid_chunk_size", 100000) or 100000),
        ):
            mu, sigma = model.predict(x_candidates, return_std=True)
            ei = expected_improvement(mu, sigma, float(np.min(y_train)), args.bo_epsilon)
            idx = int(np.argmax(ei))
            if float(ei[idx]) > best_ei:
                best_ei = float(ei[idx])
                best_candidate = x_candidates[idx]
        return None if best_candidate is None else np.asarray(best_candidate, dtype=float)

    candidates = sample_grid_points(axes, args.bo_acq_candidates, rng, excluded=evaluated)
    if not candidates:
        return None
    x_candidates = np.asarray(candidates, dtype=float)
    mu, sigma = model.predict(x_candidates, return_std=True)
    ei = expected_improvement(mu, sigma, float(np.min(y_train)), args.bo_epsilon)
    return np.asarray(candidates[int(np.argmax(ei))], dtype=float)


def penalty_logp_for_target(target):
    target = float(target)
    return -10.0 if target >= 0 else 10.0


def initial_fit_aij(key, table, bead_solubility, initial_aij):
    value = solubility_pair_aij(key[0], key[1], bead_solubility)
    if value is not None:
        return float(value)
    existing = lookup_pair(table, key[0], key[1])
    if existing is not None:
        return float(existing["A_ij"])
    return float(initial_aij)


def history_columns():
    return [
        "iteration",
        "optimizer",
        "status",
        "failure_reason",
        "target_logp",
        "target_std",
        "effective_tolerance",
        "observed_logp",
        "error",
        "abs_error",
        "objective",
        "fit_pairs",
        "fit_aij",
        "run_dir",
    ]


def run_fit(args):
    if args.max_iter < 1:
        raise ValueError("--max-iter must be at least 1")
    solute = parse_recipe(args.solute, allow_custom=args.allow_custom_beads)
    fit_pairs = parse_fit_pairs(args.fit_pairs, solute.beads)
    if not fit_pairs:
        raise ValueError("no fit pairs selected")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    target_report = None
    if args.target_logp is None:
        if not args.target_smiles:
            raise ValueError("provide --target-logp or --target-smiles")
        rows = collect_logp_values(
            args.target_smiles,
            external_values=args.target_logp_value,
            use_pubchem=not args.no_pubchem,
            use_openbabel=not args.no_openbabel,
        )
        target_report = robust_consensus(
            rows,
            outlier_method=args.target_outlier_method,
            min_methods=args.target_min_methods,
        )
        args.target_logp = target_report["target_logp"]
        write_logp_report(outdir / "target_logp_report.json", target_report)
    target_std = target_report["target_std"] if target_report else None
    if target_std is None:
        effective_tolerance = min(args.tolerance, args.max_abs_error)
    else:
        effective_tolerance = min(args.max_abs_error, args.target_std_factor * target_std)

    table = load_logp_table(args.table)
    overrides = parse_pair_overrides(args.override)
    bead_solubility = parse_bead_solubility(args.bead_solubility)
    args.bead_solubility_values = bead_solubility
    for key in fit_pairs:
        if key not in overrides:
            value = initial_fit_aij(key, table, bead_solubility, args.initial_aij)
            overrides[key] = clamp_aij(value, args.min_aij, args.max_aij)

    history = []
    best = None
    lammps = shutil.which(args.lammps)
    if not lammps:
        raise FileNotFoundError(f"LAMMPS executable not found: {args.lammps}")

    progress = None

    def evaluate(candidate):
        nonlocal best
        iteration = len(history) + 1
        candidate = np.asarray(candidate, dtype=float)
        candidate = np.clip(candidate, args.min_aij, args.max_aij)
        candidate_overrides = dict(overrides)
        for idx, key in enumerate(fit_pairs):
            candidate_overrides[key] = float(candidate[idx])

        iter_dir = outdir / f"iter_{iteration:03d}"
        iter_args = argparse.Namespace(**vars(args))
        iter_args.outdir = str(iter_dir)
        iter_args.pair_overrides = candidate_overrides
        status = "ok"
        failure_reason = ""
        try:
            build_system(iter_args)
            cmd = ["mpirun", "-np", str(args.job), lammps]
            if args.gpu:
                cmd.extend(["-sf", "gpu", "-pk", "gpu", "1"])
            cmd.extend(["-i", "logp_partition.in"])
            with open(iter_dir / "run_command.txt", "w", encoding="utf-8") as handle:
                handle.write(" ".join(cmd) + "\n")
            subprocess.run(cmd, cwd=iter_dir, check=True)
            summary = analyze(iter_args)
            observed = float(summary["log10_partition_ratio"])
            error = args.target_logp - observed
            objective = error * error
        except Exception as exc:
            if not args.penalize_failures:
                raise
            status = "failed"
            failure_reason = repr(exc).replace(",", ";")
            observed = penalty_logp_for_target(args.target_logp)
            error = args.target_logp - observed
            objective = float(args.failure_penalty_loss)
        row = {
            "iteration": iteration,
            "optimizer": args.optimizer,
            "status": status,
            "failure_reason": failure_reason,
            "target_logp": args.target_logp,
            "target_std": target_std,
            "effective_tolerance": effective_tolerance,
            "observed_logp": observed,
            "error": error,
            "abs_error": abs(error),
            "objective": objective,
            "fit_pairs": ";".join(f"{a}:{b}" for a, b in fit_pairs),
            "fit_aij": ";".join(f"{a}:{b}={candidate_overrides[pair_key(a, b)]:.6f}" for a, b in fit_pairs),
            "run_dir": str(iter_dir),
        }
        history.append(row)
        if best is None or abs(error) < best["abs_error"]:
            best = dict(row)
            best["overrides"] = dict(candidate_overrides)
        write_csv(outdir / "fit_history.csv", history, history_columns())
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                {
                    "abs_err": f"{abs(error):.4g}",
                    "best": f"{best['abs_error']:.4g}",
                }
            )
        return objective

    x0 = np.asarray([overrides[key] for key in fit_pairs], dtype=float)
    with progress_bar(total=args.max_iter, desc="logP fit", unit="eval", disable=args.no_progress) as progress_handle:
        progress = progress_handle
        if args.optimizer == "manual":
            current = np.array(x0, copy=True)
            while len(history) < args.max_iter:
                evaluate(current)
                if best and best["abs_error"] <= effective_tolerance:
                    break
                print(
                    "manual logP fit iteration "
                    f"{len(history)}/{args.max_iter}: target={args.target_logp:.6g}, "
                    f"observed={history[-1]['observed_logp']:.6g}, "
                    f"error={history[-1]['error']:.6g}, "
                    f"best_abs_error={best['abs_error']:.6g}"
                )
                print(
                    "current Aij: "
                    + ", ".join(f"{a}:{b}={current[idx]:.6g}" for idx, (a, b) in enumerate(fit_pairs))
                )
                print(
                    "enter next Aij as comma values or BEAD1:BEAD2=Aij entries; "
                    "blank keeps the current best and stops:"
                )
                try:
                    user_value = input("> ")
                except EOFError:
                    break
                next_candidate = parse_manual_candidate(user_value, fit_pairs, current)
                if next_candidate is None:
                    break
                current = np.asarray(
                    [clamp_aij(value, args.min_aij, args.max_aij) for value in next_candidate],
                    dtype=float,
                )
        elif args.optimizer == "nelder-mead":
            from scipy.optimize import minimize

            simplex = [x0]
            for idx in range(len(x0)):
                point = np.array(x0, copy=True)
                direction = 1.0
                if point[idx] + args.simplex_step > args.max_aij:
                    direction = -1.0
                point[idx] = clamp_aij(point[idx] + direction * args.simplex_step, args.min_aij, args.max_aij)
                simplex.append(point)
            minimize(
                evaluate,
                x0,
                method="Nelder-Mead",
                options={
                    "maxfev": args.max_iter,
                    "maxiter": args.max_iter,
                    "initial_simplex": np.asarray(simplex),
                    "xatol": args.xatol,
                    "fatol": args.fatol,
                    "disp": False,
                },
            )
        elif args.optimizer == "coordinate":
            # Fallback derivative-free coordinate search. It is not a linear error update.
            current = np.array(x0, copy=True)
            step = args.simplex_step
            while len(history) < args.max_iter:
                evaluate(current)
                if best and best["abs_error"] <= effective_tolerance:
                    break
                dim = (len(history) - 1) % len(current)
                trial_plus = np.array(current, copy=True)
                trial_plus[dim] = clamp_aij(trial_plus[dim] + step, args.min_aij, args.max_aij)
                if len(history) < args.max_iter:
                    evaluate(trial_plus)
                trial_minus = np.array(current, copy=True)
                trial_minus[dim] = clamp_aij(trial_minus[dim] - step, args.min_aij, args.max_aij)
                if len(history) < args.max_iter:
                    evaluate(trial_minus)
                current = np.asarray([best["overrides"][key] for key in fit_pairs], dtype=float)
                step *= 0.5
                if best and best["abs_error"] <= effective_tolerance:
                    break
        elif args.optimizer == "bayesian":
            rng = np.random.default_rng(args.seed)
            axes = build_bayesian_grid_axes(fit_pairs, args)
            observations = []
            losses = []
            initial_points = sample_grid_points(axes, args.bo_initial, rng)
            for point in initial_points:
                if len(history) >= args.max_iter:
                    break
                objective = evaluate(point)
                observations.append(tuple(float(item) for item in point))
                losses.append(float(objective))
                if best and best["abs_error"] <= effective_tolerance:
                    break
            while len(history) < args.max_iter and not (best and best["abs_error"] <= effective_tolerance):
                candidate = choose_bayesian_candidate(axes, observations, losses, rng, args)
                if candidate is None:
                    break
                objective = evaluate(candidate)
                observations.append(tuple(float(item) for item in candidate))
                losses.append(float(objective))
        else:
            raise ValueError(f"unsupported optimizer: {args.optimizer}")
        progress = None

    final_args = argparse.Namespace(**vars(args))
    final_args.outdir = str(outdir / "best_table_build")
    final_args.pair_overrides = best["overrides"]
    build_system(final_args)
    best_output = dict(best)
    best_output["overrides"] = serialize_overrides(best["overrides"])
    with open(outdir / "best_fit.json", "w", encoding="utf-8") as handle:
        json.dump(best_output, handle, indent=2)
    rows = merged_fitted_table_rows(table, best["overrides"])
    write_csv(outdir / "fitted_interactions.csv", rows, ["beadi", "beadj", "A_ij", "R_ij", "r0"])
    print(json.dumps(best_output, indent=2))
    return best


def parse_config_overrides(mapping):
    overrides = {}
    for pair, value in (mapping or {}).items():
        bead_i, bead_j = pair.split(":")
        overrides[pair_key(bead_i.strip(), bead_j.strip())] = float(value)
    return overrides


def staged_history_columns():
    return [
        "stage",
        "evaluation",
        "optimizer",
        "status",
        "failure_reason",
        "rmse",
        "max_abs_error",
        "effective_tolerance",
        "fit_pairs",
        "fit_aij",
        "target_errors",
        "run_dir",
    ]


def target_logp_from_spec(target, stage_dir, args):
    if "target_logp" in target:
        return float(target["target_logp"]), None
    if "target_smiles" not in target:
        raise ValueError(f"target {target.get('name', target)} needs target_logp or target_smiles")
    rows = collect_logp_values(
        target["target_smiles"],
        external_values=target.get("target_logp_value", []),
        use_pubchem=not target.get("no_pubchem", args.no_pubchem),
        use_openbabel=not target.get("no_openbabel", args.no_openbabel),
    )
    report = robust_consensus(
        rows,
        outlier_method=target.get("target_outlier_method", args.target_outlier_method),
        min_methods=target.get("target_min_methods", args.target_min_methods),
    )
    report_path = stage_dir / f"target_{target.get('name', target['solute'])}_logp_report.json"
    write_logp_report(report_path, report)
    return float(report["target_logp"]), report


def namespace_for_stage_target(args, stage, target, run_dir, overrides, bead_solubility):
    values = vars(args).copy()
    for key in [
        "n_solute",
        "water_count",
        "organic_count",
        "box",
        "organic_solvent",
        "density",
        "water_density",
        "octanol_density",
        "organic_density",
        "solute_bead_fraction",
        "steps",
        "dump_every",
        "timestep",
        "ensemble",
        "pressure",
        "pressure_damp",
        "solute_init",
        "interface_width",
        "temperature",
        "sigma",
        "seed",
        "default_missing_aij",
        "min_aij",
        "max_aij",
        "allow_custom_beads",
        "heavy_atom_correction",
        "heavy_radius_scale",
        "bonded_r0_factor",
        "angle_param_mode",
        "keep_fraction",
        "pseudocount",
        "analysis_method",
        "equilibration_steps",
        "analysis_frames",
        "slabs",
        "interface_gradient_fraction",
        "interface_slab_padding",
        "charge_method",
        "bead_charge",
        "charge_unit_scale",
        "charge_lambda",
        "coul_cutoff",
        "kspace_accuracy",
    ]:
        if key in stage:
            values[key] = stage[key]
        if key in target:
            values[key] = target[key]
    values["solute"] = target["solute"]
    values["outdir"] = str(run_dir)
    values["pair_overrides"] = overrides
    values["bead_solubility_values"] = bead_solubility
    return argparse.Namespace(**values)


def namespace_for_density_target(args, stage, target, run_dir, overrides, bead_solubility):
    values = vars(args).copy()
    key_aliases = {
        "n_molecules": ["n_molecules", "density_n_molecules"],
        "initial_density": ["initial_density"],
        "steps": ["steps", "density_steps"],
        "dump_every": ["dump_every", "density_dump_every"],
        "pressure": ["pressure"],
        "temperature": ["temperature"],
        "sigma": ["sigma"],
        "seed": ["seed"],
        "default_missing_aij": ["default_missing_aij"],
        "min_aij": ["min_aij"],
        "max_aij": ["max_aij"],
        "allow_custom_beads": ["allow_custom_beads"],
        "heavy_atom_correction": ["heavy_atom_correction"],
        "heavy_radius_scale": ["heavy_radius_scale"],
        "bonded_r0_factor": ["bonded_r0_factor"],
        "angle_param_mode": ["angle_param_mode"],
        "keep_fraction": ["keep_fraction"],
        "density_mixing_rule": ["density_mixing_rule"],
        "charge_method": ["charge_method"],
        "bead_charge": ["bead_charge"],
        "charge_unit_scale": ["charge_unit_scale"],
        "charge_lambda": ["charge_lambda"],
        "coul_cutoff": ["coul_cutoff"],
        "kspace_accuracy": ["kspace_accuracy"],
    }
    for output_key, aliases in key_aliases.items():
        for alias in aliases:
            if alias in stage:
                values[output_key] = stage[alias]
            if alias in target:
                values[output_key] = target[alias]
    values["solute"] = target.get("molecule", target.get("solute"))
    values["outdir"] = str(run_dir)
    values["pair_overrides"] = overrides
    values["bead_solubility_values"] = bead_solubility
    return argparse.Namespace(**values)


def lammps_run_completed(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        return False
    try:
        with open(log_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 8192))
            tail = handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return False
    return "Total wall time:" in tail


def run_stage_target(args, stage, target, run_dir, overrides, bead_solubility, lammps):
    run_args = namespace_for_stage_target(args, stage, target, run_dir, overrides, bead_solubility)
    summary_path = Path(run_dir) / "partition_summary.json"
    dump_path = Path(run_dir) / "dump.lammpstrj"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if dump_path.exists() and lammps_run_completed(Path(run_dir) / "log.lammps"):
        return analyze(run_args)
    build_system(run_args)
    cmd = ["mpirun", "-np", str(args.job), lammps]
    if args.gpu:
        cmd.extend(["-sf", "gpu", "-pk", "gpu", "1"])
    cmd.extend(["-i", "logp_partition.in"])
    with open(Path(run_dir) / "run_command.txt", "w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n")
    subprocess.run(cmd, cwd=run_dir, check=True)
    return analyze(run_args)


def run_density_target(args, stage, target, run_dir, overrides, bead_solubility, lammps):
    run_args = namespace_for_density_target(args, stage, target, run_dir, overrides, bead_solubility)
    summary_path = Path(run_dir) / "density_summary.json"
    dump_path = Path(run_dir) / "dump.lammpstrj"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if dump_path.exists() and lammps_run_completed(Path(run_dir) / "log.lammps"):
        return analyze_density(run_args)
    build_density_system(run_args)
    cmd = ["mpirun", "-np", str(args.job), lammps, "-i", "pure_density.in"]
    with open(Path(run_dir) / "run_command.txt", "w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n")
    subprocess.run(cmd, cwd=run_dir, check=True)
    return analyze_density(run_args)


def stage_effective_tolerance(args, stage, target_reports):
    std_values = [
        float(report["target_std"])
        for report in target_reports
        if report is not None and report.get("target_std") is not None
    ]
    if std_values:
        return min(args.max_abs_error, args.target_std_factor * float(np.mean(std_values)))
    return min(args.tolerance, args.max_abs_error)


def write_staged_outputs(outdir, args, table, global_overrides, first_target, first_stage, bead_solubility):
    if first_target is not None and first_stage is not None:
        final_args = namespace_for_stage_target(
            args,
            first_stage,
            first_target,
            outdir / "best_table_build",
            global_overrides,
            bead_solubility,
        )
    else:
        final_args = argparse.Namespace(**vars(args))
        final_args.solute = getattr(args, "solute", "")
        final_args.outdir = str(outdir / "best_table_build")
        final_args.pair_overrides = global_overrides
        final_args.bead_solubility_values = bead_solubility
    build_system(final_args)
    rows = merged_fitted_table_rows(table, global_overrides)
    write_csv(outdir / "staged_fitted_interactions.csv", rows, ["beadi", "beadj", "A_ij", "R_ij", "r0"])


def run_fit_staged(args):
    with open(args.stage_config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "stage_config_used.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    table = load_logp_table(args.table)
    bead_solubility = parse_bead_solubility(args.bead_solubility)
    for bead, value in config.get("bead_solubility", {}).items():
        bead_solubility[bead] = float(value)
    args.bead_solubility_values = bead_solubility

    global_overrides = parse_pair_overrides(args.override)
    global_overrides.update(parse_config_overrides(config.get("fixed_overrides", {})))
    lammps = shutil.which(args.lammps)
    if not lammps:
        raise FileNotFoundError(f"LAMMPS executable not found: {args.lammps}")

    all_history = []
    stage_summaries = []
    first_target_for_output = None
    first_stage_for_output = None

    for stage_index, stage in enumerate(config.get("stages", []), start=1):
        stage_name = stage["name"]
        stage_dir = outdir / f"stage_{stage_index:02d}_{stage_name}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        if stage.get("type") == "set":
            global_overrides.update(parse_config_overrides(stage.get("set_pairs", {})))
            stage_summaries.append({"stage": stage_name, "type": "set", "overrides": serialize_overrides(global_overrides)})
            continue

        if stage.get("type") in {"density", "density_self"}:
            targets = stage.get("targets", [])
            if not targets:
                raise ValueError(f"density stage {stage_name} has no targets")
            fit_pairs = [pair_key(*item.split(":")) for item in stage.get("optimize_pairs", [])]
            if not fit_pairs:
                raise ValueError(f"density stage {stage_name} has no optimize_pairs")
            for key in fit_pairs:
                if key not in global_overrides:
                    value = initial_fit_aij(key, table, bead_solubility, args.initial_aij)
                    global_overrides[key] = clamp_aij(value, args.min_aij, args.max_aij)

            effective_tolerance = float(stage.get("density_tolerance", args.density_tolerance))
            stage_history = []
            best = None
            stage_progress = None
            restored_observations = []
            restored_losses = []
            checkpoint_rows = read_csv_rows(stage_dir / "stage_history.csv")
            if checkpoint_rows:
                best, restored_observations, restored_losses = restore_stage_history(
                    checkpoint_rows,
                    all_history,
                    fit_pairs,
                )
                stage_history.extend(checkpoint_rows)
                global_overrides.update(best["overrides"])

            def evaluate_density_stage(candidate):
                nonlocal best
                evaluation = len(stage_history) + 1
                candidate = np.asarray(candidate, dtype=float)
                candidate = np.clip(candidate, args.min_aij, args.max_aij)
                candidate_overrides = dict(global_overrides)
                for idx, key in enumerate(fit_pairs):
                    candidate_overrides[key] = float(candidate[idx])

                errors = []
                target_error_text = []
                status = "ok"
                failure_reasons = []
                eval_dir = stage_dir / f"eval_{evaluation:03d}"
                for target_index, target in enumerate(targets, start=1):
                    target_name = target.get("name", target.get("molecule", target.get("solute")))
                    run_dir = eval_dir / f"target_{target_index:02d}_{target_name}"
                    target_density = float(target["target_density"])
                    try:
                        summary = run_density_target(args, stage, target, run_dir, candidate_overrides, bead_solubility, lammps)
                        density_key = target.get("density_kind", "bead_number_density")
                        observed = float(summary[density_key])
                        scale = float(target.get("error_scale", target_density if target_density != 0 else 1.0))
                        error = (target_density - observed) / scale
                    except Exception as exc:
                        if not args.penalize_failures:
                            raise
                        status = "failed"
                        failure_reasons.append(f"{target_name}:{repr(exc).replace(',', ';')}")
                        observed = float("nan")
                        error = math.sqrt(float(args.failure_penalty_loss))
                    errors.append(error)
                    target_error_text.append(
                        f"{target_name}:target={target_density:.6f}:observed={observed:.6f}:error={error:.6f}"
                    )

                rmse = float(np.sqrt(np.mean(np.square(errors))))
                max_abs_error = float(np.max(np.abs(errors)))
                row = {
                    "stage": stage_name,
                    "evaluation": evaluation,
                    "optimizer": args.optimizer,
                    "status": status,
                    "failure_reason": "||".join(failure_reasons),
                    "rmse": rmse,
                    "max_abs_error": max_abs_error,
                    "effective_tolerance": effective_tolerance,
                    "fit_pairs": ";".join(f"{a}:{b}" for a, b in fit_pairs),
                    "fit_aij": ";".join(f"{a}:{b}={candidate_overrides[pair_key(a, b)]:.6f}" for a, b in fit_pairs),
                    "target_errors": "|".join(target_error_text),
                    "run_dir": str(eval_dir),
                }
                stage_history.append(row)
                all_history.append(row)
                if best is None or rmse < best["rmse"]:
                    best = dict(row)
                    best["overrides"] = dict(candidate_overrides)
                write_csv(stage_dir / "stage_history.csv", stage_history, staged_history_columns())
                write_csv(outdir / "staged_fit_history.csv", all_history, staged_history_columns())
                if stage_progress is not None:
                    stage_progress.update(1)
                    stage_progress.set_postfix(
                        {
                            "rmse": f"{rmse:.4g}",
                            "max": f"{max_abs_error:.4g}",
                        }
                    )
                return rmse * rmse

            x0 = np.asarray([global_overrides[key] for key in fit_pairs], dtype=float)
            max_iter = int(stage.get("max_iter", args.max_iter))
            if max_iter < 1:
                raise ValueError(f"stage {stage_name} max_iter must be at least 1")
            with progress_bar(
                total=max_iter,
                desc=f"stage {stage_index}:{stage_name}",
                unit="eval",
                disable=args.no_progress,
            ) as stage_progress_handle:
                stage_progress = stage_progress_handle
                if stage_history:
                    stage_progress.update(len(stage_history))
                    if best:
                        stage_progress.set_postfix(
                            {
                                "rmse": f"{best['rmse']:.4g}",
                                "max": f"{best['max_abs_error']:.4g}",
                            }
                        )
                if args.optimizer == "nelder-mead":
                    from scipy.optimize import minimize

                    simplex = [x0]
                    for idx in range(len(x0)):
                        point = np.array(x0, copy=True)
                        direction = -1.0 if point[idx] + args.simplex_step > args.max_aij else 1.0
                        point[idx] = clamp_aij(point[idx] + direction * args.simplex_step, args.min_aij, args.max_aij)
                        simplex.append(point)
                    minimize(
                        evaluate_density_stage,
                        x0,
                        method="Nelder-Mead",
                        options={
                            "maxfev": max_iter,
                            "maxiter": max_iter,
                            "initial_simplex": np.asarray(simplex),
                            "xatol": args.xatol,
                            "fatol": args.fatol,
                            "disp": False,
                        },
                    )
                elif args.optimizer == "coordinate":
                    current = np.array(x0, copy=True)
                    step = float(stage.get("simplex_step", args.simplex_step))
                    while len(stage_history) < max_iter:
                        evaluate_density_stage(current)
                        if best and best["max_abs_error"] <= effective_tolerance:
                            break
                        dim = (len(stage_history) - 1) % len(current)
                        current[dim] = clamp_aij(current[dim] + step, args.min_aij, args.max_aij)
                        step *= -0.5
                elif args.optimizer == "bayesian":
                    rng = np.random.default_rng(int(stage.get("seed", args.seed)) + stage_index)
                    axes = build_bayesian_grid_axes(fit_pairs, args)
                    observations = list(restored_observations)
                    losses = list(restored_losses)
                    for point in sample_grid_points(axes, args.bo_initial, rng):
                        if len(stage_history) >= max_iter:
                            break
                        if tuple(float(item) for item in point) in observations:
                            continue
                        objective = evaluate_density_stage(point)
                        observations.append(tuple(float(item) for item in point))
                        losses.append(float(objective))
                        if best and best["max_abs_error"] <= effective_tolerance:
                            break
                    while len(stage_history) < max_iter and not (best and best["max_abs_error"] <= effective_tolerance):
                        candidate = choose_bayesian_candidate(axes, observations, losses, rng, args)
                        if candidate is None:
                            break
                        objective = evaluate_density_stage(candidate)
                        observations.append(tuple(float(item) for item in candidate))
                        losses.append(float(objective))
                else:
                    raise ValueError(f"unsupported optimizer: {args.optimizer}")
                stage_progress = None

            global_overrides.update(best["overrides"])
            best_output = dict(best)
            best_output["type"] = "density"
            best_output["overrides"] = serialize_overrides(best["overrides"])
            with open(stage_dir / "stage_best.json", "w", encoding="utf-8") as handle:
                json.dump(best_output, handle, indent=2)
            stage_summaries.append(best_output)
            continue

        targets = stage.get("targets", [])
        if not targets:
            raise ValueError(f"stage {stage_name} has no targets")
        if first_target_for_output is None:
            first_target_for_output = dict(targets[0])
            first_stage_for_output = dict(stage)

        fit_pairs = [pair_key(*item.split(":")) for item in stage.get("optimize_pairs", [])]
        if not fit_pairs:
            raise ValueError(f"stage {stage_name} has no optimize_pairs")
        for key in fit_pairs:
            if key not in global_overrides:
                value = initial_fit_aij(key, table, bead_solubility, args.initial_aij)
                global_overrides[key] = clamp_aij(value, args.min_aij, args.max_aij)

        target_values = []
        target_reports = []
        for target in targets:
            value, report = target_logp_from_spec(target, stage_dir, args)
            target_values.append(value)
            target_reports.append(report)
        effective_tolerance = stage_effective_tolerance(args, stage, target_reports)

        stage_history = []
        best = None
        stage_progress = None
        restored_observations = []
        restored_losses = []
        checkpoint_rows = read_csv_rows(stage_dir / "stage_history.csv")
        if checkpoint_rows:
            best, restored_observations, restored_losses = restore_stage_history(
                checkpoint_rows,
                all_history,
                fit_pairs,
            )
            stage_history.extend(checkpoint_rows)
            global_overrides.update(best["overrides"])

        def evaluate_stage(candidate):
            nonlocal best
            evaluation = len(stage_history) + 1
            candidate = np.asarray(candidate, dtype=float)
            candidate = np.clip(candidate, args.min_aij, args.max_aij)
            candidate_overrides = dict(global_overrides)
            for idx, key in enumerate(fit_pairs):
                candidate_overrides[key] = float(candidate[idx])

            errors = []
            target_error_text = []
            status = "ok"
            failure_reasons = []
            eval_dir = stage_dir / f"eval_{evaluation:03d}"
            for target_index, target in enumerate(targets, start=1):
                target_name = target.get("name", target["solute"])
                run_dir = eval_dir / f"target_{target_index:02d}_{target_name}"
                try:
                    summary = run_stage_target(args, stage, target, run_dir, candidate_overrides, bead_solubility, lammps)
                    observed = float(summary["log10_partition_ratio"])
                except Exception as exc:
                    if not args.penalize_failures:
                        raise
                    status = "failed"
                    failure_reasons.append(f"{target_name}:{repr(exc).replace(',', ';')}")
                    observed = penalty_logp_for_target(target_values[target_index - 1])
                error = target_values[target_index - 1] - observed
                errors.append(error)
                target_error_text.append(
                    f"{target_name}:target={target_values[target_index - 1]:.6f}:observed={observed:.6f}:error={error:.6f}"
                )

            rmse = float(np.sqrt(np.mean(np.square(errors))))
            max_abs_error = float(np.max(np.abs(errors)))
            row = {
                "stage": stage_name,
                "evaluation": evaluation,
                "optimizer": args.optimizer,
                "status": status,
                "failure_reason": "||".join(failure_reasons),
                "rmse": rmse,
                "max_abs_error": max_abs_error,
                "effective_tolerance": effective_tolerance,
                "fit_pairs": ";".join(f"{a}:{b}" for a, b in fit_pairs),
                "fit_aij": ";".join(f"{a}:{b}={candidate_overrides[pair_key(a, b)]:.6f}" for a, b in fit_pairs),
                "target_errors": "|".join(target_error_text),
                "run_dir": str(eval_dir),
            }
            stage_history.append(row)
            all_history.append(row)
            if best is None or rmse < best["rmse"]:
                best = dict(row)
                best["overrides"] = dict(candidate_overrides)
            write_csv(stage_dir / "stage_history.csv", stage_history, staged_history_columns())
            write_csv(outdir / "staged_fit_history.csv", all_history, staged_history_columns())
            if stage_progress is not None:
                stage_progress.update(1)
                stage_progress.set_postfix(
                    {
                        "rmse": f"{rmse:.4g}",
                        "max": f"{max_abs_error:.4g}",
                    }
                )
            return rmse * rmse

        x0 = np.asarray([global_overrides[key] for key in fit_pairs], dtype=float)
        max_iter = int(stage.get("max_iter", args.max_iter))
        if max_iter < 1:
            raise ValueError(f"stage {stage_name} max_iter must be at least 1")
        with progress_bar(
            total=max_iter,
            desc=f"stage {stage_index}:{stage_name}",
            unit="eval",
            disable=args.no_progress,
        ) as stage_progress_handle:
            stage_progress = stage_progress_handle
            if stage_history:
                stage_progress.update(len(stage_history))
                if best:
                    stage_progress.set_postfix(
                        {
                            "rmse": f"{best['rmse']:.4g}",
                            "max": f"{best['max_abs_error']:.4g}",
                        }
                    )
            if args.optimizer == "nelder-mead":
                from scipy.optimize import minimize

                simplex = [x0]
                for idx in range(len(x0)):
                    point = np.array(x0, copy=True)
                    direction = -1.0 if point[idx] + args.simplex_step > args.max_aij else 1.0
                    point[idx] = clamp_aij(point[idx] + direction * args.simplex_step, args.min_aij, args.max_aij)
                    simplex.append(point)
                minimize(
                    evaluate_stage,
                    x0,
                    method="Nelder-Mead",
                    options={
                        "maxfev": max_iter,
                        "maxiter": max_iter,
                        "initial_simplex": np.asarray(simplex),
                        "xatol": args.xatol,
                        "fatol": args.fatol,
                        "disp": False,
                    },
                )
            elif args.optimizer == "coordinate":
                current = np.array(x0, copy=True)
                step = float(stage.get("simplex_step", args.simplex_step))
                while len(stage_history) < max_iter:
                    evaluate_stage(current)
                    if best and best["max_abs_error"] <= effective_tolerance:
                        break
                    dim = (len(stage_history) - 1) % len(current)
                    current[dim] = clamp_aij(current[dim] + step, args.min_aij, args.max_aij)
                    step *= -0.5
            elif args.optimizer == "bayesian":
                rng = np.random.default_rng(int(stage.get("seed", args.seed)) + stage_index)
                axes = build_bayesian_grid_axes(fit_pairs, args)
                observations = list(restored_observations)
                losses = list(restored_losses)
                for point in sample_grid_points(axes, args.bo_initial, rng):
                    if len(stage_history) >= max_iter:
                        break
                    if tuple(float(item) for item in point) in observations:
                        continue
                    objective = evaluate_stage(point)
                    observations.append(tuple(float(item) for item in point))
                    losses.append(float(objective))
                    if best and best["max_abs_error"] <= effective_tolerance:
                        break
                while len(stage_history) < max_iter and not (best and best["max_abs_error"] <= effective_tolerance):
                    candidate = choose_bayesian_candidate(axes, observations, losses, rng, args)
                    if candidate is None:
                        break
                    objective = evaluate_stage(candidate)
                    observations.append(tuple(float(item) for item in candidate))
                    losses.append(float(objective))
            else:
                raise ValueError(f"unsupported optimizer: {args.optimizer}")
            stage_progress = None

        global_overrides.update(best["overrides"])
        best_output = dict(best)
        best_output["overrides"] = serialize_overrides(best["overrides"])
        with open(stage_dir / "stage_best.json", "w", encoding="utf-8") as handle:
            json.dump(best_output, handle, indent=2)
        stage_summaries.append(best_output)

    with open(outdir / "staged_best.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"stages": stage_summaries, "final_overrides": serialize_overrides(global_overrides)},
            handle,
            indent=2,
        )
    write_staged_outputs(
        outdir,
        args,
        table,
        global_overrides,
        first_target_for_output,
        first_stage_for_output,
        bead_solubility,
    )
    print(json.dumps({"final_overrides": serialize_overrides(global_overrides), "stages": stage_summaries}, indent=2))


def parser():
    base = argparse.ArgumentParser(description="Water/octanol DPD partition sampling")
    sub = base.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--solute", default="ethanol", help="built-in name or comma-separated Anderson bead list")
        p.add_argument("--organic-solvent", default="octanol", help="organic phase recipe name or comma-separated bead list")
        p.add_argument("--n-solute", type=int, default=40)
        p.add_argument("--water-count", type=int, help="explicit number of water molecules/beads")
        p.add_argument("--organic-count", type=int, help="explicit number of organic solvent molecules")
        p.add_argument("--box", type=float, nargs=3, default=[10.0, 10.0, 30.0])
        p.add_argument("--density", type=float, default=1.0, help="coarse bead number density per phase")
        p.add_argument("--water-density", type=float, help="override coarse bead density in the water phase")
        p.add_argument("--octanol-density", type=float, help="override coarse bead density in the octanol phase")
        p.add_argument("--organic-density", type=float, help="override coarse bead density in the organic phase")
        p.add_argument("--solute-bead-fraction", type=float, help="set solute count from this fraction of solvent beads")
        p.add_argument("--solute-init", choices=["split", "water", "octanol", "interface"], default="split")
        p.add_argument("--interface-width", type=float, default=2.0)
        p.add_argument("--steps", type=int, default=200000)
        p.add_argument("--dump-every", type=int, default=2000)
        p.add_argument("--timestep", type=float, default=0.01)
        p.add_argument("--ensemble", choices=["nve", "nph"], default="nve", help="integrator for partition simulations")
        p.add_argument("--pressure", type=float, default=23.7, help="reduced pressure used with --ensemble nph")
        p.add_argument("--pressure-damp", type=float, default=2.0, help="LAMMPS pressure damping used with --ensemble nph")
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--sigma", type=float, default=4.5)
        p.add_argument("--seed", type=int, default=20260506)
        p.add_argument("--table", default="pdf/logp/machine_readable_interactions.cvs")
        p.add_argument("--outdir", default="logp_partition_run")
        p.add_argument("--override", action="append", default=[], help="override pair as BEAD1:BEAD2=Aij")
        p.add_argument("--default-missing-aij", type=float, help="Aij for missing non-fitted pairs")
        p.add_argument("--bead-solubility", action="append", default=[], help="bead solubility as BEAD=VALUE")
        p.add_argument("--bead-heavy-atoms", action="append", default=[], help="heavy atom count as BEAD=COUNT; repeat as needed")
        p.add_argument(
            "--heavy-atom-correction",
            choices=["none", "missing", "all"],
            default="missing",
            help="use heavy atom counts to estimate missing/all Rij and missing/all bonded r0",
        )
        p.add_argument("--heavy-radius-scale", type=float, default=DEFAULT_HEAVY_ATOM_RADIUS_SCALE)
        p.add_argument("--bonded-r0-factor", type=float, default=DEFAULT_BONDED_R0_FACTOR)
        p.add_argument(
            "--angle-param-mode",
            choices=["article", "geometry", "heuristic", "none"],
            default=None,
            help=(
                "angle parameters: article uses literature 105/125 degree rules, "
                "geometry uses template coordinates with heuristic fallback, "
                "heuristic uses weak rules for unknown beads, none disables angles"
            ),
        )
        p.add_argument("--min-aij", type=float, default=5.0)
        p.add_argument("--max-aij", type=float, default=80.0)
        p.add_argument("--allow-custom-beads", action="store_true")
        p.add_argument("--article-protocol", action="store_true", help="use the literature logP simulation protocol defaults")
        p.add_argument("--analysis-method", choices=["fixed", "ummap"], default="fixed")
        p.add_argument("--equilibration-steps", type=int, default=0)
        p.add_argument("--analysis-frames", type=int, help="number of post-equilibration frames used for logP analysis")
        p.add_argument("--slabs", type=int, default=30, help="slabs for UMMAP-style z-profile analysis")
        p.add_argument("--interface-gradient-fraction", type=float, default=0.3)
        p.add_argument("--interface-slab-padding", type=int, default=1)
        p.add_argument(
            "--charge-method",
            choices=["auto", "explicit", "none"],
            default="auto",
            help="explicit-charge DPD for charged logP beads: auto enables Slater+PPPM only when charges are present",
        )
        p.add_argument("--bead-charge", action="append", default=[], help="override bead charge as BEAD=CHARGE")
        p.add_argument("--charge-unit-scale", type=float, default=1.0, help="multiplier for reduced LAMMPS charges")
        p.add_argument("--charge-lambda", type=float, default=0.25, help="Slater charge smearing parameter")
        p.add_argument("--coul-cutoff", type=float, default=3.0, help="Coulomb cutoff for dpd/coul/slater/long")
        p.add_argument("--kspace-accuracy", type=float, default=1.0e-4, help="PPPM accuracy for charged logP simulations")
        p.add_argument("--no-progress", action="store_true", help="disable progress bars")

    def add_bayesian_options(p):
        p.add_argument("--bo-initial", type=int, default=10, help="random initial discrete Aij samples")
        p.add_argument("--bo-grid", action="append", default=[], help="LOW:HIGH:STEP or BEAD1:BEAD2=LOW:HIGH:STEP")
        p.add_argument("--bo-grid-step", type=float, default=1.0, help="default discrete Aij grid spacing")
        p.add_argument("--bo-acq-candidates", type=int, default=5000, help="random EI candidates used only with --no-bo-full-grid-ei")
        p.add_argument("--bo-epsilon", type=float, default=0.01, help="expected-improvement exploration margin")
        p.add_argument("--bo-noise", type=float, default=1e-6, help="Gaussian-process white-noise level")
        p.add_argument("--bo-gp-restarts", type=int, default=2, help="Gaussian-process hyperparameter restarts")
        p.add_argument("--bo-full-grid-ei", action=argparse.BooleanOptionalAction, default=True, help="scan the full discrete grid for maximum EI")
        p.add_argument("--bo-grid-chunk-size", type=int, default=100000, help="candidate chunk size for full-grid EI scans")
        p.add_argument("--penalize-failures", action="store_true", default=True, help="keep fitting after failed LAMMPS/analysis runs")
        p.add_argument("--no-penalize-failures", dest="penalize_failures", action="store_false", help="raise failed LAMMPS/analysis runs immediately")
        p.add_argument("--failure-penalty-loss", type=float, default=100.0, help="objective value used for failed single-target evaluations")

    build_p = sub.add_parser("build", help="write LAMMPS data/input files only")
    add_common(build_p)

    run_p = sub.add_parser("run", help="build, run LAMMPS, then analyze")
    add_common(run_p)
    run_p.add_argument("--job", type=int, default=1)
    run_p.add_argument("--lammps", default=default_lammps_executable())
    run_p.add_argument("--gpu", action="store_true")
    run_p.add_argument("--keep-fraction", type=float, default=0.5)
    run_p.add_argument("--pseudocount", type=float, default=0.5)

    fit_p = sub.add_parser("fit", help="iteratively fit selected Aij values to a target logP")
    add_common(fit_p)
    fit_p.add_argument("--target-logp", type=float)
    fit_p.add_argument("--target-smiles", help="SMILES used to estimate target logP when --target-logp is omitted")
    fit_p.add_argument("--target-logp-value", action="append", default=[], help="external estimate as METHOD=VALUE")
    fit_p.add_argument("--target-outlier-method", choices=["mad", "iqr", "none"], default="mad")
    fit_p.add_argument("--target-min-methods", type=int, default=2)
    fit_p.add_argument("--target-std-factor", type=float, default=0.5)
    fit_p.add_argument("--max-abs-error", type=float, default=0.05)
    fit_p.add_argument("--no-pubchem", action="store_true")
    fit_p.add_argument("--no-openbabel", action="store_true")
    fit_p.add_argument("--fit-pairs", default="auto", help="auto or comma list like BEAD:H2O,BEAD:CH2CH2")
    fit_p.add_argument("--initial-aij", type=float, default=25.0)
    fit_p.add_argument("--optimizer", choices=["nelder-mead", "coordinate", "manual", "bayesian"], default="nelder-mead")
    fit_p.add_argument("--simplex-step", type=float, default=2.0)
    fit_p.add_argument("--xatol", type=float, default=0.1)
    fit_p.add_argument("--fatol", type=float, default=0.0025)
    fit_p.add_argument("--max-iter", type=int, default=5)
    fit_p.add_argument("--tolerance", type=float, default=0.1)
    fit_p.add_argument("--job", type=int, default=1)
    fit_p.add_argument("--lammps", default=default_lammps_executable())
    fit_p.add_argument("--gpu", action="store_true")
    fit_p.add_argument("--keep-fraction", type=float, default=0.5)
    fit_p.add_argument("--pseudocount", type=float, default=0.5)
    add_bayesian_options(fit_p)

    staged_p = sub.add_parser("fit-staged", help="multi-stage Anderson-style logP fitting from a JSON config")
    add_common(staged_p)
    staged_p.add_argument("--stage-config", required=True)
    staged_p.add_argument("--target-outlier-method", choices=["mad", "iqr", "none"], default="mad")
    staged_p.add_argument("--target-min-methods", type=int, default=2)
    staged_p.add_argument("--target-std-factor", type=float, default=0.5)
    staged_p.add_argument("--max-abs-error", type=float, default=0.05)
    staged_p.add_argument("--density-tolerance", type=float, default=0.05, help="relative RMSE tolerance for density stages")
    staged_p.add_argument("--initial-density", type=float, default=3.0, help="initial reduced bead density for pure-liquid density stages")
    staged_p.add_argument("--n-molecules", type=int, default=100, help="molecules in pure-liquid density stages")
    staged_p.add_argument("--density-mixing-rule", choices=["self_arithmetic", "table"], default="self_arithmetic")
    staged_p.add_argument("--no-pubchem", action="store_true")
    staged_p.add_argument("--no-openbabel", action="store_true")
    staged_p.add_argument("--initial-aij", type=float, default=25.0)
    staged_p.add_argument("--optimizer", choices=["nelder-mead", "coordinate", "bayesian"], default="nelder-mead")
    staged_p.add_argument("--simplex-step", type=float, default=2.0)
    staged_p.add_argument("--xatol", type=float, default=0.1)
    staged_p.add_argument("--fatol", type=float, default=0.0025)
    staged_p.add_argument("--max-iter", type=int, default=5)
    staged_p.add_argument("--tolerance", type=float, default=0.1)
    staged_p.add_argument("--job", type=int, default=1)
    staged_p.add_argument("--lammps", default=default_lammps_executable())
    staged_p.add_argument("--gpu", action="store_true")
    staged_p.add_argument("--keep-fraction", type=float, default=0.5)
    staged_p.add_argument("--pseudocount", type=float, default=0.5)
    add_bayesian_options(staged_p)

    analyze_p = sub.add_parser("analyze", help="analyze an existing dump.lammpstrj")
    analyze_p.add_argument("--outdir", default="logp_partition_run")
    analyze_p.add_argument("--keep-fraction", type=float, default=0.5)
    analyze_p.add_argument("--interface-width", type=float)
    analyze_p.add_argument("--pseudocount", type=float, default=0.5)
    analyze_p.add_argument("--analysis-method", choices=["fixed", "ummap"], default="fixed")
    analyze_p.add_argument("--equilibration-steps", type=int, default=0)
    analyze_p.add_argument("--analysis-frames", type=int)
    analyze_p.add_argument("--slabs", type=int, default=30)
    analyze_p.add_argument("--interface-gradient-fraction", type=float, default=0.3)
    analyze_p.add_argument("--interface-slab-padding", type=int, default=1)
    return base


def apply_article_protocol(args):
    if not getattr(args, "article_protocol", False):
        return args
    args.box = [20.0, 20.0, 60.0]
    args.water_count = 4410
    args.organic_count = 2205
    args.n_solute = 180
    args.steps = 1000000
    args.equilibration_steps = 500000
    args.timestep = 0.01
    args.ensemble = "nph"
    args.pressure = 23.7
    args.pressure_damp = 2.0
    args.dump_every = 1000
    args.analysis_method = "ummap"
    args.slabs = 30
    args.interface_gradient_fraction = 0.3
    if getattr(args, "angle_param_mode", None) is None:
        args.angle_param_mode = "article"
    if getattr(args, "analysis_frames", None) is None:
        args.analysis_frames = 50
    return args


def main():
    args = parser().parse_args()
    args = apply_article_protocol(args)
    if hasattr(args, "angle_param_mode") and args.angle_param_mode is None:
        args.angle_param_mode = "heuristic"
    if args.command == "build":
        args.pair_overrides = parse_pair_overrides(args.override)
        args.bead_solubility_values = parse_bead_solubility(args.bead_solubility)
        args.bead_heavy_atom_counts = parse_bead_heavy_atoms(args.bead_heavy_atoms)
        outdir = build_system(args)
        print(f"wrote partition inputs to {outdir}")
    elif args.command == "run":
        args.pair_overrides = parse_pair_overrides(args.override)
        args.bead_solubility_values = parse_bead_solubility(args.bead_solubility)
        args.bead_heavy_atom_counts = parse_bead_heavy_atoms(args.bead_heavy_atoms)
        run_lammps(args)
    elif args.command == "fit":
        args.bead_heavy_atom_counts = parse_bead_heavy_atoms(args.bead_heavy_atoms)
        run_fit(args)
    elif args.command == "fit-staged":
        args.bead_heavy_atom_counts = parse_bead_heavy_atoms(args.bead_heavy_atoms)
        run_fit_staged(args)
    elif args.command == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
