#!/usr/bin/env python3
"""Build, run, and analyze a water/octanol DPD partition simulation.

This script is intentionally independent from main.py. It creates a slab
system with water on one side, octanol on the other side, multiple solute
molecules, and then estimates C_octanol / C_water from solute molecule centers
in the production trajectory.
"""

import argparse
import json
import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from logp_param import ANDERSON_TYPES, load_logp_table, lookup_pair, solubility_aij
from logp_targets import collect_logp_values, robust_consensus, write_logp_report


BEAD_RECIPES = {
    "water": ["H2O"],
    "octanol": ["CH3", "CH2CH2", "CH2CH2", "CH2CH2", "CH2OH"],
    "ethanol": ["CH3", "CH2OH"],
    "butanol": ["CH3", "CH2CH2", "CH2OH"],
    "hexane": ["CH3", "CH2CH2", "CH2CH2", "CH3"],
    "benzene": ["aCHCH", "aCHCH", "aCHCH"],
}

PHASE_BEADS = {"H2O", "CH3", "CH2CH2", "CH2OH"}
BOND_FORCE_CONSTANT = 150.0
ANGLE_FORCE_CONSTANT = 5.0
DEFAULT_BOND_R0 = 0.39
DEFAULT_BEAD_SOLUBILITY = {
    "H2O": 47.9,
    "CH3": 14.5,
    "CH2CH2": 16.0,
    "CH2OH": 29.0,
    "CH2NH2": 28.0,
    "CH2OCH2": 19.0,
    "CH3OCH2": 18.0,
    "aCHCH": 18.5,
}


@dataclass
class MoleculeTemplate:
    name: str
    beads: list[str]

    @property
    def bonds(self):
        if self.name == "benzene" and len(self.beads) == 3:
            return [(1, 2), (2, 3), (3, 1)]
        return [(idx + 1, idx + 2) for idx in range(len(self.beads) - 1)]

    @property
    def angles(self):
        if self.name == "benzene" and len(self.beads) == 3:
            return [(1, 2, 3, 2), (2, 3, 1, 2), (3, 1, 2, 2)]
        return [(idx + 1, idx + 2, idx + 3, 1) for idx in range(len(self.beads) - 2)]


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
            if not bead.replace("_", "").isalnum():
                raise ValueError(f"custom bead names must be alphanumeric/underscore: {bead}")
    return MoleculeTemplate(name=name, beads=beads)


def add_molecule(atoms, bonds, angles, template, type_ids, bond_type_ids, mol_id, origin, spacing=0.45):
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
    for left, center, right, angle_type in template.angles:
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
):
    overrides = overrides or {}
    table = load_logp_table(table_path)
    rows = []
    for i, bead_i in enumerate(type_names, start=1):
        for j, bead_j in enumerate(type_names[i - 1 :], start=i):
            key = pair_key(bead_i, bead_j)
            pair = lookup_pair(table, bead_i, bead_j)
            cutoff = cutoff_from_self_radii(table, bead_i, bead_j, sigma)
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


def bond_type_data(templates, table_path):
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
            bond_type_ids[key] = len(bond_type_ids) + 1
            bond_coeffs.append((bond_type_ids[key], BOND_FORCE_CONSTANT, r0, key[0], key[1]))
    return bond_type_ids, bond_coeffs


def write_lammps_data(path, atoms, bonds, angles, box, type_names, bond_coeffs):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("LAMMPS water/octanol partition data\n\n")
        handle.write(f"{len(atoms)} atoms\n")
        handle.write(f"{len(bonds)} bonds\n")
        handle.write(f"{len(angles)} angles\n\n")
        handle.write(f"{len(type_names)} atom types\n")
        if bonds:
            handle.write(f"{len(bond_coeffs)} bond types\n")
        if angles:
            handle.write("2 angle types\n")
        handle.write("\n")
        handle.write(f"0.0 {box[0]:.6f} xlo xhi\n")
        handle.write(f"0.0 {box[1]:.6f} ylo yhi\n")
        handle.write(f"0.0 {box[2]:.6f} zlo zhi\n\n")
        handle.write("Atoms # molecular\n\n")
        for atom in atoms:
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


def write_lammps_input(path, pair_rows, bond_coeffs, steps, dump_every, temperature, sigma, use_bonds, use_angles):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            f"""# Water/octanol logP partition sampling
dimension       3
units           lj
boundary        p p p
atom_style      molecular
read_data       logp_partition.data

mass * 1.0
variable        sigma equal {sigma}
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
        if use_angles:
            handle.write(
                f"angle_style     harmonic\n"
                f"angle_coeff     1 {ANGLE_FORCE_CONSTANT:.6f} 180.0\n"
                f"angle_coeff     2 {ANGLE_FORCE_CONSTANT:.6f} 60.0\n\n"
            )
        handle.write(f"pair_style      dpd {temperature} 1.0 92894\n")
        for i, j, aij, cutoff in pair_rows:
            handle.write(f"pair_coeff      {i} {j} {aij:.6f} {cutoff:.6f}\n")
        handle.write(
            f"""
fix             1 all nve
timestep        0.01
thermo          {dump_every}
thermo_modify   lost ignore flush yes lost/bond ignore
dump            1 all custom {dump_every} dump.lammpstrj id mol type x y z

run             {steps}
"""
        )


def write_density_lammps_input(path, pair_rows, bond_coeffs, steps, dump_every, temperature, pressure, sigma, use_bonds, use_angles):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            f"""# Pure liquid density fitting
dimension       3
units           lj
boundary        p p p
atom_style      molecular
read_data       pure_density.data

mass * 1.0
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
        if use_angles:
            handle.write(
                f"angle_style     harmonic\n"
                f"angle_coeff     1 {ANGLE_FORCE_CONSTANT:.6f} 180.0\n"
                f"angle_coeff     2 {ANGLE_FORCE_CONSTANT:.6f} 60.0\n\n"
            )
        handle.write("pair_style      dpd ${T} 1.0 92894\n".replace("${T}", str(temperature)))
        for i, j, aij, cutoff in pair_rows:
            handle.write(f"pair_coeff      {i} {j} {aij:.6f} {cutoff:.6f}\n")
        handle.write(
            f"""
fix             1 all npt temp {temperature} {temperature} 1.0 iso {pressure} {pressure} 10.0
timestep        0.01
thermo          {dump_every}
thermo_modify   lost ignore flush yes lost/bond ignore
dump            1 all custom {dump_every} dump.lammpstrj id mol type x y z

run             {steps}
"""
        )


def build_system(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    allow_custom = getattr(args, "allow_custom_beads", False)
    solute = parse_recipe(args.solute, allow_custom=allow_custom)
    water = parse_recipe("water")
    octanol = parse_recipe("octanol")
    box = tuple(args.box)
    half_z = box[2] / 2.0

    beads_per_water = len(water.beads)
    beads_per_octanol = len(octanol.beads)
    half_volume = box[0] * box[1] * half_z
    water_density = float(getattr(args, "water_density", None) or args.density)
    octanol_density = float(getattr(args, "octanol_density", None) or args.density)
    water_count = max(1, int(water_density * half_volume / beads_per_water))
    octanol_count = max(1, int(octanol_density * half_volume / beads_per_octanol))
    n_solute = int(args.n_solute)
    solute_bead_fraction = getattr(args, "solute_bead_fraction", None)
    if solute_bead_fraction is not None:
        solvent_beads = water_count * beads_per_water + octanol_count * beads_per_octanol
        n_solute = max(1, int(round(float(solute_bead_fraction) * solvent_beads / len(solute.beads))))

    type_names = sorted(set(water.beads + octanol.beads + solute.beads))
    type_ids = {bead: idx + 1 for idx, bead in enumerate(type_names)}
    bond_type_ids, bond_coeffs = bond_type_data([water, octanol, solute], args.table)
    atoms = []
    bonds = []
    angles = []
    solute_mol_ids = []
    mol_id = 1

    for _ in range(water_count):
        add_molecule(atoms, bonds, angles, water, type_ids, bond_type_ids, mol_id, random_point(box, 0.0, half_z))
        mol_id += 1

    for _ in range(octanol_count):
        add_molecule(atoms, bonds, angles, octanol, type_ids, bond_type_ids, mol_id, random_point(box, half_z, box[2]))
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
        add_molecule(atoms, bonds, angles, solute, type_ids, bond_type_ids, mol_id, random_point(box, zlo, zhi))
        solute_mol_ids.append(mol_id)
        mol_id += 1

    overrides = getattr(args, "pair_overrides", {})
    default_missing_aij = getattr(args, "default_missing_aij", None)
    bead_solubility = getattr(args, "bead_solubility_values", None)
    pair_rows = pair_coeff_rows(
        type_names,
        args.table,
        args.sigma,
        overrides,
        default_missing_aij,
        bead_solubility,
        getattr(args, "min_aij", None),
        getattr(args, "max_aij", None),
    )
    write_lammps_data(outdir / "logp_partition.data", atoms, bonds, angles, box, type_names, bond_coeffs)
    write_lammps_input(
        outdir / "logp_partition.in",
        pair_rows,
        bond_coeffs,
        args.steps,
        args.dump_every,
        args.temperature,
        args.sigma,
        bool(bonds),
        bool(angles),
    )

    metadata = {
        "solute": args.solute,
        "solute_beads": solute.beads,
        "solute_mol_ids": solute_mol_ids,
        "water_molecules": water_count,
        "octanol_molecules": octanol_count,
        "n_solute": n_solute,
        "box": list(box),
        "density": args.density,
        "water_density": water_density,
        "octanol_density": octanol_density,
        "solute_bead_fraction": solute_bead_fraction,
        "interface_z": half_z,
        "interface_width": args.interface_width,
        "type_ids": type_ids,
        "type_names": type_names,
        "bond_coeffs": [
            {"type": bond_type, "force_constant": force_constant, "r0": r0, "bead_i": bead_i, "bead_j": bead_j}
            for bond_type, force_constant, r0, bead_i, bead_j in bond_coeffs
        ],
        "angle_coeffs": [
            {"type": 1, "force_constant": ANGLE_FORCE_CONSTANT, "theta0": 180.0},
            {"type": 2, "force_constant": ANGLE_FORCE_CONSTANT, "theta0": 60.0},
        ] if angles else [],
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
    molecule = parse_recipe(args.solute, allow_custom=allow_custom)
    n_molecules = int(getattr(args, "n_molecules", 100))
    initial_density = float(getattr(args, "initial_density", 3.0))
    total_beads = max(1, n_molecules * len(molecule.beads))
    box_length = (total_beads / initial_density) ** (1.0 / 3.0)
    box = (box_length, box_length, box_length)

    type_names = sorted(set(molecule.beads))
    type_ids = {bead: idx + 1 for idx, bead in enumerate(type_names)}
    bond_type_ids, bond_coeffs = bond_type_data([molecule], args.table)
    atoms = []
    bonds = []
    angles = []
    for mol_id in range(1, n_molecules + 1):
        add_molecule(atoms, bonds, angles, molecule, type_ids, bond_type_ids, mol_id, random_point(box, 0.0, box[2], margin=0.2))

    overrides = getattr(args, "pair_overrides", {})
    if getattr(args, "density_mixing_rule", "self_arithmetic") == "self_arithmetic":
        overrides = arithmetic_self_mixing_overrides(molecule.beads, overrides)
    default_missing_aij = getattr(args, "default_missing_aij", None)
    bead_solubility = getattr(args, "bead_solubility_values", None)
    pair_rows = pair_coeff_rows(
        type_names,
        args.table,
        args.sigma,
        overrides,
        default_missing_aij,
        bead_solubility,
        getattr(args, "min_aij", None),
        getattr(args, "max_aij", None),
    )
    write_lammps_data(outdir / "pure_density.data", atoms, bonds, angles, box, type_names, bond_coeffs)
    write_density_lammps_input(
        outdir / "pure_density.in",
        pair_rows,
        bond_coeffs,
        int(getattr(args, "steps", 30000)),
        int(getattr(args, "dump_every", 1000)),
        float(getattr(args, "temperature", 1.0)),
        float(getattr(args, "pressure", 23.7)),
        float(getattr(args, "sigma", 1.0)),
        bool(bonds),
        bool(angles),
    )
    metadata = {
        "molecule": args.solute,
        "molecule_beads": molecule.beads,
        "molecules": n_molecules,
        "atoms": len(atoms),
        "initial_density": initial_density,
        "pressure": float(getattr(args, "pressure", 23.7)),
        "type_ids": type_ids,
        "type_names": type_names,
        "bond_coeffs": [
            {"type": bond_type, "force_constant": force_constant, "r0": r0, "bead_i": bead_i, "bead_j": bead_j}
            for bond_type, force_constant, r0, bead_i, bead_j in bond_coeffs
        ],
        "angle_coeffs": [
            {"type": 1, "force_constant": ANGLE_FORCE_CONSTANT, "theta0": 180.0},
            {"type": 2, "force_constant": ANGLE_FORCE_CONSTANT, "theta0": 60.0},
        ] if angles else [],
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


def analyze(args):
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
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(col, "")) for col in columns) + "\n")


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
        row = {
            "iteration": iteration,
            "optimizer": args.optimizer,
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
        return objective

    x0 = np.asarray([overrides[key] for key in fit_pairs], dtype=float)
    if args.optimizer == "nelder-mead":
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
    else:
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
        "box",
        "density",
        "water_density",
        "octanol_density",
        "solute_bead_fraction",
        "steps",
        "dump_every",
        "solute_init",
        "interface_width",
        "temperature",
        "sigma",
        "seed",
        "default_missing_aij",
        "min_aij",
        "max_aij",
        "allow_custom_beads",
        "keep_fraction",
        "pseudocount",
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
        "keep_fraction": ["keep_fraction"],
        "density_mixing_rule": ["density_mixing_rule"],
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


def run_stage_target(args, stage, target, run_dir, overrides, bead_solubility, lammps):
    run_args = namespace_for_stage_target(args, stage, target, run_dir, overrides, bead_solubility)
    summary_path = Path(run_dir) / "partition_summary.json"
    dump_path = Path(run_dir) / "dump.lammpstrj"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if dump_path.exists():
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
    if dump_path.exists():
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


def write_staged_outputs(outdir, args, table, global_overrides, first_solute, bead_solubility):
    final_args = argparse.Namespace(**vars(args))
    final_args.solute = first_solute
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
    first_solute = None

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
                eval_dir = stage_dir / f"eval_{evaluation:03d}"
                for target_index, target in enumerate(targets, start=1):
                    target_name = target.get("name", target.get("molecule", target.get("solute")))
                    run_dir = eval_dir / f"target_{target_index:02d}_{target_name}"
                    summary = run_density_target(args, stage, target, run_dir, candidate_overrides, bead_solubility, lammps)
                    density_key = target.get("density_kind", "bead_number_density")
                    observed = float(summary[density_key])
                    target_density = float(target["target_density"])
                    scale = float(target.get("error_scale", target_density if target_density != 0 else 1.0))
                    error = (target_density - observed) / scale
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
                return rmse * rmse

            x0 = np.asarray([global_overrides[key] for key in fit_pairs], dtype=float)
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
                        "maxfev": int(stage.get("max_iter", args.max_iter)),
                        "maxiter": int(stage.get("max_iter", args.max_iter)),
                        "initial_simplex": np.asarray(simplex),
                        "xatol": args.xatol,
                        "fatol": args.fatol,
                        "disp": False,
                    },
                )
            else:
                current = np.array(x0, copy=True)
                max_iter = int(stage.get("max_iter", args.max_iter))
                step = float(stage.get("simplex_step", args.simplex_step))
                while len(stage_history) < max_iter:
                    evaluate_density_stage(current)
                    if best and best["max_abs_error"] <= effective_tolerance:
                        break
                    dim = (len(stage_history) - 1) % len(current)
                    current[dim] = clamp_aij(current[dim] + step, args.min_aij, args.max_aij)
                    step *= -0.5

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
        if first_solute is None:
            first_solute = targets[0]["solute"]

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
            eval_dir = stage_dir / f"eval_{evaluation:03d}"
            for target_index, target in enumerate(targets, start=1):
                target_name = target.get("name", target["solute"])
                run_dir = eval_dir / f"target_{target_index:02d}_{target_name}"
                summary = run_stage_target(args, stage, target, run_dir, candidate_overrides, bead_solubility, lammps)
                observed = float(summary["log10_partition_ratio"])
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
            return rmse * rmse

        x0 = np.asarray([global_overrides[key] for key in fit_pairs], dtype=float)
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
                    "maxfev": int(stage.get("max_iter", args.max_iter)),
                    "maxiter": int(stage.get("max_iter", args.max_iter)),
                    "initial_simplex": np.asarray(simplex),
                    "xatol": args.xatol,
                    "fatol": args.fatol,
                    "disp": False,
                },
            )
        else:
            current = np.array(x0, copy=True)
            max_iter = int(stage.get("max_iter", args.max_iter))
            step = float(stage.get("simplex_step", args.simplex_step))
            while len(stage_history) < max_iter:
                evaluate_stage(current)
                if best and best["max_abs_error"] <= effective_tolerance:
                    break
                dim = (len(stage_history) - 1) % len(current)
                current[dim] = clamp_aij(current[dim] + step, args.min_aij, args.max_aij)
                step *= -0.5

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
    write_staged_outputs(outdir, args, table, global_overrides, first_solute or args.solute, bead_solubility)
    print(json.dumps({"final_overrides": serialize_overrides(global_overrides), "stages": stage_summaries}, indent=2))


def parser():
    base = argparse.ArgumentParser(description="Water/octanol DPD partition sampling")
    sub = base.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--solute", default="ethanol", help="built-in name or comma-separated Anderson bead list")
        p.add_argument("--n-solute", type=int, default=40)
        p.add_argument("--box", type=float, nargs=3, default=[10.0, 10.0, 30.0])
        p.add_argument("--density", type=float, default=1.0, help="coarse bead number density per phase")
        p.add_argument("--water-density", type=float, help="override coarse bead density in the water phase")
        p.add_argument("--octanol-density", type=float, help="override coarse bead density in the octanol phase")
        p.add_argument("--solute-bead-fraction", type=float, help="set solute count from this fraction of solvent beads")
        p.add_argument("--solute-init", choices=["split", "water", "octanol", "interface"], default="split")
        p.add_argument("--interface-width", type=float, default=2.0)
        p.add_argument("--steps", type=int, default=200000)
        p.add_argument("--dump-every", type=int, default=2000)
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--sigma", type=float, default=4.5)
        p.add_argument("--seed", type=int, default=20260506)
        p.add_argument("--table", default="pdf/logp/machine_readable_interactions.cvs")
        p.add_argument("--outdir", default="logp_partition_run")
        p.add_argument("--override", action="append", default=[], help="override pair as BEAD1:BEAD2=Aij")
        p.add_argument("--default-missing-aij", type=float, help="Aij for missing non-fitted pairs")
        p.add_argument("--bead-solubility", action="append", default=[], help="bead solubility as BEAD=VALUE")
        p.add_argument("--min-aij", type=float, default=5.0)
        p.add_argument("--max-aij", type=float, default=80.0)
        p.add_argument("--allow-custom-beads", action="store_true")

    build_p = sub.add_parser("build", help="write LAMMPS data/input files only")
    add_common(build_p)

    run_p = sub.add_parser("run", help="build, run LAMMPS, then analyze")
    add_common(run_p)
    run_p.add_argument("--job", type=int, default=1)
    run_p.add_argument("--lammps", default="lmp_mpi")
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
    fit_p.add_argument("--optimizer", choices=["nelder-mead", "coordinate"], default="nelder-mead")
    fit_p.add_argument("--simplex-step", type=float, default=2.0)
    fit_p.add_argument("--xatol", type=float, default=0.1)
    fit_p.add_argument("--fatol", type=float, default=0.0025)
    fit_p.add_argument("--max-iter", type=int, default=5)
    fit_p.add_argument("--tolerance", type=float, default=0.1)
    fit_p.add_argument("--job", type=int, default=1)
    fit_p.add_argument("--lammps", default="lmp_mpi")
    fit_p.add_argument("--gpu", action="store_true")
    fit_p.add_argument("--keep-fraction", type=float, default=0.5)
    fit_p.add_argument("--pseudocount", type=float, default=0.5)

    staged_p = sub.add_parser("fit-staged", help="multi-stage Anderson-style logP fitting from a JSON config")
    add_common(staged_p)
    staged_p.add_argument("--stage-config", required=True)
    staged_p.add_argument("--target-outlier-method", choices=["mad", "iqr", "none"], default="mad")
    staged_p.add_argument("--target-min-methods", type=int, default=2)
    staged_p.add_argument("--target-std-factor", type=float, default=0.5)
    staged_p.add_argument("--max-abs-error", type=float, default=0.05)
    staged_p.add_argument("--density-tolerance", type=float, default=0.05, help="relative RMSE tolerance for density stages")
    staged_p.add_argument("--pressure", type=float, default=23.7, help="reduced NPT pressure for pure-liquid density stages")
    staged_p.add_argument("--initial-density", type=float, default=3.0, help="initial reduced bead density for pure-liquid density stages")
    staged_p.add_argument("--n-molecules", type=int, default=100, help="molecules in pure-liquid density stages")
    staged_p.add_argument("--density-mixing-rule", choices=["self_arithmetic", "table"], default="self_arithmetic")
    staged_p.add_argument("--no-pubchem", action="store_true")
    staged_p.add_argument("--no-openbabel", action="store_true")
    staged_p.add_argument("--initial-aij", type=float, default=25.0)
    staged_p.add_argument("--optimizer", choices=["nelder-mead", "coordinate"], default="nelder-mead")
    staged_p.add_argument("--simplex-step", type=float, default=2.0)
    staged_p.add_argument("--xatol", type=float, default=0.1)
    staged_p.add_argument("--fatol", type=float, default=0.0025)
    staged_p.add_argument("--max-iter", type=int, default=5)
    staged_p.add_argument("--tolerance", type=float, default=0.1)
    staged_p.add_argument("--job", type=int, default=1)
    staged_p.add_argument("--lammps", default="lmp_mpi")
    staged_p.add_argument("--gpu", action="store_true")
    staged_p.add_argument("--keep-fraction", type=float, default=0.5)
    staged_p.add_argument("--pseudocount", type=float, default=0.5)

    analyze_p = sub.add_parser("analyze", help="analyze an existing dump.lammpstrj")
    analyze_p.add_argument("--outdir", default="logp_partition_run")
    analyze_p.add_argument("--keep-fraction", type=float, default=0.5)
    analyze_p.add_argument("--interface-width", type=float)
    analyze_p.add_argument("--pseudocount", type=float, default=0.5)
    return base


def main():
    args = parser().parse_args()
    if args.command == "build":
        args.pair_overrides = parse_pair_overrides(args.override)
        args.bead_solubility_values = parse_bead_solubility(args.bead_solubility)
        outdir = build_system(args)
        print(f"wrote partition inputs to {outdir}")
    elif args.command == "run":
        args.pair_overrides = parse_pair_overrides(args.override)
        args.bead_solubility_values = parse_bead_solubility(args.bead_solubility)
        run_lammps(args)
    elif args.command == "fit":
        run_fit(args)
    elif args.command == "fit-staged":
        run_fit_staged(args)
    elif args.command == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
