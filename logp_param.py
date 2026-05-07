import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


ANDERSON_TYPES = {
    "H2O",
    "CH3",
    "CH2CH2",
    "CH2OH",
    "CH2NH2",
    "CH2OCH2",
    "CH3OCH2",
    "aCHCH",
}

LOGP_EXTENSION_TYPES = {
    "CH2",
    "CH3CH2",
    "CH2OSO3-",
    "Na+",
    "Cl-",
    "COO-",
}

LOGP_SUPPORTED_TYPES = ANDERSON_TYPES | LOGP_EXTENSION_TYPES

EXTENSION_SELF = {
    "CH2": {"A_ij": 24.0, "R_ij": 0.9250, "r0": np.nan},
    "CH3CH2": {"A_ij": 22.0, "R_ij": 1.0980, "r0": np.nan},
    "CH2OSO3-": {"A_ij": 13.3, "R_ij": 1.2340, "r0": np.nan},
    "Na+": {"A_ij": 25.0, "R_ij": 1.0000, "r0": np.nan},
    "Cl-": {"A_ij": 25.0, "R_ij": 1.0000, "r0": np.nan},
    "COO-": {"A_ij": 25.77, "R_ij": 0.9900, "r0": np.nan},
}

LITERATURE_EXTENSION_AIJ = {
    tuple(sorted(("CH2OSO3-", "CH3"))): 28.5,
    tuple(sorted(("CH2OSO3-", "CH2"))): 30.0,
    tuple(sorted(("CH2OSO3-", "CH3CH2"))): 28.5,
    tuple(sorted(("CH2OSO3-", "CH2CH2"))): 28.5,
    tuple(sorted(("CH2OSO3-", "CH2OCH2"))): 15.5,
    tuple(sorted(("CH2OSO3-", "CH2NH2"))): 13.57,
    tuple(sorted(("CH2OSO3-", "COO-"))): 18.18,
}

IONIC_TYPES = {"CH2OSO3-", "Na+", "Cl-", "COO-"}
ALKYL_TYPES = {"CH3", "CH2", "CH3CH2", "CH2CH2"}


@dataclass
class LogpPairResult:
    i: int
    j: int
    type_i: str
    type_j: str
    value: float
    source: str
    r_ij: object = np.nan
    r0: object = np.nan


def canonicalize_smiles(smiles):
    if smiles in {"HOH", "H2O", "O"}:
        return "O"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _heavy_atoms(mol):
    return [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def _has_formal_charge(mol):
    return any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())


def _has_sulfate_or_sulfonate(mol):
    patterns = [
        "[OX2][SX4](=O)(=O)[O-]",
        "[OX2][SX4](=O)(=O)[OX2H]",
        "[CX4][SX4](=O)(=O)[O-]",
        "[CX4][SX4](=O)(=O)[OX2H]",
    ]
    return any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in patterns)


def _has_carboxylate_or_carboxylic_acid(mol):
    patterns = ["[CX3](=O)[O-]", "[CX3](=O)[OX2H]"]
    return any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in patterns)


def classify_bead_smiles(smiles, is_peg=0):
    """Map an OB-DPD fragment SMILES to the bead palette from Anderson et al.

    The mapping is intentionally conservative. Fragments outside the published
    logP parameter palette are marked UNKNOWN so callers can reject the current
    coarse graining instead of forcing an oversized fragment into one bead type.
    """
    if smiles in LOGP_SUPPORTED_TYPES:
        status = "matched" if smiles in ANDERSON_TYPES else "matched_extension"
        return {
            "smiles": smiles,
            "canonical_smiles": smiles,
            "assigned_type": smiles,
            "status": status,
            "reason": "article logP bead type",
        }

    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return {
            "smiles": smiles,
            "canonical_smiles": np.nan,
            "assigned_type": "UNKNOWN",
            "status": "invalid_smiles",
            "reason": "RDKit could not parse this fragment",
        }

    if canonical == "O":
        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "assigned_type": "H2O",
            "status": "matched",
            "reason": "water",
        }

    if canonical == "[Na+]":
        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "assigned_type": "Na+",
            "status": "matched_extension",
            "reason": "sodium counterion from literature extension",
        }

    if canonical == "[Cl-]":
        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "assigned_type": "Cl-",
            "status": "matched_extension",
            "reason": "chloride counterion from literature extension",
        }

    mol = Chem.MolFromSmiles(canonical)

    if _has_sulfate_or_sulfonate(mol):
        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "assigned_type": "CH2OSO3-",
            "status": "matched_extension",
            "reason": "sulfate/sulfonate headgroup from literature extension",
        }

    if _has_carboxylate_or_carboxylic_acid(mol):
        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "assigned_type": "COO-",
            "status": "matched_extension",
            "reason": "carboxyl/carboxylate headgroup from literature extension",
        }

    if _has_formal_charge(mol):
        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "assigned_type": "UNKNOWN",
            "status": "out_of_domain_electrostatics",
            "reason": "charged fragments need explicit or separately fitted electrostatics",
        }

    heavy_atoms = _heavy_atoms(mol)
    symbols = [atom.GetSymbol() for atom in heavy_atoms]
    heavy_count = len(heavy_atoms)

    if any(atom.GetIsAromatic() for atom in heavy_atoms):
        if all(atom.GetSymbol() == "C" for atom in heavy_atoms) and heavy_count == 2:
            return {
                "smiles": smiles,
                "canonical_smiles": canonical,
                "assigned_type": "aCHCH",
                "status": "matched",
                "reason": "aromatic hydrocarbon fragment",
            }
        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "assigned_type": "UNKNOWN",
            "status": "out_of_domain",
            "reason": "aromatic fragments must match one aCHCH bead (two aromatic carbons)",
        }

    if set(symbols) == {"C"}:
        if heavy_count == 1:
            assigned = "CH3"
        elif heavy_count == 2:
            assigned = "CH2CH2"
        else:
            return {
                "smiles": smiles,
                "canonical_smiles": canonical,
                "assigned_type": "UNKNOWN",
                "status": "oversized_fragment",
                "reason": "alkyl fragments larger than two carbons should be split into CH3/CH2CH2 beads",
            }
        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "assigned_type": assigned,
            "status": "matched",
            "reason": "alkyl fragment",
        }

    alcohol = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX4][OX2H]"))
    ether = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX4][OX2][CX4]"))
    amine = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX4][NX3;H2,H1;!$(NC=O)]"))

    if alcohol and set(symbols).issubset({"C", "O"}) and heavy_count == 2:
        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "assigned_type": "CH2OH",
            "status": "matched",
            "reason": "alcohol fragment",
        }

    if ether and set(symbols).issubset({"C", "O"}) and heavy_count == 3:
        assigned = "CH3OCH2" if is_peg else "CH2OCH2"
        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "assigned_type": assigned,
            "status": "matched",
            "reason": "ether fragment",
        }

    if amine and set(symbols).issubset({"C", "N"}) and heavy_count == 2:
        return {
            "smiles": smiles,
            "canonical_smiles": canonical,
            "assigned_type": "CH2NH2",
            "status": "matched",
            "reason": "primary amine fragment",
        }

    return {
        "smiles": smiles,
        "canonical_smiles": canonical,
        "assigned_type": "UNKNOWN",
        "status": "out_of_domain",
        "reason": "fragment is not covered by the Anderson logP bead palette",
    }


def _heavy_degree(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    return sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1)


def _heavy_neighbors(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    return [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]


def _ordered_component(mol, atoms):
    atom_set = set(atoms)
    degrees = {
        idx: sum(1 for nbr in _heavy_neighbors(mol, idx) if nbr in atom_set)
        for idx in atoms
    }
    endpoints = [idx for idx, degree in degrees.items() if degree <= 1]
    if not endpoints:
        raise ValueError("logP article mapping only supports linear non-aromatic segments")

    ordered = []
    previous = None
    current = endpoints[0]
    while current is not None:
        ordered.append(current)
        next_atoms = [
            nbr
            for nbr in _heavy_neighbors(mol, current)
            if nbr in atom_set and nbr != previous
        ]
        next_atom = next_atoms[0] if next_atoms else None
        previous, current = current, next_atom

    if set(ordered) != atom_set:
        raise ValueError("logP article mapping could not order a branched segment")
    return ordered


def _aromatic_ring_order(mol, ring_atoms):
    ring_set = set(ring_atoms)
    start = ring_atoms[0]
    ordered = [start]
    previous = None
    current = start
    while len(ordered) < len(ring_atoms):
        candidates = [
            nbr
            for nbr in _heavy_neighbors(mol, current)
            if nbr in ring_set and nbr != previous
        ]
        next_atom = None
        for candidate in candidates:
            if candidate not in ordered:
                next_atom = candidate
                break
        if next_atom is None:
            break
        previous, current = current, next_atom
        ordered.append(current)
    if len(ordered) != len(ring_atoms):
        return list(ring_atoms)
    return ordered


def _embed_heavy_atom_coordinates(mol):
    mol_h = Chem.AddHs(Chem.Mol(mol))
    ok = AllChem.EmbedMolecule(mol_h, randomSeed=1)
    if ok == 0:
        try:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        conf = mol_h.GetConformer()
        return np.array(
            [
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ],
            dtype=float,
        )
    return np.array([[float(i), 0.0, 0.0] for i in range(mol.GetNumAtoms())], dtype=float)


def map_smiles_to_logp_article_beads(smiles, is_peg=0):
    """Split a molecule into the Anderson logP bead palette.

    This mapper is intentionally separate from the OB-DPD solubility-parameter
    mapper. It supports the article's linear alkanes, alcohols, polyols, amines,
    ethers, benzene rings, and combinations of these motifs. Unsupported motifs
    raise a ValueError instead of being coerced into a table bead.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES for logP mapping: {smiles}")

    assigned = {}
    bead_types = []
    bead_atom_groups = []

    def add_bead(bead_type, atom_group):
        bead_idx = len(bead_types)
        for atom_idx in atom_group:
            if atom_idx in assigned:
                raise ValueError(
                    f"atom {atom_idx} is assigned to more than one logP article bead"
                )
        bead_types.append(bead_type)
        bead_atom_groups.append(list(atom_group))
        for atom_idx in atom_group:
            assigned[atom_idx] = bead_idx

    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        ring_atoms = list(ring)
        if len(ring_atoms) != 6:
            continue
        if not all(
            mol.GetAtomWithIdx(idx).GetIsAromatic()
            and mol.GetAtomWithIdx(idx).GetSymbol() == "C"
            for idx in ring_atoms
        ):
            continue
        ordered = _aromatic_ring_order(mol, ring_atoms)
        for start in range(0, 6, 2):
            add_bead("aCHCH", ordered[start : start + 2])

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in assigned or atom.GetAtomicNum() != 8:
            continue
        neighbors = _heavy_neighbors(mol, idx)
        if len(neighbors) == 2 and all(
            mol.GetAtomWithIdx(nbr).GetAtomicNum() == 6 for nbr in neighbors
        ):
            if any(nbr in assigned for nbr in neighbors):
                continue
            terminal_neighbor = any(_heavy_degree(mol, nbr) == 1 for nbr in neighbors)
            bead_type = "CH3OCH2" if (terminal_neighbor or is_peg) else "CH2OCH2"
            add_bead(bead_type, [neighbors[0], idx, neighbors[1]])

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in assigned or atom.GetAtomicNum() != 8:
            continue
        neighbors = _heavy_neighbors(mol, idx)
        if len(neighbors) == 1:
            carbon = neighbors[0]
            if carbon not in assigned and mol.GetAtomWithIdx(carbon).GetAtomicNum() == 6:
                add_bead("CH2OH", [carbon, idx])

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in assigned or atom.GetAtomicNum() != 7:
            continue
        neighbors = _heavy_neighbors(mol, idx)
        if len(neighbors) == 1:
            carbon = neighbors[0]
            if carbon not in assigned and mol.GetAtomWithIdx(carbon).GetAtomicNum() == 6:
                add_bead("CH2NH2", [carbon, idx])

    remaining_carbons = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6
        and not atom.GetIsAromatic()
        and atom.GetIdx() not in assigned
    ]
    remaining = set(remaining_carbons)
    while remaining:
        stack = [next(iter(remaining))]
        component = []
        remaining.remove(stack[0])
        while stack:
            current = stack.pop()
            component.append(current)
            for nbr in _heavy_neighbors(mol, current):
                if nbr in remaining:
                    remaining.remove(nbr)
                    stack.append(nbr)

        ordered = _ordered_component(mol, component)
        start = 0
        end = len(ordered)
        if end > start and _heavy_degree(mol, ordered[start]) == 1:
            add_bead("CH3", [ordered[start]])
            start += 1
        if end > start and _heavy_degree(mol, ordered[end - 1]) == 1:
            end -= 1
            right_terminal = ordered[end]
        else:
            right_terminal = None

        middle = ordered[start:end]
        if len(middle) % 2 != 0:
            add_bead("CH2", [middle[0]])
            middle = middle[1:]
        for offset in range(0, len(middle), 2):
            add_bead("CH2CH2", middle[offset : offset + 2])
        if right_terminal is not None:
            add_bead("CH3", [right_terminal])

    unassigned = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in assigned
    ]
    if unassigned:
        labels = [
            f"{idx}:{mol.GetAtomWithIdx(idx).GetSymbol()}" for idx in sorted(unassigned)
        ]
        raise ValueError(
            "unsupported atoms for Anderson logP bead mapping: " + ", ".join(labels)
        )

    bead_count = len(bead_types)
    adjacency = np.zeros((bead_count, bead_count), dtype=int)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bead_i = assigned.get(begin)
        bead_j = assigned.get(end)
        if bead_i is not None and bead_j is not None and bead_i != bead_j:
            adjacency[bead_i, bead_j] = 1
            adjacency[bead_j, bead_i] = 1

    atom_coords = _embed_heavy_atom_coordinates(mol)
    bead_coords = np.array(
        [atom_coords[group].mean(axis=0) for group in bead_atom_groups],
        dtype=float,
    )
    return {
        "bead_types": bead_types,
        "bead_atom_groups": bead_atom_groups,
        "adjacency": adjacency,
        "coords": bead_coords,
    }


def load_logp_table(table_path):
    if not os.path.exists(table_path):
        module_relative = os.path.join(os.path.dirname(os.path.abspath(__file__)), table_path)
        if os.path.exists(module_relative):
            table_path = module_relative
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"logP parameter table not found: {table_path}")
    table = pd.read_csv(table_path)
    table.columns = [col.strip() for col in table.columns]
    for col in ["beadi", "beadj"]:
        table[col] = table[col].astype(str).str.strip()
    for col in ["A_ij", "R_ij"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    if "r0" not in table.columns:
        table["r0"] = np.nan
    table["r0"] = pd.to_numeric(table["r0"], errors="coerce")
    return table


def lookup_pair(table, type_i, type_j):
    pair = table[
        ((table["beadi"] == type_i) & (table["beadj"] == type_j))
        | ((table["beadi"] == type_j) & (table["beadj"] == type_i))
    ]
    if pair.empty:
        return None
    row = pair.iloc[0]
    return {
        "A_ij": float(row["A_ij"]),
        "R_ij": row.get("R_ij", np.nan),
        "r0": row.get("r0", np.nan),
    }


def lookup_self_parameter(table, bead_type):
    pair = lookup_pair(table, bead_type, bead_type)
    if pair is not None:
        return pair
    return EXTENSION_SELF.get(bead_type)


def interpolated_rij(table, type_i, type_j):
    self_i = lookup_self_parameter(table, type_i)
    self_j = lookup_self_parameter(table, type_j)
    if self_i is None or self_j is None:
        return np.nan
    r_i = self_i.get("R_ij", np.nan)
    r_j = self_j.get("R_ij", np.nan)
    if pd.isna(r_i) or pd.isna(r_j):
        return np.nan
    return 0.5 * (float(r_i) + float(r_j))


def interpolate_ch2_pair(table, type_i, type_j, r_ij):
    if "CH2" not in {type_i, type_j}:
        return None
    other = type_j if type_i == "CH2" else type_i
    if other == "CH2":
        return None

    references = []
    for alkyl_ref in ("CH3", "CH2CH2"):
        pair = lookup_pair(table, alkyl_ref, other)
        if pair is not None:
            references.append(pair)
    if not references:
        return None

    return {
        "A_ij": float(np.mean([pair["A_ij"] for pair in references])),
        "R_ij": r_ij,
        "r0": np.nan,
        "source": "logp_ch2_alkyl_interpolation",
    }


def lookup_pair_with_interpolation(table, type_i, type_j):
    pair = lookup_pair(table, type_i, type_j)
    if pair is not None:
        pair["source"] = "logp_table"
        return pair

    if type_i not in LOGP_SUPPORTED_TYPES or type_j not in LOGP_SUPPORTED_TYPES:
        return None

    if type_i == type_j:
        self_pair = lookup_self_parameter(table, type_i)
        if self_pair is not None:
            result = dict(self_pair)
            result["source"] = "logp_extension_self"
            return result
        return None

    key = tuple(sorted((type_i, type_j)))
    r_ij = interpolated_rij(table, type_i, type_j)
    if pd.isna(r_ij):
        return None

    if key in LITERATURE_EXTENSION_AIJ:
        return {
            "A_ij": LITERATURE_EXTENSION_AIJ[key],
            "R_ij": r_ij,
            "r0": np.nan,
            "source": "logp_literature_extension",
        }

    ch2_pair = interpolate_ch2_pair(table, type_i, type_j, r_ij)
    if ch2_pair is not None:
        return ch2_pair

    if (type_i in IONIC_TYPES and type_j in IONIC_TYPES) or (
        "H2O" in {type_i, type_j} and ({type_i, type_j} & IONIC_TYPES)
    ):
        return {
            "A_ij": 25.0 / (float(r_ij) ** 3),
            "R_ij": r_ij,
            "r0": np.nan,
            "source": "logp_ionic_mean_field",
        }

    if ({type_i, type_j} & {"Na+", "Cl-"}) and ({type_i, type_j} & ALKYL_TYPES):
        alkyl = type_j if type_i in {"Na+", "Cl-"} else type_i
        water_pair = lookup_pair(table, "H2O", alkyl)
        if water_pair is None:
            return None
        return {
            "A_ij": water_pair["A_ij"],
            "R_ij": r_ij,
            "r0": np.nan,
            "source": "logp_counterion_alkyl_copy_water",
        }

    if ({type_i, type_j} & {"Na+", "Cl-"}) and ({type_i, type_j} & {"CH2OCH2"}):
        water_pair = lookup_pair(table, "H2O", "CH2OCH2")
        if water_pair is None:
            return None
        return {
            "A_ij": water_pair["A_ij"],
            "R_ij": r_ij,
            "r0": np.nan,
            "source": "logp_counterion_ether_copy_water",
        }

    return None


def flory_huggins_parameter(sigma1, sigma2):
    V = 129
    k = 8.31451
    T = 273.15 + 25
    return (V * (sigma1 - sigma2) ** 2) / (k * T)


def solubility_aij(sigma1, sigma2, same_type=False):
    if same_type:
        return 25.0
    return 25 + 3.27 * flory_huggins_parameter(sigma1, sigma2)


def optimize_missing_logp_params(*args, **kwargs):
    raise NotImplementedError(
        "automatic logP fitting is not part of the main parameterization path. "
        "Use a completed table with --logp_table or run logp_partition.py separately "
        "for research calibration."
    )


def create_logp_aij_list(
    bead_smiles,
    solubility_values=None,
    table_path="pdf/logp/machine_readable_interactions.cvs",
    missing="error",
    is_peg_list=None,
    assignment_output="bead_type_assignment.csv",
    pair_output="a_ij_source.csv",
):
    if missing not in {"fallback", "error", "optimize"}:
        raise ValueError("missing must be one of: fallback, error, optimize")

    table = load_logp_table(table_path)
    is_peg_list = is_peg_list or [0] * len(bead_smiles)
    assignments = [
        classify_bead_smiles(smiles, is_peg=is_peg_list[idx] if idx < len(is_peg_list) else 0)
        for idx, smiles in enumerate(bead_smiles)
    ]
    pd.DataFrame(assignments).to_csv(assignment_output, index=False)

    a_ij = []
    pair_rows = []
    count = len(bead_smiles)
    for i in range(count):
        for j in range(i, count):
            type_i = assignments[i]["assigned_type"]
            type_j = assignments[j]["assigned_type"]
            pair = None
            if type_i in LOGP_SUPPORTED_TYPES and type_j in LOGP_SUPPORTED_TYPES:
                pair = lookup_pair_with_interpolation(table, type_i, type_j)

            if pair is not None:
                value = pair["A_ij"]
                source = pair.get("source", "logp_table")
                r_ij = pair["R_ij"]
                r0 = pair["r0"]
            elif missing == "fallback":
                if solubility_values is None:
                    raise ValueError("solubility_values are required when missing='fallback'")
                value = solubility_aij(solubility_values[i], solubility_values[j], i == j)
                source = "solubility_fallback"
                r_ij = np.nan
                r0 = np.nan
            elif missing == "error":
                raise ValueError(
                    f"No logP table parameter for pair {i + 1}-{j + 1}: "
                    f"{bead_smiles[i]} ({type_i}) / {bead_smiles[j]} ({type_j})"
                )
            else:
                optimize_missing_logp_params()

            a_ij.append((i + 1, j + 1, value))
            pair_rows.append(
                LogpPairResult(
                    i=i + 1,
                    j=j + 1,
                    type_i=type_i,
                    type_j=type_j,
                    value=value,
                    source=source,
                    r_ij=r_ij,
                    r0=r0,
                ).__dict__
            )

    pd.DataFrame(pair_rows).to_csv(pair_output, index=False)
    return a_ij, assignments, pair_rows
