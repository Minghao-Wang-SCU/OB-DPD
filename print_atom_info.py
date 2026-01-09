from rdkit import Chem
def print_atom_info(mol):
    print("Molecule Information:")
    print("\nAtom Details:")
    print("Idx | Symbol | AtomicNum | Degree | Valence | FormalCharge | NumExplicitHs | NumImplicitHs | IsAromatic")
    print("----|--------|-----------|--------|---------|--------------|---------------|---------------|------------")
    for idx, atom in enumerate(mol.GetAtoms()):
        print(f"{idx:3} | {atom.GetSymbol():6} | {atom.GetAtomicNum():9} | {atom.GetDegree():6} | {atom.GetExplicitValence():7} | {atom.GetFormalCharge():12} | {atom.GetNumExplicitHs():13} | {atom.GetNumImplicitHs():13} | {atom.GetIsAromatic():9}")