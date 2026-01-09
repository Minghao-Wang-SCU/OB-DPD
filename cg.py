from rdkit import Chem
from rdkit.Chem import AllChem,rdmolops 
from rdkit.Chem import Draw
from tqdm import tqdm
import auto_martini as am
from smiles2split import create_mol_atom_dir
import random

def mol2pdb(mol,name):
    # 添加氢原子
    print(mol)
    mol = mol
    mol_with_h = Chem.AddHs(mol)
    # 生成 3D 构象
    AllChem.EmbedMolecule(mol_with_h)
    # 创建 PDB 文件写入器
    writer = Chem.PDBWriter(f'{name}.pdb')
    # 将分子写入 PDB 文件
    writer.write(mol_with_h)

    # 关闭写入器
    writer.close()


def fix_aromaticity(mol):
    """
    去除分子中非环原子的芳香性标记
    """
    for atom in mol.GetAtoms():
        if not atom.IsInRing() and atom.GetIsAromatic():
            atom.SetIsAromatic(False)
    return mol

def kekulize_mol(mol):
    """
    尝试Kekulize分子，处理可能的芳香性键
    """
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        for bond in mol.GetBonds():
            print('无法凯库勒化')
            if bond.GetBondType() == Chem.BondType.AROMATIC:
                bond.SetBondType(Chem.BondType.DOUBLE)
    return mol


def get_element(mol,atom_id):

    element_symbols = {}
    for idx in atom_id:
        atom = mol.GetAtomWithIdx(idx)
        element_symbols[f'{idx}'] = atom.GetSymbol()
    return element_symbols

def get_bonds(mol,atom_id):
    bonds = {}
    for i in range(len(atom_id)):
        for j in range(i + 1, len(atom_id)):
            idx1 = atom_id[i]
            idx2 = atom_id[j]   
            bond = mol.GetBondBetweenAtoms(idx1, idx2)      
            # 获取两个原子之间的键
            if bond is not None:
                
                bonds[f'{idx1}{idx2}'] = bond.GetBondType()
            if bond is None:
                bonds[f'{idx1}{idx2}'] = None
    return bonds


def create_cg_mol(idxs,element,bond):
    # 添加原子到新分子
    new_mol = Chem.RWMol()
    atom_map = {}  # 用于记录原子在新分子中的索引
    for idx in idxs:

        new_atom_idx = new_mol.AddAtom(Chem.Atom(element[f'{idx}']))
        atom_map[idx] = new_atom_idx

    # 添加键到新分子


    for i in range(len(idxs)):
        idx1 = idxs[i]
        for j in range(i+1,len(idxs)):
            idx2 = idxs[j]
            if bond[f'{idx1}{idx2}'] is not None:
                # 获取新分子中的原子索引
                new_idx1 = atom_map[idx1]
                new_idx2 = atom_map[idx2]
                new_mol.AddBond(new_idx1, new_idx2, bond[f'{idx1}{idx2}'])

    # 转换为分子对象
    new_mol = new_mol.GetMol()

    return new_mol

def split_mol_by_cg_groups(mol, cg_groups):
    """
    根据粗粒化分组切分分子
    """
    #global cg_pdb_name
    idxs = []
    elements = []
    bonds = []
    for i in range(len(cg_groups)): #遍历珠子
        group = cg_groups[i]
        idxs.append(cg_groups[i])   #遍历每个珠子的原子
        elements.append(get_element(mol,cg_groups[i]))
        bonds.append(get_bonds(mol,cg_groups[i]))

    split_mol_list = []
    for i in range(len(cg_groups)):#遍历珠子生成新分子

        cg_mol = create_cg_mol(idxs[i],elements[i],bonds[i])
        atom_dir = create_mol_atom_dir(cg_mol)
        for ii,vv in enumerate(atom_dir):
            atom_dir[ii]['cg_id'] = i
        split_mol_list.append(atom_dir)
        #print(split_mol_list)
       
    return split_mol_list

def change_new_smiles_id(splited_atom_dir,split_id,split_smiles_id,split_smiles_cg_id):
    #split_id 分割第几段
    #split_smiles_id分割后一段中的第几个原子
    for i,v in enumerate(splited_atom_dir):
        tmp = splited_atom_dir[i]['id'] 
        if splited_atom_dir[i]['new_smiles_id'] == f'{tmp}'+'-'+f'{split_id}'+'-'+f'{split_smiles_id}':
            splited_atom_dir[i]['new_smiles_id'] = f'{tmp}'+'-'+f'{split_id}'+'-'+f'{split_smiles_id}'+'-'+f'{split_smiles_cg_id}'
    return splited_atom_dir

def generate_cg_pdb(splited_atom_dir,smiles,split_id):
    """
    从SMILES生成粗粒化后的分子并保存为SMILES文件
    """
    
    mol = Chem.MolFromSmiles(smiles)
    # 添加氢原子并嵌入分子构型
    Chem.SanitizeMol(mol)
    mol_with_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_h)
    
    # 粗粒化处理
    #mol_am, _ = am.topology.gen_molecule_smi(smiles)
    sdf = Chem.MolToMolBlock(mol)
    with Chem.SDWriter("mol.sdf") as f:
        f.write(mol)
    print(Chem.MolToSmiles(mol))
    #print(Chem.SDMolSupplier("mol.sdf")[0])
    mol_am,_= am.topology.gen_molecule_smi("O[CH2:1][CH2:2][CH2:3][CH2:4][O:5][C:6]([NH:7][CH3:8])=[O:9]")
    #mol_am = am.topology.gen_molecule_sdf("mol.sdf")
    cg = am.solver.Cg_molecule(mol_am, "MOL")

    # 获取粗粒化后的原子分组信息
    cg_groups = list(cg.atom_partitioning.values())
    unique_cg_groups = list(set(cg_groups))
    atom_groups = [[] for _ in unique_cg_groups]
    for atom_id, group_id in cg.atom_partitioning.items():
        #print('atom_id',atom_id)
        #print('group_id',group_id)
        splited_atom_dir = change_new_smiles_id(splited_atom_dir,split_id,atom_id,group_id)
        atom_groups[unique_cg_groups.index(group_id)].append(atom_id)
    #print('--'*20)
    #print('splited_atom_dir',splited_atom_dir)
    # 去除芳香性并处理双键
    nmol = fix_aromaticity(mol_with_h)
    nmol = kekulize_mol(nmol)

    # 按粗粒化后的原子分组切分分子保存为pdb
    split_mol_list = split_mol_by_cg_groups(nmol, atom_groups)
    #print('nmol',nmol)
    #print('atom_groups',atom_groups)
    return splited_atom_dir,split_mol_list



def split_smiles2splited_atom_dir(splited_atom_dir,split_smiles,cg_id):
    #输入
    #splited_atom_dir 包含整个聚合物原子，不同片段用cg_id区分
    #split_smiles
    #split_smiles正在处理的cg_id

    for i,v in enumerate(splited_atom_dir):
        pass

    pass


def cg(splited_atom_dir,smiles_list,mols):
    #输入 
    # splited_atom_dir 包含整个聚合物原子，不同片段用cg_id区分
    # smiles_list 不同cg_id对应的smiles
    for i in range(len(smiles_list)):
        smiles = smiles_list[i]
        split_id = i
        print(splited_atom_dir)
        splited_atom_dir,split_mol_list = generate_cg_pdb(splited_atom_dir,smiles,split_id)

    return splited_atom_dir
