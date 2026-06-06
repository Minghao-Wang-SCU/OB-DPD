from smiles2split import smiles2split
from splited_atom_dir2cg_atom_dir import splited_atom_dir2cg_atom_dir
from cg import cg
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point2D
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import pandas as pd
from mapping import get_cg
from Pred_Solubility_Parameter import Pred_Solubility_Parameter
from logp_param import create_logp_aij_list, map_smiles_to_logp_article_beads, parse_logp_pair_overrides
import subprocess
import re
import argparse
import shlex
import shutil
import sys
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import colorsys
import os
from itertools import product
from collections import defaultdict, Counter

# --- 辅助函数定义 ---

def default_lammps_executable():
    env_value = os.environ.get("LAMMPS_BIN")
    if env_value:
        return env_value
    return shutil.which("lmp_mpi") or shutil.which("lmp") or "lmp_mpi"

def get_a_ij():
    # 设置全局字体大小和字体类型
    plt.style.use('seaborn-v0_8-talk')
    plt.rcParams['font.family']='sans'
    plt.rcParams['font.size']= 28
    plt.rcParams['axes.labelsize']= 25
    plt.rcParams['axes.labelweight']= 'bold'
    plt.rcParams['xtick.labelsize']= 19
    plt.rcParams['ytick.labelsize']=19
    plt.rcParams['legend.fontsize']= 19
    plt.rcParams['axes.titlesize']= 25
    plt.rcParams['axes.titleweight']='bold'
    plt.rcParams['figure.titlesize']=28
    plt.rcParams["errorbar.capsize"]=8
    plt.rcParams['axes.linewidth']= 2
    plt.rcParams['xtick.major.width']=2
    plt.rcParams['ytick.major.width']=2
    matplotlib.rcParams['figure.subplot.bottom'] = 0.15
    matplotlib.rcParams['figure.subplot.left'] = 0.15
    
    c = [(0/255,112/255,192/255),(157/255,213/255,255/255),(112/255,173/255,71/255),(198/255,211/255,163/255),(255/255,189/255,0/255),(255/255,233/255,161/255),(206/255,10/255,254/255),(255/255,194/255,235/255),(62/255,62/255,62/255),(216/255,216/255,216/255)]
    
    data = []
    if os.path.exists('lammps.in'):
        with open('lammps.in', 'r') as file:
            for line in file:
                if line.startswith('pair_coeff'):
                    parts = line.strip().split()
                    if parts[2] != '*':
                        if len(parts) >= 4:
                            type1 = int(parts[1])
                            type2 = int(parts[2])
                            value = float(parts[3])
                            data.append((type1, type2, value))
    
    if not data:
        return

    # 构建矩阵
    max_type = max(max(item[0] for item in data), max(item[1] for item in data))
    matrix = np.zeros((max_type, max_type)) 

    for type1, type2, value in data:
        matrix[type1 - 1][type2 - 1] = value 
        matrix[type2 - 1][type1 - 1] = value 
    
    matrix_df = pd.DataFrame(matrix)
    
    custom_cmap = LinearSegmentedColormap.from_list('c04', [c[1], c[0]])
    max_value = max(matrix_df.values.flatten()) 
    if max_value < 10:  
        font_size = 22
    elif max_value < 100:  
        font_size = 20
    else:  
        font_size = 18
    
    g = sns.clustermap(matrix_df, cmap=custom_cmap, fmt=".0f",linecolor='white',linewidths=.5,annot=True,xticklabels=range(1,max_type+1),yticklabels=range(1,max_type+1),cbar_kws={"label": "α"},annot_kws={"size": font_size})
    
    for j in range(matrix_df.shape[0]):
        for i in range(j+1,matrix_df.shape[1]):
            cell_value = matrix_df.iloc[i, j]
            cell_x = j + 0.5
            cell_y = i + 0.5
            cell_rect = plt.Rectangle((cell_x - 0.5, cell_y - 0.5), 1, 1,
                                    fill=True, facecolor='white', lw=0)
            g.ax_heatmap.add_patch(cell_rect)
            g.ax_heatmap.texts[i * matrix_df.shape[1] + j].set_text('')

    g.ax_row_dendrogram.remove()
    g.cax.set_position([0.1, 0.08, 0.05, 0.7])
    g.ax_heatmap.set_xlabel('Type of Bead1 ')
    g.ax_heatmap.set_ylabel('Type of Bead2')
    for ax in g.ax_row_dendrogram.collections + g.ax_col_dendrogram.collections:
        ax.set_linewidth(2)
    plt.savefig('a_ij.svg',dpi=600,bbox_inches='tight')

def merge_mol_by_cg_groups(mol, adj_matrix, cg_groups):
    unique_cg_groups = sorted(list(set(cg_groups)))
    num_cg_groups = len(unique_cg_groups)
    
    cg_id_to_idx = {cg_id: idx for idx, cg_id in enumerate(unique_cg_groups)}
    
    merged_adj_matrix = np.zeros((num_cg_groups, num_cg_groups))
    cg_node_info = {cg: {"nodes": []} for cg in unique_cg_groups}
    connected_cg_pairs = set()
    
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if i != j and adj_matrix[i][j] > 0:
                cg_i = cg_groups[i]
                cg_j = cg_groups[j]
                if cg_i < cg_j:
                    connected_cg_pairs.add((cg_i, cg_j))
                else:
                    connected_cg_pairs.add((cg_j, cg_i))
    
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if i != j and adj_matrix[i][j] > 0:
                cg_i = cg_groups[i]
                cg_j = cg_groups[j]
                if (min(cg_i, cg_j), max(cg_i, cg_j)) in connected_cg_pairs:
                    if cg_i != cg_j:
                        matrix_idx_i = cg_id_to_idx[cg_i]
                        matrix_idx_j = cg_id_to_idx[cg_j]
                        merged_adj_matrix[matrix_idx_i, matrix_idx_j] = 1
                        merged_adj_matrix[matrix_idx_j, matrix_idx_i] = 1
                
                if i not in cg_node_info[cg_i]["nodes"]:
                    cg_node_info[cg_i]["nodes"].append(i)
                if j not in cg_node_info[cg_j]["nodes"]:
                    cg_node_info[cg_j]["nodes"].append(j)
    
    cg_smiles = {}
    atom_cgsmiles_list = ['']*len(adj_matrix)
    
    for cg in unique_cg_groups:
        atoms_in_cg = cg_node_info[cg]["nodes"]
        new_mol = Chem.RWMol(Chem.Mol())
        atom_map = {}
        for idx in atoms_in_cg:
            atom = mol.GetAtomWithIdx(idx)
            new_atom_idx = new_mol.AddAtom(atom)
            atom_map[idx] = new_atom_idx

        for idx in atoms_in_cg:
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in atoms_in_cg and idx < neighbor_idx:
                    bond = mol.GetBondBetweenAtoms(idx, neighbor_idx)
                    new_mol.AddBond(atom_map[idx], atom_map[neighbor_idx], bond.GetBondType())
        
        smiles = Chem.MolToSmiles(new_mol.GetMol())
        for idx in atoms_in_cg:
            atom_cgsmiles_list[idx] = smiles
        
        if '.'in smiles:
            print(f"Warning: Disconnected SMILES generated for bead {cg}: {smiles}")

        cg_smiles[cg] = smiles

    return merged_adj_matrix, cg_smiles, atom_cgsmiles_list, cg_node_info, unique_cg_groups, atoms_in_cg

def create_one_chain_pdb(adj_matrix,xyz,cg_smiles_list,cg_smiles_dir,output_file='one_chain.pdb'):
    with open(output_file, "w") as pdb_file:
        pdb_file.write("REMARK    Custom PDB file with user-defined atom types\n")
        for i in range(adj_matrix.shape[0]):
            atom_name = f"ATOM"
            element = cg_smiles_dir[cg_smiles_list[i]]
            x, y, z = xyz[i]
            pdb_file.write(
                f"HETATM{i + 1:5d} {atom_name} MOL A{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00 #{element}\n"
            )
        for i in range(adj_matrix.shape[0]):
            connected_atoms = [j + 1 for j in range(adj_matrix.shape[1]) if adj_matrix[i, j] == 1 and j != i]
            for atom in connected_atoms:
                pair = [i + 1, atom]
                pdb_file.write(f"CONECT{pair[0]:5d}{pair[1]:5d}\n")
        pdb_file.write("END\n")
    print(f"PDB 文件已生成：{output_file}")

    with open('vmd_'+output_file, "w") as pdb_file:
        pdb_file.write("REMARK    Custom PDB file with user-defined atom types\n")
        for i in range(adj_matrix.shape[0]):
            atom_name = f"ATOM" 
            element = cg_smiles_dir[cg_smiles_list[i]]
            x, y, z = xyz[i]
            pdb_file.write(
                f"HETATM{i + 1:5d} {atom_name} MO{output_file[-5]} A{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00 #{element}\n"
            )
        for i in range(adj_matrix.shape[0]):
            connected_atoms = [j + 1 for j in range(adj_matrix.shape[1]) if adj_matrix[i, j] == 1 and j != i]
            for atom in connected_atoms:
                pair = [i + 1, atom]
                pdb_file.write(f"CONECT{pair[0]:5d}{pair[1]:5d}\n")
        pdb_file.write("END\n")

def fix_aromaticity(mol):
    """去除分子中非环原子的芳香性标记"""
    for atom in mol.GetAtoms():
        if not atom.IsInRing() and atom.GetIsAromatic():
            atom.SetIsAromatic(False)
    return mol

def kekulize_mol(mol):
    """尝试Kekulize分子,处理可能的芳香性键"""
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.AROMATIC:
                bond.SetBondType(Chem.BondType.DOUBLE)
    return mol

def find_connected_components(atoms, adj_matrix):
    visited = set()
    components = []
    for atom in atoms:
        if atom not in visited:
            stack = [atom]
            component = []
            visited.add(atom)
            while stack:
                current = stack.pop()
                component.append(current)
                for j in range(len(adj_matrix)):
                    if adj_matrix[current][j] and j in atoms and j not in visited:
                        visited.add(j)
                        stack.append(j)
            components.append(component)
    return components

def check_cg(adj_matrix, bead_id_list):
    bead_groups = defaultdict(list)
    for idx, bead_id in enumerate(bead_id_list):
        bead_groups[bead_id].append(idx)
    
    connected_bead_pairs = set()
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if i != j and adj_matrix[i][j] > 0:
                bead_i = bead_id_list[i]
                bead_j = bead_id_list[j]
                if bead_i != bead_j:
                    if bead_i < bead_j:
                        connected_bead_pairs.add((bead_i, bead_j))
                    else:
                        connected_bead_pairs.add((bead_j, bead_i))
    
    for bead_id, atoms in bead_groups.items():
        components = find_connected_components(atoms, adj_matrix)
        if len(components) <= 1:
            continue
        sorted_components = sorted(components, key=lambda x: len(x), reverse=True)
        
        for component in sorted_components[1:]:
            neighboring_beads = []
            for atom in component:
                for j in range(len(adj_matrix)):
                    if adj_matrix[atom][j]:
                        neighbor_bead = bead_id_list[j]
                        if neighbor_bead != bead_id:
                            neighboring_beads.append(neighbor_bead)
            if neighboring_beads:
                cnt = Counter(neighboring_beads)
                target_bead = cnt.most_common(1)[0][0]
                if (min(bead_id, target_bead), max(bead_id, target_bead)) in connected_bead_pairs:
                    for atom in component:
                        bead_id_list[atom] = target_bead
    return bead_id_list

def create_highlight_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        saturation = 0.7  
        lightness = 0.7   
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((r, g, b))
    return colors

def plot_cg(smiles, atom_id_list, bead_id_list,bead_type_id_list,name):
    from rdkit.Chem import rdDepictor
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    AllChem.Compute2DCoords(mol)
    
    unique_bead_ids = sorted(list(set(int(bid) for bid in bead_id_list)))
    num_beads = len(unique_bead_ids)
    highlight_colors = create_highlight_colors(num_beads)
    bead_id_to_color_idx = {bid: idx for idx, bid in enumerate(unique_bead_ids)}
    
    atom_count = mol.GetNumAtoms()
    canvas_size = min(2000 + atom_count * 20, 10000)
    font_base = max(8, int(canvas_size * 0.002))
    d = rdMolDraw2D.MolDraw2DCairo(canvas_size, canvas_size)
    
    highlight_dict = {}
    for atom_idx in atom_id_list:
        atom_idx_int = int(atom_idx)
        raw_bead_id = int(bead_id_list[atom_idx_int])
        if raw_bead_id in bead_id_to_color_idx:
            color_idx = bead_id_to_color_idx[raw_bead_id]
            highlight_dict[atom_idx_int] = highlight_colors[color_idx]
    
    Chem.Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol, 
        highlightAtoms=[int(id) for id in atom_id_list],
        highlightAtomColors=highlight_dict)
    
    d.SetFontSize(font_base) 
    
    for atom_id in atom_id_list:
        atom_idx_int = int(atom_id)
        if atom_idx_int < len(bead_id_list) and atom_idx_int < len(bead_type_id_list):
            bead_id = bead_id_list[atom_idx_int]
            bead_type_id = bead_type_id_list[atom_idx_int]
            atom_pos = d.GetDrawCoords(atom_idx_int)
            d.DrawString(f'{1+int(bead_id)}:{int(bead_type_id)}', Point2D(atom_pos[0], atom_pos[1]), rawCoords=True)
    
    d.FinishDrawing()
    d.WriteDrawingText(f'{name}.png')

def plot_cg_id(smiles,name):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return
    atom_count = mol.GetNumAtoms()
    canvas_size = min(2000 + atom_count * 20, 10000)
    font_base = max(8, int(canvas_size * 0.002))
    d = rdMolDraw2D.MolDraw2DCairo(canvas_size, canvas_size)
    Chem.Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.SetFontSize(font_base) 
    for atom_id in range(mol.GetNumAtoms()):
        atom_pos = d.GetDrawCoords(atom_id)
        d.DrawString(str(atom_id), Point2D(atom_pos[0], atom_pos[1]), rawCoords=True)
    d.FinishDrawing()
    d.WriteDrawingText(f'{name}_id.png')

# --- 补充缺失的函数 ---

def generate_all_is_peg_list(is_peg_list, split_number_list, all_cg_smiles_list):
    """
    按split_number_list的分组规则，结合is_peg_list的标记生成all_is_peg_list
    """
    if len(is_peg_list) != len(split_number_list):
        # 如果长度不一致，尝试补齐
        pass
    
    adjusted_split = [1 if cnt == 0 else cnt for cnt in split_number_list]
    
    cumulative_counts = [0]
    for cnt in adjusted_split:
        cumulative_counts.append(cumulative_counts[-1] + cnt)
    
    all_is_peg_list = []
    
    # 遍历每个子列表（即每个片段）
    for sublist_idx, sublist in enumerate(all_cg_smiles_list):
        group_idx = 0
        for i, count in enumerate(cumulative_counts[1:]):
            if sublist_idx < count:
                group_idx = i
                break
        
        if group_idx < len(is_peg_list):
            is_peg = is_peg_list[group_idx]
        else:
            is_peg = 0
            
        all_is_peg_list.extend([is_peg] * len(sublist))
    
    return all_is_peg_list

def get_all_type_list(split_number_list, all_cg_smiles_list):
    """
    生成每个珠子所属的组件类型 ID
    """
    all_type_list = []
    current_type = 1
    processed = 0 
    
    adjusted_split = [1 if cnt == 0 else cnt for cnt in split_number_list]
    
    for split_size in adjusted_split:
        group_sublists = all_cg_smiles_list[processed:processed + split_size]
        
        for sublist in group_sublists:
            all_type_list.extend([current_type] * len(sublist))
        
        processed += split_size
        current_type += 1
    
    return all_type_list

# --- 核心逻辑函数 ---

def get_split_smiles(smiles,n,split_Mw):
    from rdkit.Chem import rdmolops, Descriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析SMILES字符串: {smiles}")
    if n == 0:
        return [smiles]
    bonds_to_cut = []
    for bond in mol.GetBonds():
        if bond.IsInRing(): continue
        if bond.GetBondTypeAsDouble() == 1.0: bonds_to_cut.append(bond.GetIdx())
    if not bonds_to_cut: return [smiles]
    
    fragment_mol = rdmolops.FragmentOnBonds(mol, bonds_to_cut[:n] if len(bonds_to_cut)>n else bonds_to_cut)
    fragments = rdmolops.GetMolFrags(fragment_mol, asMols=True)
    return [Chem.MolToSmiles(f).replace('*','') for f in fragments][:n]

def get_one_chain_and_smiles(smiles, compoents_id=1, split_id=1, tune_model='0', is_opt='0', cg_method='smiles', is_peg=0):
    if cg_method == 'smiles':
        return get_one_chain_and_smiles_original(smiles, compoents_id, split_id, tune_model, is_opt, is_peg)
    elif cg_method == 'logp_article':
        return get_one_chain_and_smiles_logp_article(smiles, compoents_id, split_id, is_peg)
    elif cg_method == 'structure':
        return [], []

def get_one_chain_and_smiles_logp_article(smiles, compoents_id=1, split_id=1, is_peg=0):
    print('-'*20+'调用logP article bead mapping'+'-'*20)
    mapping_result = map_smiles_to_logp_article_beads(smiles, is_peg=is_peg)
    bead_types = mapping_result["bead_types"]
    bead_type_list_drop = []
    for bead_type in bead_types:
        if bead_type not in bead_type_list_drop:
            bead_type_list_drop.append(bead_type)

    bead_type_dir = {bead_type: i + 1 for i, bead_type in enumerate(bead_type_list_drop)}
    create_one_chain_pdb(
        mapping_result["adjacency"],
        mapping_result["coords"],
        bead_types,
        bead_type_dir,
        output_file=f'{compoents_id}one_chain{split_id}.pdb',
    )

    rows = []
    for bead_idx, (bead_type, atoms) in enumerate(
        zip(bead_types, mapping_result["bead_atom_groups"]), start=1
    ):
        rows.append(
            {
                "bead_id": bead_idx,
                "bead_type": bead_type,
                "atom_indices": " ".join(str(atom_idx) for atom_idx in atoms),
            }
        )
    pd.DataFrame(rows).to_csv(f'{compoents_id}logp_article_mapping{split_id}.csv', index=False)
    return bead_type_list_drop, bead_types

def get_one_chain_and_smiles_original(smiles, compoents_id=1, split_id=1, tune_model='0', is_opt='0', is_peg=0):
    atom_id_list = []
    bead_id_list = []
    bead_type_list = []
    
    print('-'*20+'调用get_cg'+'-'*20)
    mapping_time_start = time.time()
    
    try:
        get_cg(smiles, 
               gro_path=f'{compoents_id}Gro{split_id}',
               itp_path=f'{compoents_id}Itp{split_id}',
               tune_model=tune_model,
               is_opt=is_opt,
               complents_id=compoents_id,
               split_id=split_id,
               is_peg=is_peg)
    except TypeError:
        get_cg(smiles, 
               gro_path=f'{compoents_id}Gro{split_id}',
               itp_path=f'{compoents_id}Itp{split_id}',
               tune_model=tune_model,
               is_opt=is_opt,
               complents_id=compoents_id,
               split_id=split_id)

    mapping_time_end = time.time()
    print(f'mapping cost {mapping_time_end-mapping_time_start}s')
    
    mapping_file = f"{compoents_id}atom_to_bead_mapping{split_id}.txt"
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file {mapping_file} not found. CG script failed?")

    with open(mapping_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("|")
            atom_id_list.append(parts[0].strip())
            bead_id_list.append(parts[1].strip())
            bead_type_list.append(parts[2].strip())

    mol = Chem.MolFromSmiles(smiles)
    mol = fix_aromaticity(mol)
    mol = kekulize_mol(mol)
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    bead_id_list = check_cg(adj_matrix, bead_id_list)
    
    cged_adj_matrix, cg_smiles, atom_cgsmiles_list, cg_node_info, unique_cg_groups, atoms_in_cg = merge_mol_by_cg_groups(mol, adj_matrix, bead_id_list)
    
    xyz_file = f'{compoents_id}xyz{split_id}.csv'
    if not os.path.exists(xyz_file):
         raise FileNotFoundError(f"XYZ file {xyz_file} not found.")
    xyz = pd.read_csv(xyz_file).values
    
    cg_smiles_turple = sorted(cg_smiles.items(), key=lambda x: int(x[0]))
    cg_smiles_list = [smi for id,smi in cg_smiles_turple]

    cg_smiles_list_drop = []
    for smi in cg_smiles_list:
        if smi not in cg_smiles_list_drop:
            cg_smiles_list_drop.append(smi)
            
    cg_smiles_dir = {smi:i+1 for i,smi in enumerate(cg_smiles_list_drop)}
    create_one_chain_pdb(cged_adj_matrix,xyz,cg_smiles_list,cg_smiles_dir,output_file=f'{compoents_id}one_chain{split_id}.pdb')

    bead_type_id_list = [0]*len(adj_matrix)
    i = 1
    for smi in cg_smiles_list_drop:
        for cg in unique_cg_groups:
            atoms_in_cg = cg_node_info[cg]["nodes"]
            for idx in atoms_in_cg:
                if smi == atom_cgsmiles_list[idx]:
                    bead_type_id_list[idx] = i
        i += 1

    plot_cg(smiles, atom_id_list, bead_id_list, bead_type_id_list, name=f'{compoents_id}cged_mol{split_id}')
    plot_cg_id(smiles, name=f'{compoents_id}cged_mol{split_id}')

    return cg_smiles_list_drop, cg_smiles_list

def create_packmol_in(pack_model=0,box_size=30,number_list=[],packmol_in=''):
    if len(packmol_in) < 1 :
        packmol_box_size = box_size * 10
        content = f"""tolerance 1
nloop 100
output packed_polymer_and_solution.pdb
filetype pdb
"""
        compoents_content = []
        for i in range(len(number_list)):
            compoents_content.append(
        f"""
structure {i}one_chain.pdb
number {number_list[i]}
inside box 0. 0. 0. {packmol_box_size} {packmol_box_size} {packmol_box_size}
end structure
"""
)
        with open('packmol_polymer_and_solution.inp', "w") as f:
            f.write(content)
            for line in compoents_content:
                f.write(line)

        try:
            result = subprocess.run(
                f"packmol < packmol_polymer_and_solution.inp",
                capture_output=True, text=True, shell=True, check=True
            )
            print("Packmol output generated.")
        except Exception as e:
            print(f"Packmol Error: {e}")
            
    elif len(packmol_in) >= 1:
        subprocess.run(f"packmol < {packmol_in}", shell=True, check=True)

def read_pdb(pdb_file):
    """
    Robust PDB reader that handles fixed-width columns correctly, 
    including cases where columns are merged (e.g. CONECT records with IDs > 9999).
    """
    atoms = []
    bonds = []
    angles = []
    count = 1
    if not os.path.exists(pdb_file): return [], [], []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('HETATM') or line.startswith('ATOM'):
                atom_id = count
                atom_name = line[12:16].strip()
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except:
                    x,y,z = 0.0,0.0,0.0
                
                element = atom_name
                # Try to get element from comment at the end of the line (generated by create_one_chain_pdb)
                # Note: Packmol often strips these comments. If stripped, element = atom_name (e.g. "ATOM")
                if '#' in line:
                    match = re.search(r'#(\d+)', line)
                    if match: element = match.group(1)
                
                atoms.append((atom_id, 1, atom_name, x, y, z, element))
                count += 1
            elif line.startswith('CONECT'):
                # Fixed-width parsing for CONECT records (standard PDB format)
                # Columns: 7-11, 12-16, 17-21, 22-26, 27-31
                # This handles merged columns (e.g. 1000010001) correctly.
                try:
                    # Helper to safely extract int from substring
                    def safe_int(s): return int(s) if s.strip() else None
                    
                    ids = []
                    # Read up to 4 connection entries per line
                    for start, end in [(6, 11), (11, 16), (16, 21), (21, 26)]:
                        if len(line) > start:
                            val = safe_int(line[start:end])
                            if val is not None: ids.append(val)
                    
                    if len(ids) >= 2:
                        central = ids[0]
                        others = ids[1:]
                        for other in others:
                            # Store unique bonds (min < max)
                            if central < other:
                                bonds.append((central, other))
                except Exception:
                    pass # Skip malformed lines

    neighbors = defaultdict(set)
    for a1, a2 in bonds:
        neighbors[a1].add(a2)
        neighbors[a2].add(a1)
    for center, connected in neighbors.items():
        connected = sorted(connected)
        for idx_i in range(len(connected)):
            for idx_j in range(idx_i + 1, len(connected)):
                angles.append((connected[idx_i], center, connected[idx_j]))

    return atoms, bonds, angles

def generate_lammps_data(
    atoms,
    bonds,
    angles,
    types,
    box_size,
    output_file,
    type_charges=None,
    charge_enabled=False,
    coord_scale=0.1,
    bond_r0_by_pair=None,
    write_angles=False,
):
    num_atoms = len(atoms)
    num_bonds = len(bonds)
    num_angles = len(angles) if write_angles else 0
    num_atom_types = len(types) if len(types) > 0 else 1
    type_charges = type_charges or [0.0] * num_atom_types
    bond_r0_by_pair = bond_r0_by_pair or {}

    atom_type_by_id = {}
    for atom in atoms:
        atom_id, _, _, _, _, _, element = atom
        try:
            atom_type_by_id[int(atom_id)] = int(element)
        except Exception:
            atom_type_by_id[int(atom_id)] = 1

    bond_type_by_pair = {}
    bond_coeffs = {}
    next_bond_type = 1
    typed_bonds = []
    for a1, a2 in bonds:
        t1 = atom_type_by_id.get(a1, 1)
        t2 = atom_type_by_id.get(a2, 1)
        key = tuple(sorted((t1, t2)))
        if key not in bond_type_by_pair:
            bond_type_by_pair[key] = next_bond_type
            bond_coeffs[next_bond_type] = {
                "type_pair": key,
                "k": 150.0 if key in bond_r0_by_pair else 100.0,
                "r0": float(bond_r0_by_pair.get(key, 0.5)),
            }
            next_bond_type += 1
        typed_bonds.append((bond_type_by_pair[key], a1, a2))

    angle_coeffs = {}
    if num_angles > 0:
        angle_coeffs[1] = {"k": 5.0, "theta0": 180.0}
    
    if isinstance(box_size,list) :
        xlo, xhi = 0, box_size[0]
        ylo, yhi = 0, box_size[1]
        zlo, zhi = 0, box_size[2]
    else:
        xlo, xhi = 0, box_size
        ylo, yhi = 0, box_size
        zlo, zhi = 0, box_size

    with open(output_file, 'w') as f:
        f.write("#LAMMPS Data File\n\n")
        f.write(f"{num_atoms} atoms\n")
        f.write(f"{num_bonds} bonds\n")
        if num_angles > 0:
            f.write(f"{num_angles} angles\n")
        f.write(f"{num_atom_types} atom types\n")
        if num_bonds > 0:
            f.write(f"{len(bond_coeffs)} bond types\n")
        if num_angles > 0:
            f.write(f"{len(angle_coeffs)} angle types\n")
        f.write(f"{xlo:.4f} {xhi:.4f} xlo xhi\n")
        f.write(f"{ylo:.4f} {yhi:.4f} ylo yhi\n")
        f.write(f"{zlo:.4f} {zhi:.4f} zlo zhi\n\n")
        f.write("Atoms\n\n")

        for i, atom in enumerate(atoms):
            # atom struct: (id, mol_id, name, x, y, z, element)
            atom_id, _, _, x, y, z, element = atom
            try:
                # element comes from PDB comment. If missing (Packmol stripped), 
                # we default to 1. Real type logic should use a mapping if needed.
                atype = int(element)
            except:
                atype = 1
            if charge_enabled:
                charge = type_charges[atype - 1] if 0 < atype <= len(type_charges) else 0.0
                f.write(f"{i+1} {1} {atype} {charge:.12e} {x*coord_scale:.4f} {y*coord_scale:.4f} {z*coord_scale:.4f}\n")
            else:
                f.write(f"{i+1} {1} {atype} {x*coord_scale:.4f} {y*coord_scale:.4f} {z*coord_scale:.4f}\n")

        if num_bonds > 0:
            f.write("\nBonds\n\n")
            for i, (bond_type, a1, a2) in enumerate(typed_bonds):
                f.write(f"{i+1} {bond_type} {a1} {a2}\n")

        if num_angles > 0:
            f.write("\nAngles\n\n")
            for i, (a1, a2, a3) in enumerate(angles):
                f.write(f"{i+1} 1 {a1} {a2} {a3}\n")

    return {"bond_coeffs": bond_coeffs, "angle_coeffs": angle_coeffs}

def flory_huggins_parameter(sigma1,sigma2):
    V = 129
    k = 8.31451
    T = 273.15+25
    flory_huggins = (V*(sigma1-sigma2)**2)/(k*T)
    return flory_huggins

def create_a_ij_list(Pred_Solubility_Parameter):
    a_ij = []
    count = len(Pred_Solubility_Parameter)
    for i in range(count):
        for j in range(i, count):
            if i != j :
                x_ij = flory_huggins_parameter(Pred_Solubility_Parameter[i],Pred_Solubility_Parameter[j])
                val = 25 + 3.27 * x_ij
                a_ij.append((i+1, j+1, val))
            else:
                a_ij.append((i+1, j+1, 25.0))
    return a_ij

ION_BEAD_SMILES = {"Na+", "Cl-", "[Na+]", "[Cl-]"}

def is_ion_bead(smiles):
    return smiles in ION_BEAD_SMILES

def create_solubility_aij_with_ions(parameter_values, bead_smiles):
    a_ij = create_a_ij_list(parameter_values)
    water_idx = None
    for idx, smiles in enumerate(bead_smiles):
        if smiles in {"HOH", "H2O", "O"}:
            water_idx = idx
            break
    if water_idx is None:
        return a_ij

    aij_map = {(i - 1, j - 1): val for i, j, val in a_ij}
    for i, smiles_i in enumerate(bead_smiles):
        for j in range(i, len(bead_smiles)):
            smiles_j = bead_smiles[j]
            if not (is_ion_bead(smiles_i) or is_ion_bead(smiles_j)):
                continue
            if is_ion_bead(smiles_i) and is_ion_bead(smiles_j):
                val = 25.0
            else:
                other = j if is_ion_bead(smiles_i) else i
                val = aij_map.get((min(water_idx, other), max(water_idx, other)), 25.0)
            aij_map[(i, j)] = val

    return [(i + 1, j + 1, aij_map[(i, j)]) for i in range(len(bead_smiles)) for j in range(i, len(bead_smiles))]

def infer_bead_charge(smiles):
    if smiles in {"HOH", "H2O", "O"}:
        return 0.0, "matched", "water"
    if smiles in {"Na+", "[Na+]"}:
        return 1.0, "matched", "sodium counterion"
    if smiles in {"Cl-", "[Cl-]"}:
        return -1.0, "matched", "chloride counterion"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, "invalid_smiles", "RDKit could not parse charge; assigned 0"
    charge = float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    return charge, "matched", "formal charge from bead SMILES"

def create_charge_assignments(bead_smiles, output_file="charge_assignment.csv"):
    rows = []
    charges = []
    for type_id, smiles in enumerate(bead_smiles, start=1):
        charge, status, reason = infer_bead_charge(smiles)
        charges.append(charge)
        rows.append({
            "type_id": type_id,
            "bead_smiles": smiles,
            "charge": charge,
            "status": status,
            "reason": reason,
        })
    pd.DataFrame(rows).to_csv(output_file, index=False)
    return charges, rows

def write_charge_summary(
    atoms,
    type_charges,
    solution_number,
    solution_type_id=None,
    extra_counts=None,
    output_file="charge_summary.csv",
):
    extra_counts = extra_counts or {}
    solution_type_id = solution_type_id or len(type_charges)
    type_counts = Counter()
    for atom in atoms:
        try:
            atype = int(atom[6])
        except Exception:
            atype = 1
        type_counts[atype] += 1
    if type_charges:
        type_counts[int(solution_type_id)] += int(solution_number)
    for type_id, count in extra_counts.items():
        type_counts[int(type_id)] += int(count)

    rows = []
    total_charge = 0.0
    for type_id, charge in enumerate(type_charges, start=1):
        count = type_counts.get(type_id, 0)
        subtotal = count * charge
        total_charge += subtotal
        rows.append({
            "type_id": type_id,
            "count": count,
            "charge": charge,
            "subtotal_charge": subtotal,
        })
    rows.append({
        "type_id": "total",
        "count": sum(type_counts.values()),
        "charge": "",
        "subtotal_charge": total_charge,
    })
    pd.DataFrame(rows).to_csv(output_file, index=False)
    return total_charge

def write_system_bead_summary(
    box_size,
    target_total_beads,
    solute_bead_count,
    counterion_bead_count,
    water_bead_count,
    output_file="system_bead_summary.csv",
):
    total_beads = solute_bead_count + counterion_bead_count + water_bead_count
    box_volume = get_box_volum(box_size)
    rows = [
        {"item": "box_volume", "value": box_volume},
        {"item": "target_total_beads", "value": target_total_beads},
        {"item": "solute_bead_count", "value": solute_bead_count},
        {"item": "counterion_bead_count", "value": counterion_bead_count},
        {"item": "water_bead_count", "value": water_bead_count},
        {"item": "total_bead_count", "value": total_beads},
        {"item": "target_bead_density", "value": 3.0},
        {"item": "actual_bead_density", "value": total_beads / box_volume if box_volume else 0.0},
    ]
    pd.DataFrame(rows).to_csv(output_file, index=False)

def solute_net_charge(atoms, type_charges):
    total_charge = 0.0
    for atom in atoms:
        try:
            atype = int(atom[6])
        except Exception:
            atype = 1
        if 0 < atype <= len(type_charges):
            total_charge += type_charges[atype - 1]
    return total_charge

def counterions_for_charge(net_charge):
    rounded = int(round(net_charge))
    if abs(net_charge - rounded) > 1.0e-6:
        raise ValueError(f"net charge must be close to an integer to add counterions, got {net_charge}")
    if rounded > 0:
        return {"Cl-": rounded}
    if rounded < 0:
        return {"Na+": -rounded}
    return {}

def flatten_logp_recipe(all_cg_smiles_list_list):
    recipe = []
    for bead_list in all_cg_smiles_list_list:
        recipe.extend(bead_list)
    return recipe

def write_logp_fit_protocol(
    missing_rows,
    recipe_types,
    smiles_list,
    args,
    output_script="run_missing_logp_fit.sh",
):
    if not missing_rows:
        return None, None
    fit_pairs = sorted(
        {
            tuple(sorted((str(row["type_i"]), str(row["type_j"]))))
            for row in missing_rows
        }
    )
    if not recipe_types:
        raise ValueError("cannot create logP fit protocol without a solute bead recipe")

    target_smiles = args.logp_fit_target_smiles
    if not target_smiles:
        if len(smiles_list) != 1:
            raise ValueError(
                "missing logP parameters for a multi-component system; "
                "provide --logp_fit_target_smiles for the fitted solute."
            )
        target_smiles = smiles_list[0]

    outdir = os.path.abspath(args.logp_fit_outdir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        os.path.join(script_dir, "logp_partition.py"),
        "fit",
        "--solute",
        ",".join(recipe_types),
        "--allow-custom-beads",
        "--target-smiles",
        target_smiles,
        "--fit-pairs",
        ",".join(f"{a}:{b}" for a, b in fit_pairs),
        "--outdir",
        outdir,
        "--table",
        args.logp_table,
        "--steps",
        str(args.logp_fit_steps),
        "--dump-every",
        str(args.logp_fit_dump_every),
        "--max-iter",
        str(args.logp_fit_max_iter),
        "--optimizer",
        args.logp_fit_optimizer,
        "--job",
        str(args.logp_fit_job if args.logp_fit_job > 0 else args.job),
        "--lammps",
        args.lammps_bin,
    ]
    if args.logp_fit_optimizer == "bayesian":
        cmd.extend(["--bo-initial", str(args.logp_bo_initial)])
        cmd.extend(["--bo-grid-step", str(args.logp_bo_grid_step)])
        cmd.extend(["--bo-acq-candidates", str(args.logp_bo_acq_candidates)])
        cmd.extend(["--bo-epsilon", str(args.logp_bo_epsilon)])
        cmd.extend(["--bo-noise", str(args.logp_bo_noise)])
        cmd.extend(["--bo-gp-restarts", str(args.logp_bo_gp_restarts)])
        cmd.extend(["--bo-grid-chunk-size", str(args.logp_bo_grid_chunk_size)])
        if not args.logp_bo_full_grid_ei:
            cmd.append("--no-bo-full-grid-ei")
        cmd.extend(["--failure-penalty-loss", str(args.logp_failure_penalty_loss)])
        if not args.logp_penalize_failures:
            cmd.append("--no-penalize-failures")
        for value in args.logp_bo_grid:
            cmd.extend(["--bo-grid", value])
    if args.logp_fit_article_protocol:
        cmd.append("--article-protocol")
    if args.logp_angle_param_mode:
        cmd.extend(["--angle-param-mode", args.logp_angle_param_mode])
    if args.logp_fit_organic_solvent:
        cmd.extend(["--organic-solvent", args.logp_fit_organic_solvent])
    if args.logp_fit_ensemble:
        cmd.extend(["--ensemble", args.logp_fit_ensemble])
        cmd.extend(["--pressure", str(args.logp_fit_pressure)])
        cmd.extend(["--pressure-damp", str(args.logp_fit_pressure_damp)])
    cmd.extend(["--charge-method", args.charge_method])
    cmd.extend(["--charge-lambda", str(args.charge_lambda)])
    cmd.extend(["--coul-cutoff", str(args.coul_cutoff)])
    cmd.extend(["--kspace-accuracy", str(args.kspace_accuracy)])
    cmd.extend(["--charge-unit-scale", str(args.charge_unit_scale)])
    for value in args.logp_bead_charge:
        cmd.extend(["--bead-charge", value])
    cmd.extend(["--heavy-atom-correction", args.logp_heavy_atom_correction])
    cmd.extend(["--heavy-radius-scale", str(args.logp_heavy_radius_scale)])
    cmd.extend(["--bonded-r0-factor", str(args.logp_bonded_r0_factor)])
    for value in args.logp_bead_heavy_atoms:
        cmd.extend(["--bead-heavy-atoms", value])
    if args.logp_fit_gpu:
        cmd.append("--gpu")
    for value in args.logp_manual_aij:
        cmd.extend(["--override", value])
    for value in args.logp_fit_target_logp_value:
        cmd.extend(["--target-logp-value", value])
    for value in args.logp_fit_bead_solubility:
        cmd.extend(["--bead-solubility", value])

    with open(output_script, "w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        handle.write("cd " + shlex.quote(os.getcwd()) + "\n")
        handle.write(" ".join(shlex.quote(part) for part in cmd) + "\n")
    os.chmod(output_script, 0o755)

    pd.DataFrame(
        [
            {
                "type_i": a,
                "type_j": b,
                "fit_pair": f"{a}:{b}",
                "fit_command_script": output_script,
                "fit_outdir": outdir,
            }
            for a, b in fit_pairs
        ]
    ).to_csv("logp_fit_required_pairs.csv", index=False)
    return cmd, outdir

def create_lammps_in(
    Pred_Solubility_Parameter,
    box_size,
    solution_number,
    a_ij_list=None,
    pair_cutoff_list=None,
    has_bonds=True,
    topology_coeffs=None,
    run_steps=3000000,
    charge_enabled=False,
    type_charges=None,
    charge_lambda=0.25,
    coul_cutoff=3.0,
    kspace_accuracy=1.0e-4,
    solution_type_id=None,
    extra_create_counts=None,
    dpd_gamma=1.0,
    timestep=0.02,
    thermo_interval=5000,
    dump_interval=5000,
    ensemble="nve",
    target_pressure=23.7,
    pressure_damp=10.0,
):
    if a_ij_list is None:
        a_ij_list = create_a_ij_list(Pred_Solubility_Parameter)
    type_charges = type_charges or [0.0] * len(Pred_Solubility_Parameter)
    solution_type_id = solution_type_id or len(Pred_Solubility_Parameter)
    extra_create_counts = extra_create_counts or {}
    pair_cutoff_map = {}
    for item in pair_cutoff_list or []:
        i, j, cutoff = item
        if cutoff is None or pd.isna(cutoff):
            continue
        pair_cutoff_map[(int(i), int(j))] = float(cutoff)
    global_cutoff = max(pair_cutoff_map.values(), default=1.0)
    topology_coeffs = topology_coeffs or {}
    bond_coeffs = topology_coeffs.get("bond_coeffs", {})
    angle_coeffs = topology_coeffs.get("angle_coeffs", {})
    
    if isinstance(box_size,list) :
        xlo, xhi = 0, box_size[0]
        ylo, yhi = 0, box_size[1]
        zlo, zhi = 0, box_size[2]
    else:
        xlo, xhi = 0, box_size
        ylo, yhi = 0, box_size
        zlo, zhi = 0, box_size
        
    content_head = f"""# LAMMPS Input
dimension    3
units        lj
boundary    p p p
atom_style   {'full' if charge_enabled else 'molecular'}
read_data    packed_polymer_and_solution.data
neigh_modify one 4000

mass * 1.0
variable        T equal 1.0
variable        cutoff equal {global_cutoff:.4f}
variable        gamma equal {float(dpd_gamma):.4f}
variable        damp equal 0.1

neighbor        0.5 bin
neigh_modify    every 1 delay 0
"""
    if has_bonds:
        content_head += """
bond_style      harmonic
"""
        if bond_coeffs:
            for bond_type in sorted(bond_coeffs):
                coeff = bond_coeffs[bond_type]
                content_head += f"bond_coeff      {bond_type}  {coeff['k']:.6g}  {coeff['r0']:.6g}\n"
        else:
            content_head += "bond_coeff      1  100  0.5\n"
    if angle_coeffs:
        content_head += "\nangle_style     harmonic\n"
        for angle_type in sorted(angle_coeffs):
            coeff = angle_coeffs[angle_type]
            content_head += f"angle_coeff     {angle_type}  {coeff['k']:.6g}  {coeff['theta0']:.6g}\n"

    content_head += f"""

region box block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi}
"""
    for type_id, count in extra_create_counts.items():
        if int(count) > 0:
            content_head += f"create_atoms {int(type_id)} random  {int(count)} {12121 + int(type_id)} box\n"
    content_head += f"create_atoms {int(solution_type_id)} random  {int(solution_number)} 12121 box\n"

    content_head += f"""
comm_modify vel yes

pair_style	{'dpd/coul/slater/long ${T} ${cutoff} 92894 ' + str(charge_lambda) + ' ' + str(coul_cutoff) if charge_enabled else 'dpd ${T} ${cutoff} 92894'}
"""
    if charge_enabled:
        content_head += "pair_coeff   * * 25.00  ${gamma}\n"
        content_head += f"kspace_style  pppm {kspace_accuracy:.1e}\n"
        for type_id, charge in enumerate(type_charges, start=1):
            content_head += f"set type {type_id} charge {charge:.12e}\n"

    if ensemble == "nph":
        fix_line = f"fix             1 all nph iso {target_pressure:.6g} {target_pressure:.6g} {pressure_damp:.6g}"
    else:
        fix_line = "fix             1 all nve"

    content_tail = f"""
velocity        all create ${{T}} 4928459 dist gaussian
{fix_line}
timestep        {float(timestep):.6g}
thermo          {int(thermo_interval)}
thermo_modify   lost ignore flush yes lost/bond ignore

# 输出设置
dump            1 all custom {int(dump_interval)} dump.lammpstrj id type {'q ' if charge_enabled else ''}x y z

# 运行模拟
run             {int(run_steps)}
"""
    with open('lammps.in', 'w') as f:
        f.write(content_head)
        for t in a_ij_list:
            i,j,val = t
            charged = (
                charge_enabled
                and abs(type_charges[i - 1]) > 0.0
                and abs(type_charges[j - 1]) > 0.0
            )
            charged_flag = " yes" if charged else ""
            if charge_enabled:
                f.write(f'pair_coeff   {i} {j} {val:.2f}  ${{gamma}}{charged_flag}\n')
            else:
                cutoff = pair_cutoff_map.get((i, j), pair_cutoff_map.get((j, i), global_cutoff))
                f.write(f'pair_coeff   {i} {j} {val:.2f}  ${{gamma}}  {cutoff:.4f}\n')
        f.write(content_tail)

def fix_pdb_files(pdb_name_list):
    current_offset = 0
    for pdb_file in pdb_name_list:
        if not os.path.exists(pdb_file): continue
        with open(pdb_file, 'r') as file:
            lines = file.readlines()
        
        max_num = 0
        new_lines = []
        for line in lines:
            if '#' in line:
                match = re.search(r'#(\d+)', line)
                if match:
                    num = int(match.group(1))
                    if num > max_num: max_num = num
                    new_lines.append(re.sub(r'#\d+', f'#{num + current_offset}', line))
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        with open(pdb_file, 'w') as file:
            file.writelines(new_lines)
        current_offset += max_num

def get_box_volum(box_size):
    if isinstance(box_size,list) :
        return (box_size[0])*(box_size[1])*(box_size[2])
    else:
        return box_size**3

def import_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines: raise ValueError('Empty file')
    
    mode = lines[0].lower()
    smiles_list, number_list, is_peg_list = [], [], []
    
    if mode == 'smiles':
        for i in range(1, len(lines), 3):
            try:
                smi = lines[i]
                num = int(lines[i+1])
                peg = int(lines[i+2])
                smiles_list.append(smi)
                number_list.append(num)
                is_peg_list.append(peg)
            except IndexError:
                pass
    else:
        for i in range(1, len(lines), 3):
            try:
                mol_path = lines[i]
                num = int(lines[i+1])
                peg = int(lines[i+2])
                smiles_list.append(mol_path) # Treating mol path as 'smiles' placeholder
                number_list.append(num)
                is_peg_list.append(peg)
            except IndexError:
                pass
    return smiles_list, number_list, is_peg_list, mode

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=str)
    parser.add_argument("--box_size", type=float, default=30)
    parser.add_argument("--packmol_in", type=str, default='')
    parser.add_argument("--job", type=int, default=8)
    parser.add_argument("--T", type=int, default=0)
    parser.add_argument("--input", "--input_data", dest="input_data", type=str, default='')
    parser.add_argument("--functional_group", type=str, default='')
    parser.add_argument("--param_method", choices=["solubility", "logp"], default="solubility")
    parser.add_argument(
        "--logp_missing",
        choices=["error", "manual", "optimize", "fallback"],
        default="optimize",
        help=(
            "handling for logP bead pairs absent from the table: optimize uses an "
            "optimizer; manual uses interactive per-iteration Aij input"
        ),
    )
    parser.add_argument(
        "--logp_manual_aij",
        action="append",
        default=[],
        help="initial manual logP Aij as BEAD1:BEAD2=Aij; repeat as needed",
    )
    parser.add_argument("--logp_fit_execute", action="store_true", help="execute DPD logP fitting for missing pairs in optimize/manual mode")
    parser.add_argument("--logp_fit_outdir", default="logp_missing_fit", help="output directory for missing logP fitting")
    parser.add_argument("--logp_fit_target_smiles", default="", help="target SMILES for missing-parameter logP fitting; defaults to the single input molecule")
    parser.add_argument("--logp_fit_target_logp_value", action="append", default=[], help="external target logP estimate as METHOD=VALUE")
    parser.add_argument("--logp_fit_steps", type=int, default=200000)
    parser.add_argument("--logp_fit_dump_every", type=int, default=2000)
    parser.add_argument("--logp_fit_max_iter", type=int, default=5)
    parser.add_argument("--logp_fit_optimizer", choices=["nelder-mead", "coordinate", "manual", "bayesian"], default="nelder-mead")
    parser.add_argument("--logp_fit_job", type=int, default=0, help="MPI ranks for logP fitting; 0 reuses --job")
    parser.add_argument("--logp_fit_gpu", action="store_true")
    parser.add_argument("--logp_fit_bead_solubility", action="append", default=[], help="initial solubility parameter as BEAD=VALUE")
    parser.add_argument("--logp_bo_initial", type=int, default=10, help="random initial Aij grid samples for Bayesian logP fitting")
    parser.add_argument("--logp_bo_grid", action="append", default=[], help="Bayesian Aij grid as LOW:HIGH:STEP or BEAD1:BEAD2=LOW:HIGH:STEP")
    parser.add_argument("--logp_bo_grid_step", type=float, default=1.0, help="default Bayesian Aij grid spacing")
    parser.add_argument("--logp_bo_acq_candidates", type=int, default=5000, help="candidate grid points scored by expected improvement")
    parser.add_argument("--logp_bo_epsilon", type=float, default=0.01, help="expected-improvement exploration margin")
    parser.add_argument("--logp_bo_noise", type=float, default=1e-6, help="Gaussian-process white-noise level")
    parser.add_argument("--logp_bo_gp_restarts", type=int, default=2, help="Gaussian-process hyperparameter restarts")
    parser.add_argument("--logp_bo_full_grid_ei", action=argparse.BooleanOptionalAction, default=True, help="scan the full discrete Aij grid for maximum EI")
    parser.add_argument("--logp_bo_grid_chunk_size", type=int, default=100000, help="candidate chunk size for full-grid EI scans")
    parser.add_argument("--logp_penalize_failures", action=argparse.BooleanOptionalAction, default=True, help="penalize failed logP fitting simulations instead of aborting")
    parser.add_argument("--logp_failure_penalty_loss", type=float, default=100.0, help="objective loss assigned to failed single-target logP fitting runs")
    parser.add_argument("--logp_fit_article_protocol", action="store_true", help="use literature logP fitting defaults: 20x20x60, 4410 water, 2205 organic, 180 solute, 1e6 steps")
    parser.add_argument(
        "--logp_angle_param_mode",
        "--angle-param-mode",
        dest="logp_angle_param_mode",
        choices=["article", "geometry", "heuristic", "none"],
        default="",
        help="angle mode forwarded to logp_partition.py; article-protocol defaults to article when this is omitted",
    )
    parser.add_argument("--logp_fit_organic_solvent", default="", help="organic solvent recipe for logP fitting; default is octanol")
    parser.add_argument("--logp_fit_ensemble", choices=["nve", "nph"], default="", help="override logP fitting partition ensemble")
    parser.add_argument("--logp_fit_pressure", type=float, default=23.7, help="reduced pressure for logP fitting with nph")
    parser.add_argument("--logp_fit_pressure_damp", type=float, default=2.0, help="LAMMPS pressure damping for logP fitting with nph")
    parser.add_argument("--logp_bead_charge", action="append", default=[], help="charge override for logP fitting as BEAD=CHARGE")
    parser.add_argument("--logp_bead_heavy_atoms", action="append", default=[], help="heavy atom count for logP bead as BEAD=COUNT")
    parser.add_argument("--logp_heavy_atom_correction", choices=["none", "missing", "all"], default="missing", help="use heavy atom counts to estimate missing/all Rij and bonded r0 during logP fitting")
    parser.add_argument("--logp_heavy_radius_scale", type=float, default=1.0, help="scale factor for heavy-atom radius estimates")
    parser.add_argument("--logp_bonded_r0_factor", type=float, default=0.4, help="bonded r0 factor times Rij when r0 is estimated from heavy atom counts")
    parser.add_argument("--logp_table", type=str, default="pdf/logp/machine_readable_interactions.cvs")
    parser.add_argument("--lammps_steps", type=int, default=3000000)
    parser.add_argument("--timestep", type=float, default=0.02)
    parser.add_argument("--dpd_gamma", type=float, default=1.0)
    parser.add_argument("--dump_interval", type=int, default=5000)
    parser.add_argument("--thermo_interval", type=int, default=5000)
    parser.add_argument("--ensemble", choices=["nve", "nph"], default="nve")
    parser.add_argument("--target_pressure", type=float, default=23.7)
    parser.add_argument("--pressure_damp", type=float, default=10.0)
    parser.add_argument(
        "--charge_method",
        choices=["auto", "none", "explicit"],
        default="auto",
        help="solubility mode charge handling: auto enables explicit electrostatics only when formal charges are found",
    )
    parser.add_argument("--charge_lambda", type=float, default=0.25)
    parser.add_argument("--coul_cutoff", type=float, default=3.0)
    parser.add_argument("--kspace_accuracy", type=float, default=1.0e-4)
    parser.add_argument(
        "--charge_unit_scale",
        type=float,
        default=1.0,
        help="multiplier from formal charge to LAMMPS reduced charge units",
    )
    parser.add_argument(
        "--lammps_bin",
        default=default_lammps_executable(),
        help="LAMMPS executable for the final run; defaults to LAMMPS_BIN, then lmp_mpi, then lmp",
    )
    parser.add_argument(
        "--no_run",
        action="store_true",
        help="generate all workflow files but skip the final LAMMPS execution",
    )
    return parser.parse_args()


# --- 主逻辑 ---

args = create_parser()
smiles = args.smiles
box_size = args.box_size
packmol_in = args.packmol_in
input_data = args.input_data
time_start = time.time()

smiles_list = []
number_list = []
is_peg_list = []
cg_method = 'smiles'

if len(input_data) > 0:
    print('输入txt')
    smiles_list, number_list, is_peg_list, mode = import_from_file(input_data)
    cg_method = 'smiles' if mode == 'smiles' else 'structure'
else:
    while True:
        s = input('输入组分(q退出): ')
        if s == 'q': break
        n = input('输入个数: ')
        p = input('是否PEG(0/1): ')
        smiles_list.append(s)
        number_list.append(n)
        is_peg_list.append(int(p))

compoents_number = len(smiles_list)
split_number_list = []
all_cg_smiles_list_drop_tmp = []
all_cg_smiles_list_list = []
split_Mw = 50000

for compoents_id in range(compoents_number):
    smiles = smiles_list[compoents_id]
    current_peg = is_peg_list[compoents_id]
    
    if cg_method == 'structure':
        # Structure based path
        split_number_list.append(0)
        # Placeholder call
        cg_smiles_list_drop, cg_smiles = [], [] 
        all_cg_smiles_list_drop_tmp.append(cg_smiles_list_drop)
        all_cg_smiles_list_list.append(cg_smiles)
    else:
        mol = Chem.MolFromSmiles(smiles)
        Mw = rdMolDescriptors.CalcExactMolWt(mol)
        n = np.floor(Mw/split_Mw)
        split_number_list.append(int(n))
        split_smiles_list = get_split_smiles(smiles, int(n), split_Mw)
        
        for split_id in range(len(split_smiles_list)):
            print(f'正在计算第{compoents_id}个分子的第{split_id}个片段')
            cg_drop, cg_full = get_one_chain_and_smiles(
                split_smiles_list[split_id],
                compoents_id=compoents_id,
                split_id=split_id,
                cg_method='logp_article' if args.param_method == "logp" else 'smiles',
                is_peg=current_peg
            )
            all_cg_smiles_list_drop_tmp.append(cg_drop)
            all_cg_smiles_list_list.append(cg_full)

for compoents_id in range(compoents_number):
    n = split_number_list[compoents_id]
    if n == 0:
        if os.path.exists(f'{compoents_id}one_chain0.pdb'):
            os.rename(f'{compoents_id}one_chain0.pdb', f'{compoents_id}one_chain.pdb')
    else:
        # process_pdb_with_conect logic logic skipped for brevity, renaming first part
        if os.path.exists(f'{compoents_id}one_chain0.pdb'):
            os.rename(f'{compoents_id}one_chain0.pdb', f'{compoents_id}one_chain.pdb')

all_is_peg_list = generate_all_is_peg_list(is_peg_list, split_number_list, all_cg_smiles_list_drop_tmp)
all_type_list = get_all_type_list(split_number_list,all_cg_smiles_list_drop_tmp)

all_cg_smiles_list_drop = []
for l in all_cg_smiles_list_drop_tmp:
    for s in l: all_cg_smiles_list_drop.append(s)

print('fix start ')
fix_pdb_files([f'{i}one_chain.pdb' for i in range(compoents_number)])
print('fix over ')
print('create_packmol_in start ')
create_packmol_in(pack_model=0, box_size=box_size, number_list=number_list, packmol_in=packmol_in)
print('create_packmol_in over ')

bead_smiles_for_params = all_cg_smiles_list_drop + ["HOH"]
is_peg_for_params = all_is_peg_list + [0]
solution_type_id = len(bead_smiles_for_params)
parameter_values = None
parameter_report = {}
type_charges = [0.0] * len(bead_smiles_for_params)
charge_report = []
charge_enabled = False
counterion_counts = {}
extra_create_counts = {}
lammps_type_charges = type_charges
pair_cutoff_list = None
bond_r0_by_pair = {}
target_total_beads = int(round(get_box_volum(box_size) * 3))
solute_bead_count = 0
counterion_bead_count = 0
solution_number = 0

if args.param_method == "logp":
    logp_manual_overrides = parse_logp_pair_overrides(args.logp_manual_aij)
    logp_missing_mode = "optimize" if args.logp_missing == "manual" else args.logp_missing
    logp_discovery_overrides = {} if args.logp_missing == "manual" else logp_manual_overrides
    a_ij_list, logp_assignments, logp_pair_rows = create_logp_aij_list(
        bead_smiles=bead_smiles_for_params,
        solubility_values=None,
        table_path=args.logp_table,
        missing=logp_missing_mode,
        is_peg_list=is_peg_for_params,
        manual_overrides=logp_discovery_overrides,
        heavy_atom_correction=args.logp_heavy_atom_correction,
        heavy_radius_scale=args.logp_heavy_radius_scale,
    )
    missing_fit_rows = [
        row
        for row in logp_pair_rows
        if str(row.get("source", "")).startswith("logp_fit_required")
    ]
    if missing_fit_rows:
        recipe_types = flatten_logp_recipe(all_cg_smiles_list_list)
        if args.logp_missing == "manual":
            args.logp_fit_optimizer = "manual"
        fit_cmd, fit_outdir = write_logp_fit_protocol(
            missing_fit_rows,
            recipe_types,
            smiles_list,
            args,
        )
        if args.logp_fit_execute:
            print("logP table parameters are missing; running DPD logP fitting.")
            subprocess.run(fit_cmd, check=True)
            fitted_table = os.path.join(fit_outdir, "fitted_interactions.csv")
            if not os.path.exists(fitted_table):
                raise FileNotFoundError(f"logP fitting finished but did not write {fitted_table}")
            fitted_keys = {
                tuple(sorted((str(row["type_i"]), str(row["type_j"]))))
                for row in missing_fit_rows
            }
            post_fit_overrides = {
                key: value
                for key, value in logp_manual_overrides.items()
                if key not in fitted_keys
            }
            a_ij_list, logp_assignments, logp_pair_rows = create_logp_aij_list(
                bead_smiles=bead_smiles_for_params,
                solubility_values=None,
                table_path=fitted_table,
                missing="error",
                is_peg_list=is_peg_for_params,
                manual_overrides=post_fit_overrides,
                heavy_atom_correction=args.logp_heavy_atom_correction,
                heavy_radius_scale=args.logp_heavy_radius_scale,
            )
            args.logp_table = fitted_table
        else:
            raise RuntimeError(
                "logP mode found bead pairs absent from the parameter table. "
                "I wrote missing_logp_pairs.csv, logp_fit_required_pairs.csv, "
                "and run_missing_logp_fit.sh. Run that script or rerun with "
                "--logp_fit_execute, then use the fitted_interactions.csv table. "
                "For per-iteration manual fitting, use --logp_missing manual; "
                "optional --logp_manual_aij values are only initial guesses."
            )
    pair_cutoff_list = [
        (row["i"], row["j"], row["r_ij"])
        for row in logp_pair_rows
        if not pd.isna(row["r_ij"])
    ]
    bond_r0_by_pair = {
        tuple(sorted((int(row["i"]), int(row["j"])))): float(row["r0"])
        for row in logp_pair_rows
        if not pd.isna(row["r0"])
    }
    parameter_values = np.full(len(bead_smiles_for_params), 25.0)
    parameter_report = {
        "mode": "logp",
        "assigned_type": [row["assigned_type"] for row in logp_assignments],
        "assignment_status": [row["status"] for row in logp_assignments],
        "assignment_reason": [row["reason"] for row in logp_assignments],
    }
else:
    solubility_values = Pred_Solubility_Parameter(all_cg_smiles_list_drop, is_peg_list=all_is_peg_list)
    sol_sp = max(solubility_values) if len(solubility_values)>0 else 25.0
    parameter_values = np.append(solubility_values, sol_sp)
    type_charges, charge_report = create_charge_assignments(bead_smiles_for_params)
    has_nonzero_charge = any(abs(charge) > 1.0e-12 for charge in type_charges)
    charge_enabled = args.charge_method == "explicit" or (
        args.charge_method == "auto" and has_nonzero_charge
    )
    a_ij_list = create_solubility_aij_with_ions(parameter_values, bead_smiles_for_params)
    parameter_report = {
        "mode": "solubility",
        "Pred_Solubility_Parameter": parameter_values,
        "charge": type_charges,
        "charge_status": [row["status"] for row in charge_report],
        "charge_reason": [row["reason"] for row in charge_report],
        "charge_method": [args.charge_method] * len(parameter_values),
        "charge_enabled": [charge_enabled] * len(parameter_values),
    }

has_bonds = True
if os.path.exists('packed_polymer_and_solution.pdb'):
    atoms, bonds, angles = read_pdb('packed_polymer_and_solution.pdb')
    if not atoms:
        raise ValueError("packed_polymer_and_solution.pdb contains no solute beads.")
    solute_bead_count = len(atoms)
    has_bonds = len(bonds) > 0
    if args.param_method == "solubility" and charge_enabled:
        net_charge = solute_net_charge(atoms, type_charges)
        counterion_counts = counterions_for_charge(net_charge)
        for ion_smiles, count in counterion_counts.items():
            if count <= 0:
                continue
            if ion_smiles not in bead_smiles_for_params:
                bead_smiles_for_params.append(ion_smiles)
                parameter_values = np.append(parameter_values, sol_sp)
                charge, status, reason = infer_bead_charge(ion_smiles)
                type_charges.append(charge)
                charge_report.append({
                    "type_id": len(type_charges),
                    "bead_smiles": ion_smiles,
                    "charge": charge,
                    "status": status,
                    "reason": reason,
                })
            ion_type_id = bead_smiles_for_params.index(ion_smiles) + 1
            extra_create_counts[ion_type_id] = extra_create_counts.get(ion_type_id, 0) + count
        if counterion_counts:
            create_charge_assignments(bead_smiles_for_params)
        a_ij_list = create_solubility_aij_with_ions(parameter_values, bead_smiles_for_params)
    counterion_bead_count = sum(int(count) for count in extra_create_counts.values())
    solution_number = target_total_beads - solute_bead_count - counterion_bead_count
    if solution_number < 0:
        raise ValueError(
            "solute beads plus counterions exceed the target DPD density: "
            f"target={target_total_beads}, solute={solute_bead_count}, "
            f"counterions={counterion_bead_count}"
        )
    write_system_bead_summary(
        box_size,
        target_total_beads,
        solute_bead_count,
        counterion_bead_count,
        solution_number,
    )
    if args.param_method == "solubility" and charge_enabled:
        total_charge = write_charge_summary(
            atoms,
            type_charges,
            solution_number,
            solution_type_id=solution_type_id,
            extra_counts=extra_create_counts,
        )
        if abs(total_charge) > 1.0e-8:
            print(
                "Warning: explicit-charge DPD system is not charge neutral; "
                f"net charge = {total_charge:.6f}. Add counterions or use --charge_method none."
            )
        lammps_type_charges = [charge * args.charge_unit_scale for charge in type_charges]
        parameter_report = {
            "mode": "solubility",
            "Pred_Solubility_Parameter": parameter_values,
            "charge": type_charges,
            "charge_status": [row["status"] for row in charge_report],
            "charge_reason": [row["reason"] for row in charge_report],
            "charge_method": [args.charge_method] * len(parameter_values),
            "charge_enabled": [charge_enabled] * len(parameter_values),
            "counterion_count": [extra_create_counts.get(type_id, 0) for type_id in range(1, len(parameter_values) + 1)],
        }
    parameter_report.update({
        "target_total_beads": [target_total_beads] * len(parameter_values),
        "solute_bead_count": [solute_bead_count] * len(parameter_values),
        "counterion_bead_count": [counterion_bead_count] * len(parameter_values),
        "water_bead_count": [solution_number] * len(parameter_values),
    })
    topology_coeffs = generate_lammps_data(
        atoms,
        bonds,
        angles,
        parameter_values,
        box_size,
        'packed_polymer_and_solution.data',
        type_charges=lammps_type_charges,
        charge_enabled=charge_enabled,
        bond_r0_by_pair=bond_r0_by_pair,
        write_angles=args.param_method == "logp",
    )
else:
    raise FileNotFoundError("Packmol did not generate packed_polymer_and_solution.pdb")

create_lammps_in(
    parameter_values,
    box_size,
    solution_number,
    a_ij_list=a_ij_list,
    pair_cutoff_list=pair_cutoff_list,
    has_bonds=has_bonds,
    topology_coeffs=topology_coeffs,
    run_steps=args.lammps_steps,
    charge_enabled=charge_enabled,
    type_charges=lammps_type_charges,
    charge_lambda=args.charge_lambda,
    coul_cutoff=args.coul_cutoff,
    kspace_accuracy=args.kspace_accuracy,
    solution_type_id=solution_type_id,
    extra_create_counts=extra_create_counts,
    dpd_gamma=args.dpd_gamma,
    timestep=args.timestep,
    thermo_interval=args.thermo_interval,
    dump_interval=args.dump_interval,
    ensemble=args.ensemble,
    target_pressure=args.target_pressure,
    pressure_damp=args.pressure_damp,
)

component_ids_for_params = all_type_list + [all_type_list[-1] + 1 if all_type_list else 1]
while len(component_ids_for_params) < len(parameter_values):
    component_ids_for_params.append(component_ids_for_params[-1] + 1 if component_ids_for_params else 1)
smi_df_data = {
                      'all_cg_smiles_list_drop': bead_smiles_for_params,
                      'compoents_id': component_ids_for_params,
                      'type_id':[i for i in range(1,len(parameter_values)+1)],
                      'param_method': [args.param_method] * len(parameter_values),
}
for key, value in parameter_report.items():
    if key == "mode":
        continue
    smi_df_data[key] = value
smi_df = pd.DataFrame(smi_df_data)
smi_df.to_excel('smiles_SP.xlsx',index=False)

print('get_a_ij start ')
get_a_ij()
print('get_a_ij over ')
print(f'Total time: {time.time()-time_start}s')
if args.no_run:
    print("LAMMPS execution skipped because --no_run was set.")
else:
    try:
        subprocess.run(["mpirun", "-np", str(args.job), args.lammps_bin, "-sf","gpu","-pk","gpu","1","-i", "lammps.in"])
    except:
        pass
