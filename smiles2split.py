# %%
from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import rdchem
from collections import defaultdict
from rdkit.Chem.Draw import rdMolDraw2D
from print_atom_info import print_atom_info
# %%
def create_mol_atom_dir(mol):
    mol_atom = {}
    i = 0
    # 预生成原子索引映射（解决可能的同位素/原子映射号问题）
    idx_map = {atom.GetIdx(): i for i, atom in enumerate(mol.GetAtoms())}
    for atom in mol.GetAtoms():
        # 基础属性
        atom_dict = {
            "rd_id": atom.GetIdx(),        # RDKit原始索引
            "element": atom.GetSymbol(),    # 元素符号
            "id": idx_map[atom.GetIdx()],  # 用户自定义连续索引（可修改逻辑）
            "branch": None,                # 需自定义子链标识逻辑
            "cg_id": None,                  # 需自定义化学基团ID逻辑
            "degree": atom.GetDegree(),     # 原子度数
            "near_atom_id": [],                  # 相邻原子索引列表
            "near_bond": [],                # 相邻键类型列表
            "near_atom_degree": []               # 相邻原子度数列表
        }

        # 收集相邻原子信息
        neighbors = atom.GetNeighbors()
        bonds = [mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()) for nbr in neighbors]
        
        for nbr, bond in zip(neighbors, bonds):
            atom_dict["near_atom_id"].append(idx_map[nbr.GetIdx()])
            atom_dict["near_bond"].append(str(bond.GetBondType()))
            atom_dict["near_atom_degree"].append(nbr.GetDegree())
        mol_atom[i] = atom_dict
        i += 1 
    return mol_atom

# %%
def get_future_atom(atom_dir,near_atom_id,cged_id_list,branch_list,flag):
    min_degree = float('inf')
    min_degree_atom_id = None
    for atom_id in near_atom_id:
        if atom_id in cged_id_list:
            continue
        # 获取当前原子的信息
        atom_info =  atom_dir[atom_id]
        degree = atom_info['degree']

        # 更新最大原子度和对应的原子 ID
        if degree < min_degree:
            min_degree = degree
            min_degree_atom_id = atom_id
    if min_degree_atom_id == None:#如果没有邻居 说明到头了 则从第一支链位置重新开始
        
        
        if len(branch_list) >= 1:#具有支化点
            future_atom = atom_dir[branch_list[-1]]
            branch_list = branch_list[:-1]
        else:#如果没有支化点+
            future_atom = None
            flag = 0
        is_end = 1
    else:
        future_atom = atom_dir[min_degree_atom_id]
        is_end = 0
    return future_atom,branch_list,is_end,flag
def Is_Branch(atom_dir,atom,cged_id_list):
    uncged_neighbors = [nbr_id for nbr_id in atom['near_atom_id'] if nbr_id not in cged_id_list]
    if len(uncged_neighbors) > 1:
        is_branch = True
    else:
        is_branch = False
    return is_branch


def get_cg_group_graph(atom_dir):#返回cg组dir
    # 步骤1: 按 cg_id 分组
    cg_groups = {}
    for atom_id, info in atom_dir.items():
        cg_id = info['cg_id']
        if cg_id not in cg_groups:
            cg_groups[cg_id] = []
        cg_groups[cg_id].append(atom_id)
    
    # 步骤2: 对每个 cg_id 组，检测连通分量
    result_list = []
    for cg_id, atom_ids in cg_groups.items():
        # 构建子图邻接表（仅保留同一 cg_id 的连接）
        adj = {atom: [] for atom in atom_ids}
        for atom in atom_ids:
            neighbors = atom_dir[atom]['near_atom_id']
            valid_neighbors = [n for n in neighbors if n in adj]  # 过滤非本组的邻居
            adj[atom] = valid_neighbors
        
        # BFS 遍历找连通分量
        visited = set()
        for atom in atom_ids:
            if atom not in visited:
                queue = [atom]
                component = []
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        # 添加未访问的邻居（确保属于当前 cg_id 组）
                        queue.extend([n for n in adj[current] if n not in visited])
                result_list.append(component)
    result_dir = {}
    for i, subgraph in enumerate(result_list):
        cg_id = atom_dir[subgraph[0]]['cg_id']
        if cg_id in result_dir.keys():
            result_dir[cg_id].append(subgraph)
        else:
            result_dir[cg_id] = []
            result_dir[cg_id].append(subgraph)
    return result_dir

def is_connetc(cg_group_dir):#返回每个list中的图是否为连通图
    is_connetc_list = []
    for i in range(len(cg_group_dir)):
        cg_group = cg_group_dir[i]
        if len(cg_group) == 1:
            is_connetc_list.append(1)
        else:
            is_connetc_list.append(0)
    return is_connetc_list

def is_one_mol_in_cg_group(atom_dir):
    cg_group_graph_list = get_cg_group_graph(atom_dir)#获取不同cg组的图结构
    is_one_mol_in_cg_group_list = [is_connetc(graph) for graph in cg_group_graph_list ]#判断是否为连通图
    return is_one_mol_in_cg_group_list

def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))  # 递归处理子列表
        else:
            result.append(item)           # 非列表元素直接添加
    return result
def get_near_cg_id(atom_dir,subgraph_id):
    near_id = flatten([atom_dir[i]['near_atom_id'] for i in subgraph_id]) #获取相邻原子id
    near_id = list(set(near_id))
    #删除在片段中的原子
    for id in subgraph_id:
        if id in near_id:
            near_id.remove(id)
    near_cg_id = list(set([atom_dir[i]['cg_id'] for i in near_id]))
    
    near_cg_id_element_number = []
    for cg_id in near_cg_id:
        count = 0
        for i in range(len(atom_dir)):
            if atom_dir[i]['cg_id'] == cg_id:
                count += 1
        near_cg_id_element_number.append(count)
    #分配到周围原子数最小的组
    cg_id = near_cg_id[near_cg_id_element_number.index(min(near_cg_id_element_number))]
    return  cg_id
def regroup(atom_dir,cg_group_dir,is_one_mol_in_cg_group_list):
    #判断有问题片段的各个片段的原子数，如果原子数小则归到相邻的片段组，如果原子数大则新建组，
    #最后重新分配组序号，从0开始
    cg_id_list = list(set([atom_dir[i]['cg_id'] for i in range(len(atom_dir))]))
    for i,is_one_mol_in_cg_group in enumerate(is_one_mol_in_cg_group_list):
        if is_one_mol_in_cg_group == 0 : #有问题的片段
            for subgraph_id in cg_group_dir[i]:
                if len(subgraph_id) < 10: #如果小于阈值10
                    near_cg_id = get_near_cg_id(atom_dir,subgraph_id)
                    for id in subgraph_id:
                        atom_dir[id]['cg_id'] = near_cg_id
                else:  #否则分配到新组
                    for id in subgraph_id:
                        atom_dir[id]['cg_id'] = max(cg_id_list)+1

    return atom_dir
def fix_cg_id(atom_dir):
    #待所有原子分配cg组完成后使用
    #首先判断每个cg组的分子是否为1个  将不为1的重新分组
    #判断每个cg组原子数，过小的组整合到连接的族内
    cg_group_dir = get_cg_group_graph(atom_dir)
    is_one_mol_in_cg_group_list = is_connetc(cg_group_dir)
    
    for i in is_one_mol_in_cg_group_list:
        if i == 1: #说明为该片段划分无问题
            pass
        else:#否则片段划分有问题
            atom_dir = regroup(atom_dir,cg_group_dir,is_one_mol_in_cg_group_list)
    
    unique_cg_ids = sorted({atom['cg_id'] for atom in atom_dir.values()})

    # 创建映射关系：原始 cg_id -> 新的连续 cg_id
    cg_id_mapping = {original_id: new_id for new_id, original_id in enumerate(unique_cg_ids)}

    # 更新字典中的 cg_id 值
    for atom_id, atom_info in atom_dir.items():
        original_cg_id = atom_info['cg_id']
        atom_dir[atom_id]['cg_id'] = cg_id_mapping[original_cg_id]
        
    return atom_dir

# %%
def smiles2split(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # 生成原子列表
    atom_dir = create_mol_atom_dir(mol)
    flag  = 1
    atom_number = sum([1 for i in mol.GetAtoms()])
    past_atom = None
    now_atom = None
    future_atom = None
    branch_list = []
    cged_id_list = []
    cg_id = 0
    atom_dir = create_mol_atom_dir(mol)
    sum_atom_number = 0
    sum_cgi_number = 0

    while flag == 1:


        #如果所有原子均cg则停止
        if sum([1 for i in range(len(atom_dir)) if atom_dir[i]['cg_id'] == None]) == 0:
            flag = 0
            continue
        #如果仍有未cg的原子则继续循环
        else:
            #如果无past （刚开始循环）
            if past_atom == None:

                atom_dir[0]['cg_id'] = cg_id
                now_atom = atom_dir[0]
                if now_atom['id'] not in cged_id_list:
                    cged_id_list.append(now_atom['id'])
                sum_atom_number += 1
                
                now_atom = atom_dir[0]  
                past_atom =  now_atom
                
                if Is_Branch(atom_dir,now_atom,cged_id_list) :
                    atom_dir[now_atom['id']]['branch']   = True
                    branch_list.append(now_atom['id'])
                else:
                    atom_dir[now_atom['id']]['branch']   = False
                near_atom_id  = now_atom['near_atom_id']
                future_atom,branch_list,is_end,flag = get_future_atom(atom_dir,near_atom_id,cged_id_list,branch_list,flag)

            #已经有cg
            else:
                #更新past now future原子
                past_atom = now_atom
                now_atom = future_atom
                if Is_Branch(atom_dir,now_atom,cged_id_list) :
                    atom_dir[now_atom['id']]['branch']   = True
                    branch_list.append(now_atom['id'])
                else:
                    atom_dir[now_atom['id']]['branch']  = False
                near_atom_id  = atom_dir[now_atom['id']]['near_atom_id']
                
                future_atom,branch_list,is_end,flag = get_future_atom(atom_dir,near_atom_id,cged_id_list,branch_list,flag)

                #cg
                sum_cgi_number = sum([1 for i in range(len(atom_dir)) if atom_dir[i]['cg_id'] == cg_id])
                if is_end == 0:#取到有效邻居原子
                    if sum_cgi_number < 10:
                        if now_atom['id'] not in cged_id_list:
                            atom_dir[now_atom['id']]['cg_id'] = cg_id
                            cged_id_list.append(now_atom['id'])
                    else:
                        cg_id += 1
                        if now_atom['id'] not in cged_id_list:
                            atom_dir[now_atom['id']]['cg_id'] = cg_id
                            cged_id_list.append(now_atom['id'])
                        
                        sum_atom_number += 1
                elif is_end == 1:#如果取到尽头
                    if now_atom['id'] not in cged_id_list:
                        atom_dir[now_atom['id']]['cg_id'] = cg_id
                        cged_id_list.append(now_atom['id'])
        
        #flag += -1 
        sum_atom_umber = sum([1 for i in range(len(atom_dir)) if atom_dir[i]['cg_id'] != None])
        sum_cgi_number = sum([1 for i in range(len(atom_dir)) if atom_dir[i]['cg_id'] == cg_id])


    atom_dir = fix_cg_id(atom_dir)
    molecules = defaultdict(list)
    for  i in range(len(atom_dir)):
        atom = atom_dir[i]
        molecules[atom['cg_id']].append(atom)
    # 为每个分组创建一个 Mol 对象
    mols = []

    bond_types = {
        'SINGLE': rdchem.BondType.SINGLE,
        'DOUBLE': rdchem.BondType.DOUBLE,
        'AROMATIC': rdchem.BondType.AROMATIC
    }
    cg_id_list = []
    print(molecules)


    for cg_id, group_atoms in molecules.items():
        # 创建一个空的 RDKit 分子对象
        mol = Chem.RWMol()
        
        # 添加原子到分子中
        atom_to_idx = {}  # 用于存储原子 ID 到 RDKit 原子索引的映射
        new_id = 0
        for new_local_id,atom_info in enumerate(group_atoms):
            tmp = atom_info['id']
            atom_dir[atom_info['id']]['new_smiles_id'] = f'{tmp}'+'-'+f'{cg_id}'+'-'+f'{new_id}'
            
            atom = Chem.Atom(atom_info['element'])
            atom.SetAtomMapNum(tmp)
            idx = mol.AddAtom(atom)
            atom_to_idx[atom_info['id']] = idx
            new_id += 1 
        


        
        # 添加键到分子中
        existing_bonds = set()  # 用于记录已添加的键，避免重复
        for atom_info in group_atoms:
            for neighbor_id, bond_type in zip(atom_info['near_atom_id'], atom_info['near_bond']):
                if neighbor_id in atom_to_idx:
                    # 检查键是否已经存在
                    atom_idx1 = atom_to_idx[atom_info['id']]
                    atom_idx2 = atom_to_idx[neighbor_id]
                    bond_key = tuple(sorted([atom_idx1, atom_idx2]))
                    if bond_key not in existing_bonds:
                        bond_type_enum = bond_types.get(bond_type, rdchem.BondType.SINGLE)
                        mol.AddBond(atom_idx1, atom_idx2, bond_type_enum)
                        existing_bonds.add(bond_key)
        
        # 转换为不可变的 Mol 对象
        mol = mol.GetMol()
        cg_id_list.append(cg_id)
        mols.append(mol)

        print('--'*22)
        Chem.SanitizeMol(mol)
        
        print_atom_info(mol)
        sm = Chem.MolToSmiles(mol)
        print(sm)
        print_atom_info(Chem.MolFromSmiles(sm))

        
    

    
    smiles_list = []
    # 打印每个分子的 SMILES 表示
    for i, mol in enumerate(mols):
        #print(f"cg_id {i}: {Chem.MolToSmiles(mol)}")
        smiles_list.append(Chem.MolToSmiles(mol))
    return atom_dir,cg_id_list,smiles_list,mols

# %%

#smiles = 'OCCCCOC(NCC1(C)CC(CNC(OCCCCOC(NCC2(C)CC(CNC(OCC(COC(NCC3(C)CC(CC)CC(C)(C4=CC=CC=C4)C3)=O)(C([O-])=O)C)=O)CC(C)(C5=CC=CC=C5)C2)=O)=O)CC(C)(C6=CC=CC=C6)C1)=O'
# smiles2split(smiles)

# # %%

# c = [(0/255,112/255,192/255),(157/255,213/255,255/255),(112/255,173/255,71/255),(198/255,211/255,163/255),(255/255,189/255,0/255),(255/255,233/255,161/255),(206/255,10/255,254/255),(255/255,194/255,235/255),(62/255,62/255,62/255),(216/255,216/255,216/255),(100/255,20/255,2/255),(21/255,100/255,6/255),(255/255,0/255,0/255)]
# print(len(c))
# print(len(set(cg_)))
# cc = [c[i] for i in cg_]

# print(len(hit_ats))
# print(len(cc))
# print(hit_ats)
# print(cg_)
# print(cc)
# d = rdMolDraw2D.MolDraw2DCairo(1200,1200)
# rdMolDraw2D.PrepareAndDrawMolecule(d, Chem.MolFromSmiles(smiles), 
#                                 highlightAtoms=hit_ats,
#                                 highlightAtomColors={atom_id: color for atom_id, color in zip(hit_ats, cc)},
#                                 )
# d.DrawMolecule(mol)
# d.FinishDrawing()
# d.WriteDrawingText('tt.png')


