from smiles2split import smiles2split,flatten
from cg import cg
def splited_atom_dir2cg_atom_dir(splited_atom_dir,splited_cg_id_list,smiles_list,mols):

    splited_atom_dir = cg(splited_atom_dir,smiles_list,mols)



    cg_near_atom_list = []
    
    #print(splited_cg_id_list)
    for cg_id in splited_cg_id_list:
        atom_near_atom_list = []
        for i in range(len(splited_atom_dir)):
            if cg_id == splited_atom_dir[i]['cg_id']:#第i个原子属于该cg组
                atom_near_atom_list.append([splited_atom_dir[near_id]['cg_id'] for near_id in atom_dir[i]['near_atom_id']])
        cg_near_atom_list.append(list(set(flatten(atom_near_atom_list))))
    
    # print('cg_near_atom_list',len(cg_near_atom_list))
    # print(cg_near_atom_list)
    # print('cg_near_atom_list 0\n',cg_near_atom_list[0])
    