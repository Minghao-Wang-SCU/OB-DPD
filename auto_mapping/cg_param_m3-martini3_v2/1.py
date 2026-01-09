import subprocess
from rdkit import Chem
def run_cg_param(smiles, gro_path, itp_path, tune_mode,is_opt):
    """
    通过子进程调用cg_param_m3.py
    
    参数:
        smiles (str): 分子SMILES表达式
        gro_path (str): 输出GRO文件路径
        itp_path (str): 输出ITP文件路径
        tune_mode (bool): 是否启用调谐模式
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    heavy_atom_count = mol.GetNumHeavyAtoms()
    if is_opt == '1':
        if heavy_atom_count > 50:
                print("重原子数:", heavy_atom_count)
                print('推荐原子数小于:',50)

    # 转换布尔值为命令行参数
    tune_flag = "1" if tune_mode else "0"
    
    # 构建命令列表
    cmd = [
        "python", 
        "cg_param_m3.py", 
        smiles, 
        gro_path, 
        itp_path, 
        tune_flag,
        is_opt
    ]
    flag = 1
    i = 1
    result = None  # 提前定义 result，避免 NameError

    while flag == 1:
        print(f"尝试第 {i} 次执行命令...")
        i += 1

        try:
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True  # 自动检查返回码
            )
            flag = 0  # 命令成功，退出循环
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败，返回码: {e.returncode}")
            print(f"错误输出: {e.stderr}")
            flag = 1  # 命令失败，继续循环
        except Exception as e:
            print(f"发生未知错误: {e}")
            flag = 1  # 未知错误，继续循环

    # 输出处理
    if result:
        print("标准输出:", result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)
    else:
        print("未获取到命令结果")


smi = 'NC(C)C(NC(CCCNC(N)=N)C(NC(CC(N)=O)C(NC(CC1=CNC2=C1C=CC=C2)C(NC(CC3=CC=C(O)C=C3)C(NC(C)C(NC(CCCNC(N)=N)C(NC(CC(N)=O)C(NC(CC4=CNC5=C4C=CC=C5)C(NC(CC6=CC=C(O)C=C6)C(C)=O)=O)=O)=O)=O)=O)=O)=O)=O)=O'
is_opt = '1'
run_cg_param(
    smiles=smi,
    gro_path="ethanol.gro", 
    itp_path="topol.itp", 
    tune_mode=False,
    is_opt=is_opt
)

print('OK')