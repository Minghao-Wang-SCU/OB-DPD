import subprocess
import os
import sys

def get_cg(smiles, gro_path, itp_path, tune_model='0', is_opt='1', complents_id=1, split_id=1, is_peg=0):
    """
    调用粗粒化脚本 cg_param_m3.py 生成拓扑和坐标文件。

    参数:
    smiles (str): 分子的 SMILES 字符串
    gro_path (str): 输出 .gro 文件的路径（不含后缀）
    itp_path (str): 输出 .itp 文件的路径（不含后缀）
    tune_model (str): 是否调整模型参数 ('0' 或 '1')
    is_opt (str): 是否优化结构 ('0' 或 '1')
    complents_id (int/str): 组分 ID
    split_id (int/str): 片段 ID
    is_peg (int): 是否为 PEG 分子 (0 或 1)，用于触发特定的粗粒化策略
    """
    
    # 确保输出文件包含扩展名
    if not gro_path.endswith('.gro'):
        gro_path += '.gro'
    if not itp_path.endswith('.itp'):
        itp_path += '.itp'

    # 获取当前脚本所在目录，确保能找到 cg_param_m3.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, 'cg_param_m3.py')

    # 检查脚本是否存在
    if not os.path.exists(script_path):
        # 如果在当前目录没找到，尝试在工作目录查找
        if os.path.exists('cg_param_m3.py'):
            script_path = 'cg_param_m3.py'
        else:
            print(f"错误: 找不到 cg_param_m3.py 脚本 (路径: {script_path})")
            return

    # 构建命令参数列表
    # 参数顺序必须与 cg_param_m3.py 中的 sys.argv 索引对应
    # sys.argv[1]: smiles
    # sys.argv[2]: gro_path
    # sys.argv[3]: itp_path
    # sys.argv[4]: tune_model
    # sys.argv[5]: is_opt
    # sys.argv[6]: complents_id
    # sys.argv[7]: split_id
    # sys.argv[8]: is_peg (新增)

    command = [
        sys.executable,  # 使用当前 Python解释器
        script_path,
        smiles,
        gro_path,
        itp_path,
        str(tune_model),
        str(is_opt),
        str(complents_id),
        str(split_id),
        str(is_peg)      # 传递 PEG 标记
    ]

    print(f"执行粗粒化命令: {' '.join(command)}")

    try:
        # 执行命令并捕获输出
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # 打印脚本的标准输出（用于调试）
        if result.stdout:
            print(f"[cg_param_m3 输出]:\n{result.stdout}")
            
    except subprocess.CalledProcessError as e:
        print(f"粗粒化脚本执行出错 (返回码 {e.returncode}):")
        print(f"标准输出:\n{e.stdout}")
        print(f"错误输出:\n{e.stderr}")
        raise e
    except Exception as e:
        print(f"调用 mapping.get_cg 时发生未知错误: {str(e)}")
        raise e

if __name__ == "__main__":
    # 测试代码
    test_smiles = "CCOCCO"
    print("Testing mapping.py...")
    get_cg(test_smiles, "test_out", "test_out", is_peg=1)