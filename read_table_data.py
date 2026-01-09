import re
import pandas as pd
def read_table_data(filename):
    data = []
    headers = None
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            
            # 检查是否是表格标题行
            if 'Step' in line and 'Temp' in line and 'E_pair' in line and 'E_mol' in line and 'TotEng' in line and 'Press' in line:
                headers = re.split(r'\s+', line.strip())
                continue
            
            # 如果已经找到标题行，则读取数据行
            if headers is not None:
                values = re.split(r'\s+', line.strip())
                if len(values) == len(headers):
                    data.append(values)
    df = pd.DataFrame(data,columns=headers)
    return df
