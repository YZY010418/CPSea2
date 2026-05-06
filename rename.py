import os
import sys

def process_pdb_files(input_dir, output_dir):
    """
    处理输入文件夹中的所有PDB文件，修改指定链编号并保存到输出文件夹
    
    参数:
        input_dir: 包含PDB文件的输入文件夹路径
        output_dir: 保存修改后文件的输出文件夹路径
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        # 只处理PDB文件
        if not filename.endswith('.pdb'):
            continue
            
        pdb_path = os.path.join(input_dir, filename)
        print(f"正在处理: {filename}")
        
        # 解析文件名获取链信息和ID
        parts = filename.split('_')
        if len(parts) < 3:
            print(f"  文件名格式错误: {filename}，跳过该文件")
            continue
        
        receptor_chain = parts[0]  # 受体链（如A）
        ligand_chain = parts[1]    # 配体链（如C）
        id_part = '_'.join(parts[2:])  # 提取ID部分（处理含下划线的情况）
        
        # 提取基础ID（去除.pdb后缀）
        pdb_id = id_part.rsplit('.', 1)[0] if '.' in id_part else id_part
        
        # 读取PDB文件内容
        try:
            with open(pdb_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  读取文件失败: {str(e)}，跳过该文件")
            continue
        
        # 处理每一行，修改链编号
        new_lines = []
        for line in lines:
            # 识别需要修改链编号的记录行（ATOM/HETATM/TER）
            record_type = line[:6].strip()
            if record_type in ['ATOM', 'HETATM', 'TER']:
                # 确保行长度足够（避免索引错误）
                if len(line) >= 22:
                    current_chain = line[21]  # PDB格式中链ID在第22列（0-based索引21）
                    
                    # 替换链编号
                    if current_chain == receptor_chain:
                        new_line = line[:21] + 'R' + line[22:]  # 受体链→R
                    elif current_chain == ligand_chain:
                        new_line = line[:21] + 'L' + line[22:]  # 配体链→L
                    else:
                        new_line = line  # 其他链不修改
                else:
                    new_line = line  # 短行不处理
            else:
                new_line = line  # 非目标记录行不处理
            
            new_lines.append(new_line)
        
        # 保存修改后的文件
        output_filename = f"{pdb_id}_renamed.pdb"
        output_path = os.path.join(output_dir, output_filename)
        try:
            with open(output_path, 'w') as f:
                f.writelines(new_lines)
            print(f"  处理完成，保存为: {output_filename}")
        except Exception as e:
            print(f"  写入文件失败: {str(e)}，跳过该文件")
            continue

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("使用方法: python rename_pdb_chains.py <输入文件夹路径> <输出文件夹路径>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    # 验证输入文件夹是否存在
    if not os.path.isdir(input_directory):
        print(f"错误: 输入文件夹 '{input_directory}' 不存在")
        sys.exit(1)
    
    # 开始处理文件
    process_pdb_files(input_directory, output_directory)
    print("所有文件处理完毕")
