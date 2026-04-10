# 从cif文件中得到距离矩阵
import os
import numpy as np
from tqdm import tqdm
from Bio.PDB import MMCIFParser
import warnings

# 忽略 BioPython 读取某些不规范 CIF 时的警告
warnings.filterwarnings("ignore")

# --- 配置路径 ---
data_path = 'data/squidly_new_test/af_cif/'   # 存放 .cif 文件的路径
map_path = 'data/squidly_new_test/map/'    # 输出 .npy 的路径

# 确保输出目录存在
if not os.path.exists(map_path):
    os.makedirs(map_path)

# 20个氨基酸的字典 (保留你原来的字典，虽然BioPython也有内置的)
aa = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
      "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
aa_abbr = [x for x in "ACDEFGHIKLMNPQRSTVWY"]
aa_dict = dict(zip(aa, aa_abbr))

# --- 新增/修改的核心函数 ---

def get_structure_info_from_cif(file_path, protein_id):
    """
    解析 CIF 文件，提取序列和 CA 原子坐标
    """
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(protein_id, file_path)
    except Exception as e:
        print(f"Error parsing {protein_id}: {e}")
        return None, None

    # 获取第一个模型和第一条链 (根据实际情况调整，通常 AF2 结果只有一条链)
    model = structure[0]
    chain = list(model.get_chains())[0]

    seq_extracted = ""
    coords = []
    residue_indices = [] # 记录存在的残基编号(auth_seq_id)，用于判断缺失

    for residue in chain:
        # 跳过非标准氨基酸或水分子 (HETATM)
        if residue.id[0] != ' ':
            continue
        
        res_name_3 = residue.get_resname()
        
        # 转换氨基酸名为单字母
        if res_name_3 in aa_dict:
            res_char = aa_dict[res_name_3]
        elif res_name_3 == "MSE": # 特殊处理常见的硒代蛋氨酸
            res_char = "M"
        else:
            res_char = 'X' # 其他非标准氨基酸标记为X
        
        # 提取 Alpha Carbon (CA) 的坐标
        if 'CA' in residue:
            seq_extracted += res_char
            coords.append(residue['CA'].get_coord())
            residue_indices.append(residue.id[1]) # 获取 PDB 编号
        else:
            # 如果没有 CA 原子（通常只有甘氨酸才有 CA 但偶尔数据缺失），处理逻辑视情况而定
            pass

    return seq_extracted, np.array(coords), residue_indices

def calculate_distance_matrix_numpy(coords, seq_len):
    """
    使用 Numpy 广播机制快速计算距离矩阵
    """
    # coords shape: (N, 3)
    # 计算差值矩阵: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # 计算欧几里得距离: sqrt(sum(diff^2))
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return dist_matrix

def process_cif_pipeline(data_path, ID, fasta_seq, map_path):
    """
    主处理流程：读取CIF -> 计算矩阵 -> 处理缺失残基 -> 保存
    """
    cif_file = os.path.join(data_path, "AF-"+ ID + "-F1-model_v6.cif") #注意这里不同cif命名格式需要调整
    
    if not os.path.exists(cif_file):
        print(f"File not found: {cif_file}")
        return

    # 1. 从 CIF 提取信息
    extracted_seq, coords, res_indices = get_structure_info_from_cif(cif_file, ID)
    
    if extracted_seq is None:
        return

    # 2. 校验序列 (简单长度校验，严格校验可用序列比对)
    # 注意：AF2 预测结构通常是全长的，但如果是实验结构(PDB)可能会有缺失
    # 这里我们假设 CIF 里的序列和 FASTA 应该大致对应。
    # 为了兼容你原本的逻辑：如果 CIF 提取的序列和 FASTA 不一致，我们需要知道哪些位置是缺失的。
    
    full_len = len(fasta_seq)
    final_matrix = np.zeros((full_len, full_len))
    
    # 3. 计算现有的 CA 距离矩阵
    if len(coords) > 0:
        real_dist_matrix = calculate_distance_matrix_numpy(coords, len(coords))
    else:
        print(f"{ID} No CA atoms found.")
        return

    # 4. 填充到全长矩阵 (处理缺失残基逻辑)
    # 这一步比较复杂，因为需要将提取的结构序列对齐到 FASTA 序列。
    # 如果是 AlphaFold2 的结果，通常 seq_extracted == fasta_seq。
    
    if extracted_seq == fasta_seq:
        final_matrix = real_dist_matrix
    else:
        # 如果序列不匹配（说明有缺失残基），需要做映射
        # 这里简化处理：假设 ID 是连续的，利用 res_indices 映射到矩阵
        # 注意：这假设 fasta 是完整的，cif 是片段。
        # 真实的对齐可能需要 Biopython 的 PairwiseAligner，这里沿用你原来的思路：
        # 只要序列对不上就报错记录，或者你可以简单填充。
        
        # 按照你原代码的逻辑，如果不匹配就报错：
        if not os.path.exists("wrong.txt"):
             with open("wrong.txt", "w") as f: f.write("")
        with open("wrong.txt", "a") as f:
            f.write(f"{ID} Seq Mismatch: Fasta len {len(fasta_seq)} vs CIF len {len(extracted_seq)}\n")
        print(f"{ID} Sequence mismatch")
        return 0

    # # 5. 应用你原有的缺失值填充逻辑 (可选)
    # # 原代码中：distance_map[idx][idx - 1] = 3.8 ...
    # # 在 Numpy 矩阵中，如果数据是完整的，不需要这一步。
    # # 如果你是想强制设置相邻残基的物理距离（防止计算误差），可以保留：
    # for i in range(len(final_matrix)):
    #     if i > 0: final_matrix[i][i-1] = min(final_matrix[i][i-1], 3.8) # 确保相邻不超过3.8
    #     if i > 0: final_matrix[i-1][i] = min(final_matrix[i-1][i], 3.8)
    
    # 6. 保存
    np.save(os.path.join(map_path, ID + "_dismap.npy"), final_matrix)
    # print(ID + ' 计算成功') # 减少打印，tqdm已有进度条

def parse_fasta(fasta_file):
    protein_dict = {}
    with open(fasta_file, 'r') as f:
        protein_id = None
        protein_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if protein_id:
                    protein_dict[protein_id] = ''.join(protein_seq)
                protein_id = line.split('|')[1].strip() # fasta文件中提取出蛋白id，注意这里不同的文件的id格式可能不同
                protein_seq = []
            else:
                protein_seq.append(line)
        if protein_id:
            protein_dict[protein_id] = ''.join(protein_seq)
    return protein_dict

# --- 主执行逻辑 ---

# 读取 ID 列表
ids_path = 'data/squidly_new_test/pid/PC.txt'
with open(ids_path, "r") as f:
    IDs = [line.strip() for line in f if line.strip()]

# 读取 FASTA
fasta_file = 'data/squidly_new_test/fasta/PC.fasta'
seq_dict = parse_fasta(fasta_file)

# 循环处理
for id in tqdm(IDs):
    save_path = os.path.join(map_path, id + "_dismap.npy")
    if os.path.exists(save_path):
        continue
    
    if id not in seq_dict:
        print(f"Warning: ID {id} not found in fasta file.")
        continue
        
    fasta_seq = seq_dict[id]
    process_cif_pipeline(data_path, id, fasta_seq, map_path)