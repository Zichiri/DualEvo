# 根据cif文件得到dssp文件
# 并处理dssp文件
import os
import numpy as np
from Bio import pairwise2
from Bio.PDB import MMCIFParser # 新增：用于解析CIF文件
import tqdm
from multiprocessing import Pool
import subprocess
import time
import warnings

# 忽略Biopython解析时可能出现的非致命警告
warnings.filterwarnings("ignore")

# 20个氨基酸的全称
aa = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
      "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
aa_abbr = [x for x in "ACDEFGHIKLMNPQRSTVWY"]
# zip()将多个可迭代对象，打包成元组，这里完成氨基酸全称和单个字母表示氨基酸的映射
aa_dict = dict(zip(aa, aa_abbr))

# --- 修改点 1: 重写 get_seq 函数 ---
# PDB是固定列宽，CIF不是。强烈建议使用Biopython解析CIF以获得准确序列
def get_seq(path, ID):
    parser = MMCIFParser(QUIET=True)
    cif_file = os.path.join(path, "AF-"+ ID + "-F1-model_v6.cif")
    
    try:
        structure = parser.get_structure(ID, cif_file)
    except Exception as e:
        print(f"Error parsing {ID}: {e}")
        return ""

    seq = ""
    # 遍历模型、链和残基来获取序列
    # 注意：AF2预测结构通常只有一条链，如果有某种多链需求需额外处理
    for model in structure:
        for chain in model:
            for residue in chain:
                # 过滤掉非标准残基（如水分子、配体），id[0]为' '表示标准氨基酸
                if residue.id[0] == ' ':
                    resname = residue.get_resname()
                    if resname in aa_dict:
                        seq += aa_dict[resname]
                    # 可选：处理非标准氨基酸，这里略过
            break # 假设只需要第一条链，如果需要所有链请删除此行
        break # 假设只需要第一个模型
    
    return seq

# 对dssp文件处理得到dssp特征矩阵
# size:n*12
def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC" #八种二级结构类型
    # 二十种氨基酸溶剂可及性标准
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    # 寻找以 # 开头的行（DSSP数据的起始行）
    while p < len(lines) and lines[p].strip()[0] != "#":
        p += 1
    
    if p >= len(lines):
        return "", [] # 处理空文件或格式错误

    for i in range(p + 1, len(lines)):
        # 防御性编程：确保行足够长
        if len(lines[i]) < 115: 
            continue
            
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        # 小写字母通常代表半胱氨酸桥接，将其转为大写以便匹配
        if aa.islower():
            aa = 'C' 
            
        if aa not in aa_type: # 跳过非标准氨基酸X等
             # 如果需要占位，可以在这里处理，目前逻辑是跳过
            continue

        seq += aa
        SS = lines[i][16]
        # C是无特定结构，如果dssp文件中没有定义氨基酸属于哪种二级结构，则是无特定结构
        if SS == " ":
            SS = "C"
        # SS_vec是每个氨基酸属于哪个二级结构的独热编码矩阵，如果这个氨基酸属于缺失氨基酸，则为unknown
        SS_vec = np.zeros(9) # The last dim represents "Unknown" for missing residues
        
        if SS in SS_type:
            SS_vec[SS_type.find(SS)] = 1
        else:
            SS_vec[-1] = 1 # 处理未知结构
            
        # phi和psi确定三维结构扭转角
        try:
            PHI = float(lines[i][103:109].strip())
            PSI = float(lines[i][109:115].strip())
            # ACC和ASA表示溶剂可及性，ASA是相对溶剂可及性（相对与该氨基酸溶剂可及性标准来说）
            ACC = float(lines[i][34:38].strip())
            ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
        except ValueError:
            # 如果解析数值失败，填充默认值
            PHI, PSI, ASA = 360.0, 360.0, 0.0
            
        # 每个氨基酸返回长度为12的向量
        dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

    return seq, dssp_feature


# 返回参考序列和原始序列匹配上了的那些氨基酸对应的dssp特征矩阵
# size：m*12(m:参考序列和比较序列能匹配上的氨基酸个数)
def match_dssp(seq, dssp, ref_seq):
    # biopython中的函数，对两个序列进行全局比对，返回多种比对的结果
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq_aligned = alignments[0].seqA
    seq_aligned = alignments[0].seqB

    SS_vec = np.zeros(9) # The last dim represent "Unknown" for missing residues
    SS_vec[-1] = 1
    padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

    new_dssp = []
    # 这里的逻辑是将dssp特征根据对齐结果填充
    # 注意：这里假设 seq (来自dssp) 肯定比 ref_seq (来自fasta/cif) 短或相等
    dssp_idx = 0
    for char in seq_aligned:
        if char == "-":
            new_dssp.append(padded_item)
        else:
            if dssp_idx < len(dssp):
                new_dssp.append(dssp[dssp_idx])
                dssp_idx += 1
            else:
                 new_dssp.append(padded_item) # Fallback

    matched_dssp = []
    for i in range(len(ref_seq_aligned)):
        if ref_seq_aligned[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])

    return matched_dssp


# 计算phi和psi角度的sin和cos，然后存在特征矩阵中
# size:n*14
def transform_dssp(dssp_feature):
    if not dssp_feature:
        return np.array([])
    dssp_feature = np.array(dssp_feature)
    angle = dssp_feature[:,0:2]
    ASA_SS = dssp_feature[:,2:]

    radian = angle * (np.pi / 180)
    dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)

    return dssp_feature

# --- 修改点 2: get_dssp 函数中的文件后缀 ---
def get_dssp(data_path, dssp_path, ID, ref_seq):
    cif_file = os.path.join(data_path, "AF-"+ ID + "-F1-model_v6.cif")
    dssp_file = os.path.join(dssp_path, ID + ".dssp")
    
    # 确保 mkdssp 命令存在且支持 cif
    cmd = ["mkdssp", cif_file, dssp_file]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # print(f"Failed to run mkdssp for {ID}")
        return 1

    # 等待文件系统写入
    for _ in range(10):
        if os.path.exists(dssp_file) and os.path.getsize(dssp_file) > 0:
            break
        time.sleep(0.1)
    
    if not os.path.exists(dssp_file) or os.path.getsize(dssp_file) == 0:
        # raise FileNotFoundError(f"DSSP not generated: {dssp_file}")
        return 1

    dssp_seq, dssp_matrix = process_dssp(dssp_file)
    
    if not dssp_seq: # 如果处理出错
        return 1

    if dssp_seq != ref_seq:
        dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)
    
    np.save(os.path.join(dssp_path, ID + "_dssp.npy"), transform_dssp(dssp_matrix))
    return 0

def process_one_id(id):
    if os.path.exists(os.path.join(dssp_path, id + "_dssp.npy")):
        return id  

    # 调用修改后的 get_seq
    PDB_seq = get_seq(data_path, id)
    if not PDB_seq:
        return id # 如果解析序列失败，直接返回
        
    get_dssp(data_path, dssp_path, id, PDB_seq)
    return id

if __name__ == "__main__":
    data_path = 'data/squidly_new_test/af_cif/' # 存放 .cif 文件的路径
    dssp_path = 'data/squidly_new_test/dssp/'
    ids_path = 'data/squidly_new_test/pid/PC.txt'
    
    # 确保输出目录存在
    if not os.path.exists(dssp_path):
        os.makedirs(dssp_path)

    with open(ids_path, "r") as f:
        IDs = [line.strip() for line in f if line.strip()]
    
    # 注意：Pool process数量不要超过CPU核心数
    with Pool(processes=48) as pool:
        list(tqdm.tqdm(pool.imap(process_one_id, IDs), total=len(IDs)))