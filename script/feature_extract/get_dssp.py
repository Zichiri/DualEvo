# 根据pdb文件得到dssp文件
# 并处理dssp文件
import os
import numpy as np
from Bio import pairwise2
import tqdm
from multiprocessing import Pool
import subprocess
import time
# 20个氨基酸的全称
aa = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
      "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
aa_abbr = [x for x in "ACDEFGHIKLMNPQRSTVWY"]
# zip()将多个可迭代对象，打包成元组，这里完成氨基酸全称和单个字母表示氨基酸的映射
aa_dict = dict(zip(aa, aa_abbr))
def get_seq(path, ID):
    seq = ""
    current_pos = -1000
    with open(path + ID + ".pdb", "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0:4] == "ATOM" and int(line[22:26].strip()) != current_pos:
            aa_type = line[17:20].strip()
            seq += aa_dict[aa_type]
            current_pos = int(line[22:26].strip())
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
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        # C是无特定结构，如果dssp文件中没有定义氨基酸属于哪种二级结构，则是无特定结构
        if SS == " ":
            SS = "C"
        # SS_vec是每个氨基酸属于哪个二级结构的独热编码矩阵，如果这个氨基酸属于缺失氨基酸，则为unknown
        SS_vec = np.zeros(9) # The last dim represents "Unknown" for missing residues
        SS_vec[SS_type.find(SS)] = 1
        # phi和psi确定三维结构扭转角
        PHI = float(lines[i][103:109].strip())
        PSI = float(lines[i][109:115].strip())
        # ACC和ASA表示溶剂可及性，ASA是相对溶剂可及性（相对与该氨基酸溶剂可及性标准来说）
        ACC = float(lines[i][34:38].strip())
        ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
        # 每个氨基酸返回长度为12的向量
        dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

    return seq, dssp_feature


# 返回参考序列和原始序列匹配上了的那些氨基酸对应的dssp特征矩阵
# size：m*12(m:参考序列和比较序列能匹配上的氨基酸个数)
def match_dssp(seq, dssp, ref_seq):
    # biopython中的函数，对两个序列进行全局比对，返回多种比对的结果
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq = alignments[0].seqA
    seq = alignments[0].seqB

    SS_vec = np.zeros(9) # The last dim represent "Unknown" for missing residues
    SS_vec[-1] = 1
    padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

    new_dssp = []
    for aa in seq:
        if aa == "-":
            new_dssp.append(padded_item)
        else:
            new_dssp.append(dssp.pop(0))

    matched_dssp = []
    for i in range(len(ref_seq)):
        if ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])

    return matched_dssp


# 计算phi和psi角度的sin和cos，然后存在特征矩阵中
# size:n*14
def transform_dssp(dssp_feature):
    dssp_feature = np.array(dssp_feature)
    angle = dssp_feature[:,0:2]
    ASA_SS = dssp_feature[:,2:]

    radian = angle * (np.pi / 180)
    dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)

    return dssp_feature

# 先通过dssp命令得到dssp文件
# 处理函数处理dssp文件得到dssp特征矩阵
def get_dssp(data_path,dssp_path, ID, ref_seq):
    pdb_file = os.path.join(data_path, ID + ".pdb")
    dssp_file = os.path.join(dssp_path, ID + ".dssp")
    cmd = ["mkdssp", pdb_file, dssp_file]
    subprocess.run(cmd, check=True)
    for _ in range(10):
        if os.path.exists(dssp_file):
            break
        time.sleep(0.2)
    if not os.path.exists(dssp_file):
        raise FileNotFoundError(f"DSSP not generated: {dssp_file}")
    dssp_seq, dssp_matrix = process_dssp(dssp_path + ID + ".dssp")
    if dssp_seq != ref_seq:
        dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)
    np.save(dssp_path + ID + "_dssp.npy", transform_dssp(dssp_matrix))
    return 0

def process_one_id(id):
    if os.path.exists(dssp_path + id + "_dssp.npy"):
        return id  

    PDB_seq = get_seq(data_path, id)
    get_dssp(data_path, dssp_path, id, PDB_seq)
    return id
if __name__ == "__main__":
    data_path = 'features/AF2/'
    dssp_path = 'features/DSSP/'
    ids_path = 'data/train_cd04/train_id.txt'
    with open(ids_path, "r") as f:
        IDs = [line.strip() for line in f if line.strip()]
    with Pool(processes=48) as pool:
        list(tqdm.tqdm(pool.imap(process_one_id, IDs), total=len(IDs)))
