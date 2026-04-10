# 根据pdb文件得到氨基酸距离矩阵(使用GraphSite编写的命令)
import os
import numpy as np
from tqdm import tqdm
# 代码文件存放路径
script_path = os.path.split(os.path.realpath(__file__))[0] + "/"
data_path = 'features/AF2/'
map_path = 'features/map/'
# 20个氨基酸的全称
aa = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
      "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
aa_abbr = [x for x in "ACDEFGHIKLMNPQRSTVWY"]
# zip()将多个可迭代对象，打包成元组，这里完成氨基酸全称和单个字母表示氨基酸的映射
aa_dict = dict(zip(aa, aa_abbr))


# 从.pdb格式文件中提取出蛋白序列
def get_seq(path, ID):
    seq = ""
    current_pos = -1000
    with open(path + ID + ".pdb", "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0:4] == "ATOM" and int(line[22:26].strip()) != current_pos:
            aa_type = line[17:20].strip()
            
            # 确保氨基酸类型存在于aa_dict中
            if aa_type in aa_dict:
                seq += aa_dict[aa_type]
            else:
                print(f"Warning: Unrecognized residue {aa_type} found in {ID}. Skipping it.")
            
            current_pos = int(line[22:26].strip())
    return seq
# .map文件第二行#后保存哪些位置的残基是缺失残基
# 设置缺失残基离左右残基距离为3.8A(正常距离),相邻两个残基距离为5.4A(较远距离)
# 读取.map文件，返回邻接矩阵
def process_distance_map(distance_map_file):
    with open(distance_map_file, "r") as f:
        lines = f.readlines()

    seq = lines[0].strip()
    length = len(seq)
    distance_map = np.zeros((length, length))

    if lines[1][0] == "#": # missed residues
        missed_idx = [int(x) for x in lines[1].split(":")[1].strip().split()] # 0-based
        lines = lines[2:]
    else:
        missed_idx = []
        lines = lines[1:]

    for i in range(0, len(lines)):
        record = lines[i].strip().split()
        for j in range(0, len(record)):
            distance_map[i + 1][j] = float(record[j])

    for idx in missed_idx:
        if idx > 0:
            distance_map[idx][idx - 1] = 3.8
        if idx > 1:
            distance_map[idx][idx - 2] = 5.4
        if idx < length - 1:
            distance_map[idx + 1][idx] = 3.8
        if idx < length - 2:
            distance_map[idx + 2][idx] = 5.4

    distance_map = distance_map + distance_map.T
    return seq, distance_map

# 利用命令文件caldis_CA从.pdb文件中计算出距离矩阵保存为.map文件
def get_distance_map(data_path, ID, PDB_seq, map_path):
    os.system("{}caldis_CA {}.pdb > {}.map".format(script_path, data_path + ID, map_path + ID))
    dis_map_seq, dis_map = process_distance_map(map_path + ID + ".map")
    if PDB_seq != dis_map_seq:
        # raise Exception("PDB_seq & dismap_seq mismatch")
        # 在wrong.txt文件中保存错误的蛋白id
        # 如果文件不存在，则创建一个
        if not os.path.exists("wrong.txt"):
            with open("wrong.txt", "w") as f:
                f.write("")
        with open("wrong.txt", "a") as f:
            f.write(ID + "\n")
        print(f"{ID}  PDB_seq & dismap_seq mismatch")
    else:
        np.save(map_path + ID + "_dismap.npy", dis_map)
        print(ID + '计算成功')
        return 0

def parse_fasta(fasta_file):
    protein_dict = {}
    with open(fasta_file, 'r') as f:
        protein_id = None
        protein_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):  # Fasta header starts with '>'
                if protein_id:  # 保存上一条序列
                    protein_dict[protein_id] = ''.join(protein_seq)
                # protein_id = line[1:].split()[0]  # 去掉 '>'
                protein_id = line[1:].strip()
                protein_seq = []  # 重置序列
            else:
                protein_seq.append(line)  # 拼接序列
        if protein_id:  # 处理最后一条序列
            protein_dict[protein_id] = ''.join(protein_seq)
    return protein_dict

ids_path = 'data/test/cd04_test_id.txt'
with open(ids_path, "r") as f:
    IDs = [line.strip() for line in f if line.strip()]
fasta_file = 'data/test/cd04_test.fasta'
seq_dict = parse_fasta(fasta_file)
for id in tqdm(IDs):
    if os.path.exists(map_path + id + "_dismap.npy"):
        continue
    # PDB_seq = get_seq(data_path,id)
    PDB_seq = seq_dict[id]
    get_distance_map(data_path,id,PDB_seq,map_path)