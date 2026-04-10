# aegan模型提取特征的过程
import os
import json
import numpy as np
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import torch.nn.functional as F
import torch
from tqdm import tqdm
one_hot = [0 for _ in range(21)]
transRes = {
    'ASP': 'D',
    'PRO': 'P',
    'ARG': 'R',
    'ILE': 'I',
    'LEU': 'L',
    'GLU': 'E',
    'ALA': 'A',
    'VAL': 'V',
    'TYR': 'Y',
    'ASN': 'N',
    'HIS': 'H',
    'PHE': 'F',
    'THR': 'T',
    'GLN': 'Q',
    'GLY': 'G',
    'MET': 'M',
    'LYS': 'K',
    'SER': 'S',
    'CYS': 'C',
    'TRP': 'W',
}
Atchley = {
    'A': [-0.591, -1.302, -0.733, 1.57, -0.146],
    'C': [-1.343, 0.465, -0.862, -1.02, -0.255],
    'D': [1.05, 0.302, -3.656, -0.259, -3.242],
    'E': [1.357, -1.453, 1.477, 0.113, -0.837],
    'F': [-1.006, -0.59, 1.891, -0.397, 0.412],
    'G': [-0.384, 1.652, 1.33, 1.045, 2.064],
    'H': [0.336, -0.417, -1.673, -1.474, -0.078],
    'I': [-1.239, -0.547, 2.131, 0.393, 0.816],
    'K': [1.831, -0.561, 0.533, -0.277, 1.648],
    'L': [-1.019, -0.987, -1.505, 1.266, -0.912],
    'M': [-0.663, -1.524, 2.219, -1.005, 1.212],
    'N': [0.945, 0.828, 1.299, -0.169, 0.933],
    'P': [0.189, 2.081, -1.628, 0.421, -1.392],
    'Q': [0.931, -0.179, -3.005, -0.503, -1.853],
    'R': [1.538, -0.055, 1.502, 0.44, 2.897],
    'S': [-0.228, 1.399, -4.76, 0.67, -2.647],
    'T': [-0.032, 0.326, 2.213, 0.908, 1.313],
    'V': [-1.337, -0.279, -0.544, 1.242, -1.262],
    'W': [-0.595, 0.009, 0.672, -2.128, -0.184],
    'Y': [0.26, 0.83, 3.097, -0.838, 1.512]
}

# 计算每个残基的邻居节点:非甘氨酸根据cb原子确定位置，甘氨酸根据ca原子确定位置,node:限定的邻居节点的数量,候选残基的索引index:活性位点索引+采样出的非活性位点索引
# 返回候选残基的邻居节点索引 [len(index),nodes]
# 邻居节点中会包含自己本身
def NeighborNodeGenerator(cblist, Nodes, index):
    # 计算最近
    dist = []
    for i in index:
        # struc.distance:计算一个点到其他所有点的距离，返回一个距离数组
        dist.append(struc.distance(cblist[i], cblist))
    # [len(index),n]
    dist = torch.tensor(np.array(dist))
    # 从每个残基的距离数组中找到最近的 Nodes 个邻居
    index = torch.topk(dist, Nodes, dim=-1, largest=False)[1]
    return index

# 如果原子缺失，用前一个原子替代，将错误信息记录到文件中
def err_handle(atom, dsc):
    for i in range(3, 7):
        if atom[i] == 0:
            atom[i] = atom[i - 1]
    with open("./err_atom", 'a+') as f:
        f.write(dsc + '\n')
    return atom
def get_atom(subarray, dscribe):
    res_name = subarray[0].res_name
    try:
        atom = [0, 0, 0, 0, 0, 0, 0]
        atom[0] = subarray[0]
        atom[1] = subarray[1]
        atom[2] = subarray[2]
        if res_name == 'SER':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'OG'][0]
        elif res_name == 'PHE':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CD1'][0]
            atom[6] = subarray[subarray.atom_name == 'CZ'][0]
        elif res_name == 'THR':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG2'][0]
            atom[5] = subarray[subarray.atom_name == 'OG1'][0]
        elif res_name == 'LEU':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CD1'][0]
            atom[6] = subarray[subarray.atom_name == 'CD2'][0]
        elif res_name == 'ASN':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'OD1'][0]
            atom[6] = subarray[subarray.atom_name == 'ND2'][0]
        elif res_name == 'LYS':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CD'][0]
            atom[6] = subarray[subarray.atom_name == 'NZ'][0]
        elif res_name == 'VAL':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG1'][0]
            atom[5] = subarray[subarray.atom_name == 'CG2'][0]
        elif res_name == 'ILE':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG1'][0]
            atom[5] = subarray[subarray.atom_name == 'CG2'][0]
            atom[6] = subarray[subarray.atom_name == 'CD1'][0]
        elif res_name == 'ALA':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
        elif res_name == 'GLU':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'OE1'][0]
            atom[5] = subarray[subarray.atom_name == 'OE2'][0]
        elif res_name == 'ARG':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'NE'][0]
            atom[6] = subarray[subarray.atom_name == 'NH2'][0]
        elif res_name == 'ASP':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'OD1'][0]
            atom[6] = subarray[subarray.atom_name == 'OD2'][0]
        elif res_name == 'PRO':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CD'][0]
        elif res_name == 'TYR':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CD1'][0]
            atom[6] = subarray[subarray.atom_name == 'OH'][0]
        elif res_name == 'GLN':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CD'][0]
            atom[6] = subarray[subarray.atom_name == 'NE2'][0]
        elif res_name == 'HIS':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'ND1'][0]
            atom[6] = subarray[subarray.atom_name == 'CE1'][0]
        elif res_name == 'TRP':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CE3'][0]
            atom[6] = subarray[subarray.atom_name == 'CH2'][0]
        elif res_name == 'CYS':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'SG'][0]
        elif res_name == 'MET':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'SD'][0]
            atom[6] = subarray[subarray.atom_name == 'CE'][0]
    except:
        atom = err_handle(atom, dscribe + f"_{res_name}")
    return atom
# # 因为PSSM获取时间太长，所以暂时先不添加PSSM
# # Pssm:[n,25]
# def generatePssm(Sequence):
#     # print("start to generate pssminfo")
#     Pssm = list()
#     # 生成并获取Pssm
#     with open('_temp.fasta', 'w') as f:
#         f.write('>temp\n' + Sequence)
#     os.system(cmd)
#     with open('_temp.pssm', 'r') as f:
#         data = f.readlines()
#     for line in data:
#         line = [item for item in line.strip().split(' ') if item not in [' ', '']]
#         if len(line) != 0 and line[0].isdigit():
#             # Pssm.append(line[2: 22])
#             pssm = list(map(float, line[2: 22]))
#             try:
#                 pssm = pssm + Atchley[line[1]]
#             except:
#                 pssm = pssm + [0, 0, 0, 0, 0]
#             Pssm.append(pssm)
#     print("pssminfo has generated!")
#     os.remove('./_temp.fasta')
#     os.remove('./_temp.pssm')
#     print("pssminfo has load!")
#     return Pssm

def getAtchley(Sequence):
    Ath = []
    for i in range(len(Sequence)):
        Ath.append(Atchley[Sequence[i]])
    return Ath

def getAtomModel(length, array, desc):
    #     base = array[0].res_id
    ca = array[(array.atom_name == 'CA') & (array.hetero == False)]
    AM = []
    for i in range(length):
        subarray = array[(array.res_id == ca[i].res_id) & (array.hetero == False)]
        atom = get_atom(subarray, desc + f'_{ca[i].res_id}')
        atomModel = [0, 0, 0]
        if subarray[0].res_name == 'SER':
            atomModel = atom[4].coord - atom[1].coord
        elif subarray[0].res_name in ['PHE', 'LEU', 'ASN', 'LYS', 'ARG', 'ASP', 'TYR', 'GLN', 'TRP']:
            atomModel = 1 / 2 * (atom[6].coord + atom[5].coord) - atom[1].coord
        elif subarray[0].res_name in ['THR', 'VAL', 'GLU']:
            atomModel = 1 / 2 * (atom[5].coord + atom[4].coord) - atom[1].coord
        elif subarray[0].res_name == 'ILE':
            atomModel = 1 / 4 * (atom[5].coord + atom[4].coord) + 1 / 2 * atom[6].coord - atom[1].coord
        elif subarray[0].res_name == 'ALA':
            atomModel = atom[3].coord - atom[1].coord
        elif subarray[0].res_name == 'PRO':
            atomModel = atom[5].coord + atom[3].coord - atom[4].coord - atom[1].coord
        elif subarray[0].res_name in ['HIS', 'MET']:
            atomModel = 1 / 4 * (atom[4].coord + atom[6].coord) + 1 / 2 * atom[5].coord - atom[1].coord
        elif subarray[0].res_name == 'CYS':
            atomModel = atom[4].coord - atom[1].coord
        if subarray[0].res_name != 'GLY':
            angle = struc.dihedral(atom[4], atom[3], atom[1], atom[3])
            atomModel = np.append([np.sin(angle), np.cos(angle)], atomModel)
        else:
            atomModel = np.append([0, 0], atomModel)
        AM.append(atomModel)
    return torch.tensor(np.array(AM))

# 每个候选残基的21个最近邻居中，只有距离小于8A的才会有一条边相连
# m个候选残基构成了m个局部残基网络
def get_edge(cb, neighbor):
    # [n,3] n:序列总长
    cb = torch.tensor(cb)
    # [m,21,3]
    node_cb = cb[neighbor]
    b, s, d = node_cb.shape
    # 所有节点对的坐标
    # [m,21*21,3]
    x = node_cb.unsqueeze(2).repeat(1, 1, s, 1).reshape(b, s * s, d)
    # [m,21*21,3]
    y = node_cb.unsqueeze(1).repeat(1, s, 1, 1).reshape(b, s * s, d)
    # [m,21*21,3] 所有节点对的坐标差的单位向量
    unit = F.normalize(x - y)
    pdist = torch.nn.PairwiseDistance(p=2)
    # [m,21*21] 所有节点对的距离
    dist = pdist(x, y)
    # 邻接矩阵[m,21,21] 值为 1 表示距离小于 8 Å，值为 0 表示距离大于或等于 8 Å
    adj = (dist < 8).reshape(b, s, s).float()
    # dist 被缩放并四舍五入为整数，范围为 [0, 31]
    dist = torch.div(dist, 0.5, rounding_mode='floor').long()
    dist[dist > 31] = 31
    # [m,21*21,32] 每个距离的独热编码
    dist_code = F.one_hot(dist, num_classes=32)
    # [m,21*21,35] [m,21,21]
    return torch.cat([dist_code, unit], dim=-1), adj

# 处理每个残基的特征信息
# cb, phi_psi_omega(n,6), NeighborInfo(m,21), Pssm_Atchley(n,25), AM(n,5), len(indexinfo), target, pdbname
def _site_handle_(CB, phi_psi_omega, neighbor_info, pssm_info, AM, gap, target, pbdname):
    # [m,21,25]
    node_pssm = pssm_info[neighbor_info]
    # [m,21,6]
    node_torsion = phi_psi_omega[neighbor_info]
    # [m,21,5]
    node_AM = AM[neighbor_info]
    # [m,21,36]
    node_fea = torch.cat([node_pssm, node_torsion, node_AM], dim=-1)
    # .coord访问原子的三维坐标
    # edge:[m,21*21,35]  adj:[m,21,21]
    edge, adj = get_edge(CB.coord, neighbor_info)
    node_fea[torch.isnan(node_fea)] = 0
    node_fea[torch.isinf(node_fea)] = 0
    edge[torch.isinf(edge)] = 0
    edge[torch.isnan(edge)] = 0
    # 将特征和标签存储下来每个蛋白选择了m个候选残基,特征(点特征[21,36],边特征[21*21,35],邻接矩阵[21,21],label是否是活性位点)
    for i in range(len(neighbor_info)):
        if i < gap:
            label = 0
        else:
            label = 1
        torch.save([node_fea[i], edge[i], adj[i], label], os.path.join(target, pbdname + '_{}'.format(i)))
def sofmax(logits):
    e_x = np.exp(logits)
    probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
    return probs
# 计算每个位置残基采样概率：越在活性位点周围的非活性位点被采样到的概率越大
def guss_like_generator(mean, x, peak):
    mean = np.array(mean)[:, None]
    std = len(x)
    return (np.exp(-np.power((x - mean), 2) / (2 * std)) / (peak * 1.1)).sum(axis=0)

# prob:每个位置的残基被采样到的概率
# 总的采样个数min(4*act_num,n) 即每个蛋白质会采样该蛋白质活性位点个数4倍的残基
# 返回要采样的非活性位点编号
def choose_by_prob(prob, active):
    prob = sofmax(prob)
    act_nums = len(active)
    active = np.array(active)
    sample_num = act_nums * 4 if act_nums * 4 < len(prob) else len(prob)
    choose = np.random.choice(list(range(len(prob))), sample_num, p=prob)
    deredundance = list(set(choose) - set(active))
    return deredundance
# 对一个蛋白中所有的残基按照概率采样：越靠近活性位点的非活性位点被采样的概率越大
def sample(active_index, length, is_all=False):
    # peak:该蛋白质活性位点个数
    peak = len(active_index)
    # 进行挑选
    if not is_all:
        chosen_index = choose_by_prob(
            guss_like_generator(active_index, np.arange(length), peak),
            active_index
        )
    else:
        # 可能有错误index应该是从1开始而非从0开始
        chosen_index = list(set(range(length)) - set(active_index))
    return chosen_index
# chain:蛋白质链
def constructData(chain, pdbname, activeindex, target,is_all):
    ca_list = chain[(chain.atom_name == "CA") & (chain.hetero == False)]
    # 从蛋白质链中得到每个氨基酸的三个二面角
    phi, psi, omega = struc.dihedral_backbone(chain)
    # [n,3]:n个氨基酸的三个二面角
    phi_psi_omega = torch.stack([torch.tensor(phi), torch.tensor(psi), torch.tensor(omega)]).transpose(-1, -2)
    # [n,6]:n个氨基酸的三个二面角的sin和cos
    phi_psi_omega = torch.cat([torch.sin(phi_psi_omega), torch.cos(phi_psi_omega)], dim=-1)
    # 空值用0填充
    phi_psi_omega = torch.nan_to_num(phi_psi_omega)
    # 根据PDB文件中残基名称得到蛋白序列
    Sequence = ''.join(list(map(lambda x: transRes.get(x, ''), ca_list.res_name.tolist())))
    seq_len = len(Sequence)
    if seq_len < 30:
        return
    # [n,25]
    # Pssm_Atchley = torch.tensor(generatePssm(Sequence))
    Pssm_Atchley = torch.tensor(getAtchley(Sequence))
    # 采样出的非活性位点索引
    indexinfo = sample(activeindex, len(ca_list), is_all)
    # 每个蛋白链的空间原子模型:[n,5]
    AM = getAtomModel(len(Sequence), chain,'atmo' + f'_{pdbname}')
    # 筛选出甘氨酸中的Ca原子和非甘氨酸残基中的Cb原子
    cb = chain[
        (((chain.res_name == 'GLY') * (chain.atom_name == 'CA')) | (chain.atom_name == 'CB')) & (chain.hetero == False)]
    # 构建局部残基网络，对于每个候选残基找到最近的21个邻居节点
    # [len(indexinfo+activeindex),21]
    NeighborInfo = NeighborNodeGenerator(cb, 21, indexinfo + activeindex)
    _site_handle_(cb, phi_psi_omega, NeighborInfo, Pssm_Atchley, AM, len(indexinfo), target, pdbname)
    pass

# 使用pdb包得到单个蛋白质链和活性位点
def mainwork(pdbfilepath, targetpath, activeindexpath, is_all):
    if not os.path.exists(targetpath):
        os.makedirs(targetpath)
    with open(activeindexpath, 'r') as f:
        siteindex = json.load(f)
    for pdbfile in tqdm(siteindex.keys()):
        # 读取单个蛋白的PDB文件
        pdb_file = pdb.PDBFile.read(os.path.join(pdbfilepath, pdbfile+'.pdb'))
        # 返回蛋白质链
        arry = pdb_file.get_structure(model=1)
        # 对应蛋白的活性位点字典
        activeindex = siteindex[pdbfile]
        constructData(arry, pdbfile, activeindex, targetpath,is_all)
        
if __name__ == "__main__":
    # pdb文件路径,json标签文件,数据集存储路径
    source_path = './checkpoints/features/AF2'
    index_path = './dataset/aegan/CD-HIT/0.6hit/act_site_index.json'
    target_path = './checkpoints/features/aegan_features'
    mainwork(source_path, target_path, index_path, False)