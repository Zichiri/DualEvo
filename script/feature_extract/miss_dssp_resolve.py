# 部分蛋白质dssp中不包含所有的氨基酸信息，需要进行特殊处理
# 因为是得到的PDB文件不包含所有的氨基酸，所以只能删除
# 最终剩下来的酶有61621个
import os
IDs_path = '/public/home/ligroupprotein/Zhangcy/MutationEnzy/dataset/aegan/IDs.txt'
miss_ids_path = '/public/home/ligroupprotein/Zhangcy/MutationEnzy/dataset/aegan/miss_ids.txt'
save_path = '/public/home/ligroupprotein/Zhangcy/MutationEnzy/dataset/aegan/final_ids.txt'
with open(IDs_path, 'r', encoding='utf-8') as f:
    IDs = [line.strip() for line in f]
with open(miss_ids_path, 'r', encoding='utf-8') as f:
    miss_ids = [line.strip() for line in f]
final_ids = list(set(IDs)-set(miss_ids))
print(len(final_ids))
with open(save_path, 'w', encoding='utf-8') as f:
    for id in final_ids:
        f.write(str(id) + '\n')
    
    
