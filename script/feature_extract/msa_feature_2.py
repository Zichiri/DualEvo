#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版：ESM-MSA 特征提取 (指定 ID 列表版)
改动点：只处理 train_id_2.txt 和 mm0.3_test_id_2.txt 中指定的蛋白 ID
"""
import os, re, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import esm 
from tqdm import tqdm

# ---------- 配置区域 ----------
DEFAULT_IN_PATH = 'features/AF_MSA' 
DEFAULT_OUT_ROOT = "features/MSA_Feature_2"
# 默认的 ID 列表路径
DEFAULT_TRAIN_LIST = 'data/train_mm03/train_id_2.txt'
DEFAULT_TEST_LIST = 'data/test/mm0.3_test_id_2.txt'

OUTPUT_TYPES = ["query", "mean", "weighted"]

# ---------- 预处理逻辑 (Worker 中运行) ----------
LOWER_AND_DOT = re.compile(r"[a-z\.]")
STAR = re.compile(r"\*")

def clean_a3m_seq(s: str) -> str:
    s = LOWER_AND_DOT.sub("", s)
    s = STAR.sub("", s)
    return s

def read_a3m(path: str) -> list:
    seqs = []
    try:
        with open(path, "r") as f:
            header = None
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    header = line[1:].split()[0]
                    seqs.append([header, ""])
                else:
                    seqs[-1][1] += line
    except Exception:
        return []
        
    seqs = [(h, clean_a3m_seq(s)) for h, s in seqs]
    if not seqs: return []
    L = len(seqs[0][1])
    # 过滤长度异常
    seqs = [(h, s) for h, s in seqs if len(s) == L]
    return seqs

# 简单的去重逻辑
def dedup_exact_keep_order(msa):
    seen = set(); out = []
    for h, s in msa:
        if s not in seen:
            seen.add(s); out.append((h, s))
    return out

def subsample_msa(msa, max_m, mode="diverse", id_thresh=0.9):
    if len(msa) <= max_m:
        return msa
    
    if mode == "unique":
        msa = dedup_exact_keep_order(msa)
        return msa[:max_m]
    
    if mode == "first":
        return msa[:max_m]

    if mode == "diverse":
        # 简化版 diverse: 截取前 max_m (若需严格逻辑请替换回原代码)
        return msa[:max_m] 
    
    return msa[:max_m]

# ---------- Dataset 定义 ----------
class MSADataset(Dataset):
    def __init__(self, a3m_files, alphabet, max_m=256, subsample="diverse", out_dirs=None):
        self.files = a3m_files
        self.alphabet = alphabet
        self.max_m = max_m
        self.subsample = subsample
        self.out_dirs = out_dirs
        self.batch_converter = alphabet.get_batch_converter()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        base = os.path.splitext(os.path.basename(path))[0]

        # 检查是否已存在 (Worker 中检查，避免不必要的读取)
        if self.out_dirs:
            all_exist = all(os.path.exists(os.path.join(self.out_dirs[k], f"{base}.npz")) for k in OUTPUT_TYPES)
            if all_exist:
                return None # Skip

        msa = read_a3m(path)
        if not msa:
            return {"error": "empty", "id": base}

        # 子采样
        msa = subsample_msa(msa, max_m=self.max_m, mode=self.subsample)
        
        # Tokenize
        _, _, tokens = self.batch_converter([msa])
        
        return {
            "tokens": tokens.squeeze(0), # (M, L)
            "id": base,
            "M": tokens.shape[1],
            "L": tokens.shape[2]
        }

def custom_collate(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return None
    return batch[0]

# ---------- 核心优化：GPU 向量化权重计算 ----------
def calculate_weights_vectorized(tokens: torch.Tensor, alphabet, weight_id_thresh: float = 0.8) -> torch.Tensor:
    M, L = tokens.shape
    if M == 1:
        return torch.ones(1, device=tokens.device, dtype=torch.float32)

    gap_idx = alphabet.padding_idx 

    token_i = tokens.unsqueeze(1) # (M, 1, L)
    token_j = tokens.unsqueeze(0) # (1, M, L)
    
    is_match = (token_i == token_j) # (M, M, L)
    
    is_gap_i = (token_i == gap_idx)
    is_gap_j = (token_j == gap_idx)
    both_gap = is_gap_i & is_gap_j 
    
    valid_pos = ~both_gap 
    match_pos = is_match & valid_pos

    num_valid = valid_pos.sum(dim=-1).float()
    num_match = match_pos.sum(dim=-1).float()

    identity = num_match / (num_valid + 1e-8)
    
    connects = (identity >= weight_id_thresh).float().sum(dim=1)
    
    weights = 1.0 / connects
    weights = weights / weights.sum()
    
    return weights

# ---------- 新增：读取 ID 文件的辅助函数 ----------
def load_target_ids(file_paths):
    """读取多个 TXT 文件中的 ID，合并去重"""
    ids = set()
    for fpath in file_paths:
        if os.path.exists(fpath):
            print(f"Reading IDs from: {fpath}")
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ids.add(line)
        else:
            print(f"Warning: ID file not found: {fpath}")
    return sorted(list(ids))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a3m_dir", type=str, default=DEFAULT_IN_PATH, help="A3M 文件所在的文件夹")
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    # 新增参数：ID 列表文件路径
    p.add_argument("--train_list", type=str, default=DEFAULT_TRAIN_LIST, help="训练集 ID 列表路径")
    p.add_argument("--test_list", type=str, default=DEFAULT_TEST_LIST, help="测试集 ID 列表路径")
    
    p.add_argument("--max_m", type=int, default=256)
    p.add_argument("--subsample", choices=["first","unique","diverse"], default="first")
    p.add_argument("--weight_id_thresh", type=float, default=0.8)
    p.add_argument("--num_workers", type=int, default=20, help="CPU预处理进程数")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # 准备输出目录
    out_dirs = {k: os.path.join(args.out_root, k) for k in OUTPUT_TYPES}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    # 1. 加载所有目标 ID
    print("Loading target IDs...")
    target_ids = load_target_ids([args.train_list, args.test_list])
    print(f"Total unique IDs to process: {len(target_ids)}")

    # 2. 构建对应的 A3M 文件路径列表
    # 不再扫描整个文件夹，而是检查目标 ID 是否存在
    a3m_files = []
    missing_count = 0
    for pid in target_ids:
        # 假设文件名是 ID.a3m，如果你的后缀不同（如 .txt 或 .aln），请在这里修改
        file_path = os.path.join(args.a3m_dir, f"msa_{pid}.a3m")
        if os.path.exists(file_path):
            a3m_files.append(file_path)
        else:
            # 记录缺失的文件，可选打印
            print(f"Missing a3m file for ID: {pid}") 
            missing_count += 1
    
    print(f"Found {len(a3m_files)} valid a3m files.")
    if missing_count > 0:
        print(f"Warning: {missing_count} IDs do not have corresponding .a3m files in {args.a3m_dir}")

    if not a3m_files:
        print("No valid files found. Exiting.")
        return

    # 加载模型
    print(f"Loading ESM-MSA-1b on {args.device}...")
    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    model.eval().to(args.device)

    # 构建 Dataset 和 DataLoader
    dataset = MSADataset(
        a3m_files, # 传入筛选后的列表
        alphabet, 
        max_m=args.max_m, 
        subsample=args.subsample, 
        out_dirs=out_dirs
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=custom_collate,
        pin_memory=True
    )

    print("Starting processing...")
    
    use_amp = torch.cuda.is_available()
    
    for batch in tqdm(dataloader, total=len(dataset)):
        if batch is None or "error" in batch:
            continue
            
        base = batch['id']
        tokens = batch['tokens'].to(args.device) # (M, L)
        M, L = tokens.shape
        
        tokens_in = tokens.unsqueeze(0)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(tokens_in, repr_layers=[12], need_head_weights=False)
                reps = out["representations"][12].squeeze(0)
            
            reps = reps.float()

            results = {}
            results["query"] = reps[0].cpu().numpy()
            results["mean"] = reps.mean(dim=0).cpu().numpy()

            weights = calculate_weights_vectorized(tokens, alphabet, args.weight_id_thresh)
            feat_weighted = (weights.view(M, 1, 1) * reps).sum(dim=0)
            results["weighted"] = feat_weighted.cpu().numpy()

        common_meta = {
            "L": int(L), 
            "D": 768, 
            "M_used": int(M), 
            "protein_id": base
        }

        for k in OUTPUT_TYPES:
            save_path = os.path.join(out_dirs[k], f"{base}.npz")
            np.savez(save_path, emb=results[k], pool=k, **common_meta)

if __name__ == "__main__":
    main()