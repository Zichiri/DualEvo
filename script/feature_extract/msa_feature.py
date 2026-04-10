#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 A3M 生成残基级 (L, 768) 特征。
优化版：一次推理，同时生成 Query、Mean、Weighted 三种嵌入。
"""
import os, re, glob, argparse
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import torch
import esm 

# ---------- 配置区域 ----------
DEFAULT_IN_PATH = 'features/AF_MSA' # 输入目录
# 你指定的输出根目录
DEFAULT_OUT_ROOT = "data/squidly_new_test/msa_feature"
OUTPUT_TYPES = ["query", "mean", "weighted"]

# ---------- A3M 预处理 (保持不变) ----------
LOWER_AND_DOT = re.compile(r"[a-z\.]")
STAR = re.compile(r"\*")

def clean_a3m_seq(s: str) -> str:
    """去除 a3m 的插入（小写）和点，保留大写AA与 '-'。"""
    s = LOWER_AND_DOT.sub("", s)
    s = STAR.sub("", s)
    return s

def read_a3m(path: str) -> List[Tuple[str, str]]:
    """读取 A3M，返回 [(header, cleaned_aligned_seq), ...]"""
    seqs = []
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
    # 清理插入
    seqs = [(h, clean_a3m_seq(s)) for h, s in seqs]
    if not seqs:
        return []
    L = len(seqs[0][1])
    # 过滤长度异常的行
    seqs = [(h, s) for h, s in seqs if len(s) == L]
    return seqs

# ---------- 子采样逻辑 (保持不变) ----------
def identity_ignore_gaps(a: str, b: str) -> float:
    """按对齐位点计算序列同一性（忽略双方的 '-'）。"""
    match = 0; comp = 0
    for x, y in zip(a, b):
        if x == '-' and y == '-': continue
        if x != '-' or y != '-':
            comp += 1
            if x == y: match += 1
    return (match / comp) if comp > 0 else 0.0

def dedup_exact_keep_order(msa: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    seen = set(); out = []
    for h, s in msa:
        if s not in seen:
            seen.add(s); out.append((h, s))
    return out

def subsample_msa(msa: List[Tuple[str,str]], max_m: int, mode: str = "first", id_thresh: float = 0.9) -> List[Tuple[str,str]]:
    if len(msa) <= max_m:
        return msa
    if mode == "unique":
        msa = dedup_exact_keep_order(msa)
        return msa[:max_m]
    if mode == "diverse":
        chosen = [msa[0]]
        for h, s in msa[1:]:
            if any(identity_ignore_gaps(s, cs) >= id_thresh for _, cs in chosen):
                continue
            chosen.append((h, s))
            if len(chosen) >= max_m: break
        if len(chosen) < max_m:
            for h, s in msa[1:]:
                if (h, s) not in chosen:
                    chosen.append((h, s))
                    if len(chosen) >= max_m: break
        return chosen
    return msa[:max_m]

# ---------- ESM-MSA 特征提取 (核心优化部分) ----------
def calculate_weights(msa_seqs: List[str], weight_id_thresh: float, device: str) -> torch.Tensor:
    """
    计算加权权重 (Meff 风格)。
    为了效率，这里虽然还是 O(M^2)，但只针对筛选后的 MSA (M<=256)，速度很快。
    """
    M = len(msa_seqs)
    neigh = np.ones((M,), dtype=np.float32)
    # 这里可以使用 numpy 广播或者字符编码加速，但保持原逻辑以确保数值一致性
    for i in range(M):
        si = msa_seqs[i]
        c = 1.0
        for j in range(M):
            if i == j: continue
            if identity_ignore_gaps(si, msa_seqs[j]) >= weight_id_thresh:
                c += 1.0
        neigh[i] = c
    
    # 权重归一化: w_i = (1/k) / sum(1/k)
    w = torch.tensor(1.0 / neigh, dtype=torch.float32, device=device)
    w = w / w.sum()
    return w

@torch.inference_mode()
def process_msa_to_all_features(
        path: str,
        model,
        alphabet,
        max_m: int = 256,
        subsample: str = "diverse",
        weight_id_thresh: float = 0.8,
        device: str = "cuda"
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    核心优化：一次模型前向传播，返回 query, mean, weighted 三种特征。
    """
    msa = read_a3m(path)
    if not msa:
        return None, {"error": "empty_or_bad_a3m"}

    # 1. 子采样
    msa = subsample_msa(msa, max_m=max_m, mode=subsample)
    
    # 2. Tokenize & Move to GPU
    batch_converter = alphabet.get_batch_converter()
    labels, strs, tokens = batch_converter([msa])
    tokens = tokens.to(device) # Shape: (1, M, L)

    # 3. Model Inference (最耗时的一步，只做一次)
    out = model(tokens, repr_layers=[12], need_head_weights=False)
    # reps shape: (1, M, L, D) -> squeeze -> (M, L, D)
    reps = out["representations"][12].squeeze(0) 
    M, L, D = reps.shape

    results = {}

    # 4. 生成 Query 特征 (取第一条序列)
    results["query"] = reps[0].detach().cpu().numpy().astype(np.float32)

    # 5. 生成 Mean 特征 (沿 M 维度平均)
    results["mean"] = reps.mean(dim=0).detach().cpu().numpy().astype(np.float32)

    # 6. 生成 Weighted 特征 (加权平均)
    # 计算权重 (基于 CPU 字符串计算，或者你可以改为 tensor 计算加速，这里保持原逻辑准确性)
    msa_seqs = [s for _, s in msa]
    weights = calculate_weights(msa_seqs, weight_id_thresh, device) # shape (M,)
    # (M,) -> (M, 1, 1) * (M, L, D) -> sum(dim=0) -> (L, D)
    feat_weighted = (weights.view(M, 1, 1) * reps).sum(dim=0)
    results["weighted"] = feat_weighted.detach().cpu().numpy().astype(np.float32)

    return results, {"M_used": M, "L": L, "D": D}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a3m_dir", type=str, default=DEFAULT_IN_PATH, required=True, help="包含 .a3m 的输入目录")
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT, help="输出根目录")
    p.add_argument("--max_m", type=int, default=256, help="每个MSA保留的最大序列数")
    p.add_argument("--subsample", choices=["first","unique","diverse"], default="diverse")
    p.add_argument("--weight_id_thresh", type=float, default=0.8)
    p.add_argument("--token_budget", type=int, default=128_000) # 防止显存溢出
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # 准备输出目录: data/new_test/msa_feature/{query, mean, weighted}
    out_dirs = {k: os.path.join(args.out_root, k) for k in OUTPUT_TYPES}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    # 加载模型
    print(f"Loading ESM-MSA-1b on {args.device}...")
    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    model.eval().to(args.device)

    a3m_files = sorted(glob.glob(os.path.join(args.a3m_dir, "*.a3m")))
    
    print(f"Found {len(a3m_files)} a3m files. Processing...")

    for path in tqdm(a3m_files, desc="Processing MSAs"):
        base = os.path.splitext(os.path.basename(path))[0]
        
        # 检查是否三个文件都已存在，如果都存在则跳过
        all_exist = all(os.path.exists(os.path.join(out_dirs[k], f"{base}.npz")) for k in OUTPUT_TYPES)
        if all_exist:
            continue

        # 预读取获取长度，动态调整 max_m (显存优化)
        # 这里为了速度，我们假设 read_a3m 很快。如果文件巨大，可以只读第一行。
        # 但考虑到 cleaning 逻辑，读全部比较稳妥。
        msa_head = read_a3m(path)
        if not msa_head:
            print(f"Warning: Skipping empty or bad file {base}")
            continue
            
        L = len(msa_head[0][1])
        # 动态调整 M，避免 OOM
        current_max_m = min(args.max_m, max(8, args.token_budget // max(1, L)))

        # 核心处理：一次推理，返回三个字典
        features_dict, meta = process_msa_to_all_features(
            path, model, alphabet,
            max_m=current_max_m,
            subsample=args.subsample,
            weight_id_thresh=args.weight_id_thresh,
            device=args.device
        )

        if features_dict is None:
            # 错误处理：生成带有错误信息的空文件，防止反复读取
            for k in OUTPUT_TYPES:
                np.savez_compressed(os.path.join(out_dirs[k], f"{base}.npz"), error=meta["error"])
            continue

        # 分别保存到三个文件夹
        common_meta = {
            "L": int(meta["L"]), 
            "D": int(meta["D"]), 
            "M_used": int(meta["M_used"]), 
            "protein_id": base
        }

        for k in OUTPUT_TYPES:
            save_path = os.path.join(out_dirs[k], f"{base}.npz")
            np.savez_compressed(
                save_path, 
                emb=features_dict[k], # 对应的特征矩阵
                pool=k,               # 记录类型
                **common_meta
            )

if __name__ == "__main__":
    main()