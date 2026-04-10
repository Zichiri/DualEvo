# 得到esm2模型预测的蛋白质级别的embedding
#!/usr/bin/env python3
# usage: python get_esm2_emb.py file1 file2 swiss.fasta
import sys, re, pickle, gzip
from pathlib import Path
import torch
import esm
from tqdm import tqdm

# ---------- 1. 命令行 ----------
assert len(sys.argv) == 4, "usage: python get_esm2_emb.py file1 file2 swiss.fasta"
f1, f2, fasta_file = map(Path, sys.argv[1:4])

# ---------- 2. 收集匹配蛋白（带进度条） ----------
print("Collecting matched proteins …")
match_ids = set()
for f in tqdm((f1, f2), desc="Files"):
    with open(f) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            query_id, match_id, sim = parts[0], parts[1], parts[2]
            if query_id == match_id and float(sim) == 1.0:
                continue
            match_ids.add(match_id)

print(f"Total unique matched proteins (after filtering self-match): {len(match_ids)}")

# ---------- 3. 解析 fasta（带进度条） ----------
def parse_fasta(fa_path, wanted):
    wanted = set(wanted)
    seq_dict = {}
    header_re = re.compile(r'^>sp\|([^|]+)\|')
    current_id = None
    seq_lines = []
    opener = gzip.open if str(fa_path).endswith('.gz') else open

    # 先统计总行数，便于进度条
    total_lines = sum(1 for _ in opener(fa_path, 'rt'))
    with opener(fa_path, 'rt') as fh:
        for line in tqdm(fh, total=total_lines, desc="Parsing fasta"):
            line = line.rstrip()
            if line.startswith('>'):
                if current_id and current_id in wanted:
                    seq_dict[current_id] = ''.join(seq_lines)
                m = header_re.match(line)
                current_id = m.group(1) if m else None
                seq_lines = []
            else:
                if current_id:
                    seq_lines.append(line)
    if current_id and current_id in wanted:
        seq_dict[current_id] = ''.join(seq_lines)
    return seq_dict

sequences = parse_fasta(fasta_file, match_ids)
print(f"Found sequences for {len(sequences)} / {len(match_ids)} proteins")
missing = match_ids - set(sequences)
if missing:
    print("Warning: missing sequences for", missing)

# ---------- 4. 加载 ESM-2 ----------
print("Loading ESM-2 …")
model_name = "esm2_t33_650M_UR50D"
model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
batch_converter = alphabet.get_batch_converter()

# ---------- 5. 计算 embedding（带进度条） ----------
batch_size = 32
ids = list(sequences.keys())
embeddings = {}

for i in tqdm(range(0, len(ids), batch_size), desc="ESM-2 forward"):
    batch_ids = ids[i:i+batch_size]
    data = [(pid, sequences[pid]) for pid in batch_ids]
    _, _, batch_tokens = batch_converter(data)
    if torch.cuda.is_available():
        batch_tokens = batch_tokens.cuda()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        token_repr = results["representations"][33]
    lens = (batch_tokens != alphabet.padding_idx).sum(1)
    for j, pid in enumerate(batch_ids):
        emb = token_repr[j, 1:lens[j]-1].mean(0).cpu()
        embeddings[pid] = emb

print(f"Computed embeddings for {len(embeddings)} proteins")

# ---------- 6. 保存 ----------
with open("embedding.pkl", "wb") as fh:
    pickle.dump(embeddings, fh)
print("Saved -> embedding.pkl")