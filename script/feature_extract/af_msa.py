import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MSA_URL_TEMPLATE = "https://alphafold.ebi.ac.uk/files/msa/AF-{id}-F{variant}-msa_v6.a3m"
failed_ids = []

def download_msa(protein_id, variant, fasta_seq):
    msa_url = MSA_URL_TEMPLATE.format(id=protein_id, variant=variant)
    try:
        response = requests.get(msa_url)
        response.raise_for_status()

        msa_content = response.text.splitlines()
        msa_seq = msa_content[1] if len(msa_content) > 1 else ""

        if msa_seq == fasta_seq:
            msa_file = f"./features/AF_MSA/msa_{protein_id}.a3m"
            with open(msa_file, 'w') as file:
                file.write("\n".join(msa_content))
            print(f"Downloaded MSA for {protein_id} (variant {variant})")
            return True
        else:
            print(f"Sequence mismatch for {protein_id} (variant {variant})")
            return False

    except requests.exceptions.RequestException:
        print(f"Failed to download MSA for {protein_id} (variant {variant})")
        return False

def read_fasta(fasta_file):
    protein_dict = {}
    with open(fasta_file, 'r') as file:
        lines = file.readlines()
        protein_id = None
        sequence = ""

        for line in lines:
            if line.startswith(">"):
                if protein_id:
                    protein_dict[protein_id] = sequence
                # protein_id = line.strip().split()[0][1:]
                # 测试集fasta文件中>后直接跟的就是id
                protein_id = line.strip()[1:]
                sequence = ""
            else:
                sequence += line.strip()
        if protein_id:
            protein_dict[protein_id] = sequence
    return protein_dict

def download_msa_for_proteins(id_file, fasta_file):
    with open(id_file, 'r') as file:
        protein_ids = file.read().splitlines()

    fasta_dict = read_fasta(fasta_file)

    with ThreadPoolExecutor(max_workers=40) as executor:
        future_map = {}

        for protein_id in protein_ids:

            # -------------------------------
            # 🚀 新增逻辑：如果已有 MSA → 跳过
            # -------------------------------
            msa_file = f"./features/AF_MSA/msa_{protein_id}.a3m"
            if os.path.exists(msa_file):
                # print(f"Skip {protein_id}, MSA already exists.")
                continue

            if protein_id not in fasta_dict:
                failed_ids.append(protein_id)
                continue

            fasta_seq = fasta_dict[protein_id]

            # 目前只尝试 F1（你的原逻辑也是）
            future = executor.submit(download_msa, protein_id, 1, fasta_seq)
            future_map[future] = protein_id

        # 处理线程池结果
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Downloading MSA"):
            protein_id = future_map[future]
            ok = future.result()
            if not ok:
                failed_ids.append(protein_id)

    # 保存下载失败的 ID
    with open("failed_proteins.txt", "w") as f:
        for pid in failed_ids:
            f.write(pid + "\n")

if __name__ == "__main__":
    id_file = "data/test/cd04_test_id.txt"
    fasta_file = "data/test/cd04_test.fasta"
    download_msa_for_proteins(id_file, fasta_file)
