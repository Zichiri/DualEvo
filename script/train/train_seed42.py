# 修改：训练集路径,测试集路径,log_file文件名,tensorboardX路径名,模型保存路径
import torch, random, os
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from model import GTM
import csv
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score,precision_score, recall_score, f1_score, matthews_corrcoef, precision_recall_curve
from tensorboardX import SummaryWriter
config = {
    'hidden_unit': 64,
    'fc_layer': 2,
    'self_atten_layer': 2,
    'attention_heads': 4,
    'num_neighbor': 30,
    'fc_dropout': 0.2,
    'attention_dropout': 0,
    'class_num': 1,
    'node_dim':1166+3*768,
    'threshold':0.200713# 是否是结合位点的阈值
}
def Seed_everything(seed=2025):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class ProteinDataset(Dataset):
    def __init__(self, protein_ids, esmc_path, dssp_path, map_path, msa_path, labels_dict):
        self.protein_ids = protein_ids
        self.esmc_path = esmc_path
        self.dssp_path = dssp_path
        self.map_path = map_path
        self.msa_path = msa_path
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        ID = self.protein_ids[idx]
        node_features, dismap, masks = prepare_features(self.esmc_path, self.dssp_path, self.map_path, self.msa_path, ID)
        label = torch.tensor(self.labels_dict[ID], dtype=torch.float)  # [L]
        return  node_features.squeeze(0), dismap.squeeze(0), masks.squeeze(0), label  # 去掉 batch 维度
def train_model(train_ids, val_ids,test_ids, esmc_path, dssp_path, map_path, msa_path, labels_dict, save_path,
                epochs=20, batch_size=1, lr=1e-4):
    Seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_dataset = ProteinDataset(train_ids, esmc_path, dssp_path, map_path,msa_path, labels_dict)
    val_dataset = ProteinDataset(val_ids, esmc_path, dssp_path, map_path,msa_path, labels_dict)
    test_dataset = ProteinDataset(test_ids,esmc_path, dssp_path, map_path,msa_path, labels_dict)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = GTM(
        protein_in_dim=config['node_dim'],
        protein_out_dim=config['hidden_unit'],
        target_dim=config['class_num'],
        fc_layer_num=config['fc_layer'],
        atten_layer_num=config['self_atten_layer'],
        atten_head=config['attention_heads'],
        num_neighbor=config['num_neighbor'],
        drop_rate1=config['fc_dropout'],
        drop_rate2=config['attention_dropout']
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    log_file = os.path.join(save_path, f"mm03_GTM_MSA_true_seed42_20260116.csv")
    swriter = SummaryWriter('runs/mm03_GTM_MSA_true_seed42_20260116')
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch','AUPR', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'])
    best_aupr = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for node_feat, dist_map, mask, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            node_feat, dist_map, mask, label = node_feat.to(device), dist_map.to(device), mask.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(node_feat, dist_map, mask)  # batch加回来
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {total_loss / len(train_loader):.4f}")
        # tensorboard运行记录
        swriter.add_scalar('Loss/train',total_loss/len(train_loader),epoch)
        # 验证
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for node_feat, dist_map, mask, label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                node_feat, dist_map, mask, label = node_feat.to(device), dist_map.to(device), mask.to(device), label.to(device)
                output = model(node_feat, dist_map, mask).sigmoid().squeeze(0)
                all_labels.extend(label.view(-1).cpu().numpy())# 防止标签的形状是[1,L]
                all_preds.extend(output.view(-1).cpu().numpy())
        pred_bin = [1 if p >= config['threshold'] else 0 for p in all_preds]
        auc = roc_auc_score(all_labels, all_preds)
        aupr = average_precision_score(all_labels, all_preds)
        precision = precision_score(all_labels, pred_bin, zero_division=0)
        recall = recall_score(all_labels, pred_bin, zero_division=0)
        f1 = f1_score(all_labels, pred_bin, zero_division=0)
        mcc = matthews_corrcoef(all_labels, pred_bin)   
        print(f"Validation Metrics - AUC: {auc:.4f}, AUPR: {aupr:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
              f"F1: {f1:.4f}, MCC: {mcc:.4f}")
        # tensorboard运行记录
        swriter.add_scalar('AUPR/val',aupr,epoch)
        swriter.add_scalar('AUC/val',auc,epoch)
        swriter.add_scalar('Precision/val',precision,epoch)
        swriter.add_scalar('Recall/val',recall,epoch)
        swriter.add_scalar('F1/val',f1,epoch)
        swriter.add_scalar('MCC/val',mcc,epoch)

        # scheduler.step()
        # 测试
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for node_feat, dist_map, mask, label in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                node_feat, dist_map, mask, label = node_feat.to(device), dist_map.to(device), mask.to(device), label.to(device)
                output = model(node_feat, dist_map, mask).sigmoid().squeeze(0)
                all_labels.extend(label.view(-1).cpu().numpy())# 防止标签的形状是[1,L]
                all_preds.extend(output.view(-1).cpu().numpy())
        pred_bin = [1 if p >= config['threshold'] else 0 for p in all_preds]
        auc = roc_auc_score(all_labels, all_preds)
        aupr = average_precision_score(all_labels, all_preds)
        precision = precision_score(all_labels, pred_bin, zero_division=0)
        recall = recall_score(all_labels, pred_bin, zero_division=0)
        f1 = f1_score(all_labels, pred_bin, zero_division=0)
        mcc = matthews_corrcoef(all_labels, pred_bin)   
        print(f"Test Metrics - AUC: {auc:.4f}, AUPR: {aupr:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
              f"F1: {f1:.4f}, MCC: {mcc:.4f}")
        # tensorboard运行记录
        swriter.add_scalar('AUPR/Test',aupr,epoch)
        swriter.add_scalar('AUC/Test',auc,epoch)
        swriter.add_scalar('Precision/Test',precision,epoch)
        swriter.add_scalar('Recall/Test',recall,epoch)
        swriter.add_scalar('F1/Test',f1,epoch)
        swriter.add_scalar('MCC/Test',mcc,epoch) 
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1,aupr, auc, precision, recall, f1, mcc])
        if aupr > best_aupr:
            best_aupr = aupr
            torch.save(model.state_dict(), os.path.join(save_path, f'mm03_GTM_MSA_true_seed42_20260116.ckpt'))
            print(f"Best model saved at epoch {epoch + 1} with AUPR {aupr:.4f}")
    swriter.close()
    print(f"Training completed. Best AUPR: {best_aupr:.4f}")
    
def prepare_features(esmc_path, dssp_path, map_path, msa_path, ID):
    # --- 1. 加载 ESMC ---
    try:
        raw_esmc = np.load(esmc_path + f'{ID}.npy')
        # 强制转 float，如果报错说明文件里有非数字
        esmc = raw_esmc[np.newaxis, :, :].astype(np.float32)[:, 1:-1, :]
    except Exception as e:
        print(f"[Error] ESMC loading failed for {ID}: {e}")
        raise e

    # --- 2. 加载 DSSP ---
    try:
        raw_dssp = np.load(dssp_path + f'{ID}_dssp.npy')
        # 检查 DSSP 是否包含字符
        if raw_dssp.dtype.kind in {'U', 'S', 'O'}: 
            print(f"[Error] DSSP file for {ID} contains strings/objects, not numbers! dtype={raw_dssp.dtype}")
            # 这里可能需要你检查数据预处理步骤，确保保存的是 One-hot 编码
        dssp = np.expand_dims(raw_dssp, axis=0).astype(np.float32)
    except Exception as e:
        print(f"[Error] DSSP loading/conversion failed for {ID}. Raw dtype: {raw_dssp.dtype}")
        raise e

    # --- 3. 第一次拼接 (ESMC + DSSP) ---
    try:
        # 检查长度是否对齐
        if esmc.shape[1] != dssp.shape[1]:
            print(f"[Error] Shape Mismatch for {ID}: ESMC={esmc.shape}, DSSP={dssp.shape}")
            # 尝试通过截断对齐（仅作临时修复，建议检查原始数据）
            min_len = min(esmc.shape[1], dssp.shape[1])
            esmc = esmc[:, :min_len, :]
            dssp = dssp[:, :min_len, :]
            
        node_features = np.concatenate([esmc, dssp], axis=2)
    except Exception as e:
        print(f"[Error] Concatenation (ESMC+DSSP) failed for {ID}: {e}")
        raise e

    L = node_features.shape[1]

    # --- 4. 加载 MSA ---
    msa_feats = []
    msa_sources = ['query', 'mean', 'weighted']
    for folder in msa_sources:
        msa_file = os.path.join(msa_path, folder, f"msa_{ID}.npz")
        msa_emb = None
        if os.path.exists(msa_file):
            try:
                msa_data = np.load(msa_file, allow_pickle=True)
                # 注意：这里切片 [1:, :] 可能会导致长度与 L 不一致
                temp_emb = msa_data['emb'][1:, :] 
                
                # 强制长度对齐检查
                if temp_emb.shape[0] != L:
                    print(f"[Warning] MSA len {temp_emb.shape[0]} != Node len {L} for {ID}. Truncating/Padding.")
                    if temp_emb.shape[0] > L:
                        temp_emb = temp_emb[:L, :]
                    else:
                        pad = np.zeros((L - temp_emb.shape[0], temp_emb.shape[1]))
                        temp_emb = np.concatenate([temp_emb, pad], axis=0)
                
                msa_emb = temp_emb.astype(np.float32)

            except Exception as e:
                print(f"[Warning] Failed to load {msa_file}, using zeros. Error: {e}")
        
        if msa_emb is None:
            print(f"[Warning] Missing MSA file: {msa_file}, using zeros.")
            msa_emb = np.zeros((L, 768), dtype=np.float32)
        
        msa_feats.append(msa_emb)

    # --- 5. 第二次拼接 (Features + MSA) ---
    msa_concat = np.concatenate(msa_feats, axis=-1).astype(np.float32) # [L, 2304]
    msa_concat = np.expand_dims(msa_concat, axis=0)  # [1, L, 2304]
    
    node_features = np.concatenate([node_features, msa_concat], axis=2)

    # --- 6. 加载 Dismap ---
    dismap = np.load(map_path + f'{ID}_dismap.npy')
    # 确保 map 也是 float
    dismap = dismap.astype(np.float32)

    # --- 7. 转 Tensor ---
    # 此时 node_features 一定是 float32，如果前面没报错的话
    node_features = torch.tensor(node_features, dtype=torch.float)
    dismap = torch.tensor(dismap, dtype=torch.float).unsqueeze(0)
    masks = torch.ones(node_features.shape[1], dtype=torch.long).unsqueeze(0)

    return node_features, dismap, masks

def test_model(test_ids, esmc_path, dssp_path, map_path, msa_path, labels_dict, model_path, batch_size=1,):
    # 在测试集上按照保存的模型进行测试
    Seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    test_dataset = ProteinDataset(test_ids,esmc_path, dssp_path, map_path,msa_path, labels_dict)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = GTM(
        protein_in_dim=config['node_dim'],
        protein_out_dim=config['hidden_unit'],
        target_dim=config['class_num'],
        fc_layer_num=config['fc_layer'],
        atten_layer_num=config['self_atten_layer'],
        atten_head=config['attention_heads'],
        num_neighbor=config['num_neighbor'],
        drop_rate1=config['fc_dropout'],
        drop_rate2=config['attention_dropout']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for node_feat, dist_map, mask, label in tqdm(test_loader, desc=f"Predict threshold:{config['threshold']}", leave=False):
            node_feat, dist_map, mask, label = node_feat.to(device), dist_map.to(device), mask.to(device), label.to(device)
            output = model(node_feat, dist_map, mask).sigmoid().squeeze(0)
            all_labels.extend(label.view(-1).cpu().numpy())# 防止标签的形状是[1,L]
            all_preds.extend(output.view(-1).cpu().numpy())
    pred_bin = [1 if p >= config['threshold'] else 0 for p in all_preds]
    auc = roc_auc_score(all_labels, all_preds)
    aupr = average_precision_score(all_labels, all_preds)
    precision = precision_score(all_labels, pred_bin, zero_division=0)
    recall = recall_score(all_labels, pred_bin, zero_division=0)
    f1 = f1_score(all_labels, pred_bin, zero_division=0)
    mcc = matthews_corrcoef(all_labels, pred_bin)   
    print(f"Test Metrics - AUC: {auc:.4f}, AUPR: {aupr:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"F1: {f1:.4f}, MCC: {mcc:.4f}")
    
    # 计算什么阈值下f1分数最高
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_preds)
    # 计算 F1
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator!=0)
    # 找到最大 F1 对应的下标
    best_index = np.argmax(f1_scores)
    if best_index < len(thresholds):
        final_threshold = thresholds[best_index]
    pred_bin = [1 if p >= final_threshold else 0 for p in all_preds]
    auc = roc_auc_score(all_labels, all_preds)
    aupr = average_precision_score(all_labels, all_preds)
    precision = precision_score(all_labels, pred_bin, zero_division=0)
    recall = recall_score(all_labels, pred_bin, zero_division=0)
    f1 = f1_score(all_labels, pred_bin, zero_division=0)
    mcc = matthews_corrcoef(all_labels, pred_bin)   
    print(f"Test Metrics Best F1 - AUC: {auc:.4f}, AUPR: {aupr:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"F1: {f1:.4f}, MCC: {mcc:.4f}, threshold: {final_threshold:.4f}")

if __name__ == "__main__":
    IDs_path = 'data/train_mm03/train_id_2.txt'
    testid_path = 'data/test/mm0.3_test_id_2.txt'
    with open(IDs_path, 'r', encoding='utf-8') as f:
        IDs = [line.strip() for line in f]
    # IDs = IDs[:10] # 简单验证代码是否有错误
    # 划分成训练集和验证集
    Seed_everything(42)
    random.shuffle(IDs)
    split_idx = int(len(IDs)*0.8)
    train_ids = IDs[:split_idx]
    val_ids = IDs[split_idx:]
    with open(testid_path,'r',encoding='utf-8') as f:
        test_ids = [line.strip() for line in f]
    esmc_path = './features/esmc_npy/'
    dssp_path = './features/DSSP/'
    map_path = './features/map/'
    save_path = './checkpoints/models/'
    label_path = './labels/labels_dict.pkl'
    # msa_path = './features/MSA_Feature/'
    msa_path = 'features/MSA_Feature_2/'
    with open(label_path, 'rb') as f:
        labels_dict = pickle.load(f)
    train_model(train_ids, val_ids,test_ids, esmc_path, dssp_path, map_path, msa_path, labels_dict, save_path, epochs=40, batch_size=1, lr=1e-4)
    # test_model(model_path='checkpoints/models/mm03_GTM_MSA_seed42_20251228.ckpt', test_ids=test_ids, esmc_path=esmc_path, dssp_path=dssp_path, map_path=map_path, msa_path=msa_path, labels_dict=labels_dict)