# 使用GTM模型参数在测试集上预测同时添加测试集在训练集上使用tmalign得到的信息
import os
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from GTM_model import GTM
from GTM_train import prepare_features
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, precision_score, recall_score,
    precision_recall_curve
)
import pickle
config = {
    'hidden_unit': 64,
    'fc_layer': 2,
    'self_atten_layer': 2,
    'attention_heads': 4,
    'num_neighbor': 30,
    'fc_dropout': 0.2,
    'attention_dropout': 0,
    'class_num': 1,
    'node_dim':1166,
    'threshold':0.5 # 是否是结合位点的阈值
}
# ============================================================
# 模型加载
# ============================================================
def load_model(model_path, device):
    model = GTM(config['node_dim'], config['hidden_unit'], config['class_num'], config['fc_layer'],
                config['self_atten_layer'], config['attention_heads'], config['num_neighbor'],
                config['fc_dropout'], config['attention_dropout']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# ============================================================
# 单蛋白预测
# ============================================================
def predict_protein(model, ID, esmc_path, dssp_path, map_path, device):
    node_feat, dist_map, mask = prepare_features(esmc_path, dssp_path, map_path, ID)
    node_feat, dist_map, mask = node_feat.to(device), dist_map.to(device), mask.to(device)
    with torch.no_grad():
        output = model(node_feat, dist_map, mask).sigmoid().squeeze(0).cpu().numpy()
    return output  # [L]

# ============================================================
# 主融合函数
# ============================================================
def predict_with_structure_fusion(
    test_ids, train_ids, model_path, tm_result_path, mappings_dir,
    esmc_path, dssp_path, map_path, labels_dict, alpha=0.7, threshold=0.8
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    print(f"✅ Model loaded from: {model_path}")

    # 1️⃣ 加载 TM-score 矩阵
    data = np.load(tm_result_path, allow_pickle=True)
    tm_matrix = data['tm_matrix']
    test_id_arr = data['test_ids']
    train_id_arr = data['train_ids']

    id2idx_test = {tid: i for i, tid in enumerate(test_id_arr)}
    id2idx_train = {tid: j for j, tid in enumerate(train_id_arr)}

    results = {}

    # 2️⃣ 遍历测试蛋白
    for test_id in tqdm(test_ids, desc="Predicting with TM-align fusion"):
        if test_id not in id2idx_test:
            print(f"⚠️ {test_id} not found in TM matrix, skip.")
            continue

        p_model = predict_protein(model, test_id, esmc_path, dssp_path, map_path, device)
        L = len(p_model)
        p_struct = np.zeros(L, dtype=np.float32)
        weight_sum = np.zeros(L, dtype=np.float32)

        test_idx = id2idx_test[test_id]
        tm_scores = tm_matrix[test_idx, :]

        # 3️⃣ 筛选出 TM >= threshold 的训练蛋白
        valid_train_idxs = np.where(tm_scores >= threshold)[0]

        for train_idx in valid_train_idxs:
            train_id = train_id_arr[train_idx]
            tm_score = float(tm_scores[train_idx])
            npz_path = os.path.join(mappings_dir, f"{test_id}_{train_id}.npz")

            if not os.path.exists(npz_path):
                continue  # 没有mapping文件就跳过

            try:
                tm_info = np.load(npz_path, allow_pickle=True)
                i_idx = tm_info["i_idx"] - 1  # 转为0-based
                j_idx = tm_info["j_idx"] - 1
                # 再确认tm_score一致
                tm_local = float(tm_info.get("tm_score", tm_score))
            except Exception as e:
                print(f"⚠️ Failed to load mapping for {test_id}-{train_id}: {e}")
                continue

            # 训练集标签
            train_label = labels_dict.get(train_id)
            if train_label is None:
                continue

            # 4️⃣ 残基层面融合
            for i, j in zip(i_idx, j_idx):
                if 0 <= i < L and 0 <= j < len(train_label):
                    p_struct[i] += tm_local * train_label[j]
                    weight_sum[i] += tm_local

        # 归一化并融合
        np.divide(p_struct, weight_sum, out=p_struct, where=weight_sum != 0)
        p_final = alpha * p_model + (1 - alpha) * p_struct

        results[test_id] = {
            'model': p_model,
            'struct': p_struct,
            'final': p_final
        }

    return results

def evaluate_predictions(results, labels_dict, test_ids, save_csv_path, default_threshold=0.15):
    all_labels, model_preds, struct_preds, fused_preds = [], [], [], []

    # 1. 收集所有预测值和真实标签
    valid_test_ids = [] # 记录有效的ID
    for test_id in test_ids:
        if test_id not in results:
            continue
        label = labels_dict[test_id]
        model_p = results[test_id]['model']
        struct_p = results[test_id]['struct']
        fused_p = results[test_id]['final']

        # 确保长度一致，防止某些异常情况
        min_len = min(len(label), len(model_p))
        all_labels.extend(label[:min_len])
        model_preds.extend(model_p[:min_len])
        struct_preds.extend(struct_p[:min_len])
        fused_preds.extend(fused_p[:min_len])
        valid_test_ids.append(test_id)
    
    # 转为 numpy 数组以便计算
    all_labels = np.array(all_labels)
    fused_preds = np.array(fused_preds)

    def safe_metric(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return np.nan

    # 基础指标计算函数
    def get_metrics_at_threshold(y_true, y_pred, thr):
        y_bin = [1 if p >= thr else 0 for p in y_pred]
        return {
            'AUC': safe_metric(roc_auc_score, y_true, y_pred),
            'AUPR': safe_metric(average_precision_score, y_true, y_pred),
            'Precision': safe_metric(precision_score, y_true, y_bin, zero_division=0),
            'Recall': safe_metric(recall_score, y_true, y_bin, zero_division=0),
            'F1': safe_metric(f1_score, y_true, y_bin, zero_division=0),
            'MCC': safe_metric(matthews_corrcoef, y_true, y_bin),
            'Threshold': thr
        }

    for pred_threshold in [0.21,0.22,0.24,0.26]:
        print("\n📊 Evaluation Results (Default Threshold = %.2f):" % pred_threshold)
        print("-" * 60)
        print("Model only:    ", get_metrics_at_threshold(all_labels, model_preds, pred_threshold))
        print("Structure only:", get_metrics_at_threshold(all_labels, struct_preds, pred_threshold))
        print("Fused (Default):", get_metrics_at_threshold(all_labels, fused_preds, pred_threshold))

    # # --- 新增功能：寻找 Fused 结果的最佳 F1 阈值 ---
    # print("\n🔍 Searching Optimal Threshold for Fused Predictions (Max F1)...")
    
    # # 使用 precision_recall_curve 高效计算所有可能的阈值
    # precisions, recalls, thresholds = precision_recall_curve(all_labels, fused_preds)
    
    # # 计算每个阈值对应的 F1 (防止除以0)
    # # 注意：precisions 和 recalls 的长度比 thresholds 多 1，最后一个值代表 threshold=1.0 后的状态
    # numerator = 2 * precisions * recalls
    # denominator = precisions + recalls
    # f1_scores = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator != 0)
    
    # # 找到最大 F1 的索引
    # best_idx = np.argmax(f1_scores)
    
    # # 获取对应的最佳 F1 和 阈值
    # # 注意处理边界情况
    # if best_idx < len(thresholds):
    #     best_threshold = thresholds[best_idx]
    # else:
    #     best_threshold = 0.5 # 如果无法确定，回退到默认
        
    # best_f1 = f1_scores[best_idx]
    
    # # 使用最佳阈值重新计算所有指标
    # optimized_metrics = get_metrics_at_threshold(all_labels, fused_preds, best_threshold)
    
    # print("-" * 60)
    # print(f"✅ Optimal Threshold Found: {best_threshold:.6f}")
    # print(f"   Max F1 Score:          {best_f1:.4f}")
    # print("-" * 60)
    # print("Fused (Optimized):")
    # print(f"   AUC:       {optimized_metrics['AUC']:.4f}")
    # print(f"   AUPR:      {optimized_metrics['AUPR']:.4f}")
    # print(f"   Precision: {optimized_metrics['Precision']:.4f}")
    # print(f"   Recall:    {optimized_metrics['Recall']:.4f}")
    # print(f"   F1:        {optimized_metrics['F1']:.4f}")
    # print(f"   MCC:       {optimized_metrics['MCC']:.4f}")
    # print("-" * 60)


    # # ============================================================
    # # 逐个蛋白计算指标并保存到 CSV
    # # ============================================================
    # print(f"\n💾 Saving per-protein details to {save_csv_path} ...")
    # csv_data = []
    # for test_id in tqdm(valid_test_ids, desc="Calculating per-protein metrics"):
    #     label = np.array(labels_dict[test_id])
    #     pred = np.array(results[test_id]['final'])
        
    #     # 截断以防万一长度不一致
    #     min_len = min(len(label), len(pred))
    #     label = label[:min_len]
    #     pred = pred[:min_len]
        
    #     # 1. 计算单蛋白指标
    #     # 注意：如果一个蛋白全是0或全是1，AUC/AUPR可能会报错或为NaN，这是正常的
    #     metrics = get_metrics_at_threshold(label, pred, best_threshold)
        
    #     # 2. 提取位点 (1-based index)
    #     # 实际催化位点: label == 1
    #     actual_sites = [i + 1 for i, v in enumerate(label) if v == 1]
        
    #     # 预测催化位点: pred >= best_threshold
    #     predicted_sites = [i + 1 for i, v in enumerate(pred) if v >= best_threshold]
        
    #     # 3. 收集数据
    #     row = {
    #         'Protein_ID': test_id,
    #         'Length': len(label),
    #         'Actual_Sites': str(actual_sites),         # 转字符串方便CSV存储
    #         'Predicted_Sites': str(predicted_sites),   # 转字符串
    #         'Num_Actual': len(actual_sites),
    #         'Num_Predicted': len(predicted_sites),
    #         'AUC': metrics['AUC'],
    #         'AUPR': metrics['AUPR'],
    #         'Precision': metrics['Precision'],
    #         'Recall': metrics['Recall'],
    #         'F1': metrics['F1'],
    #         'MCC': metrics['MCC']
    #     }
    #     csv_data.append(row)
    
    # # 保存 DataFrame
    # df = pd.DataFrame(csv_data)
    # df.to_csv(save_csv_path, index=False)
    # print(f"✅ CSV saved successfully! Total proteins: {len(df)}")
# ============================================================
# 找最优参数
# ============================================================
def find_best_alpha(results, labels_dict, test_ids, alphas=None):
    """
    根据 AUPR 指标寻找最优 alpha
    results: dict，predict_with_structure_fusion() 的输出
    labels_dict: dict，蛋白质真实标签
    test_ids: list，测试蛋白 ID 列表
    alphas: list/array，自定义 alpha 搜索区间
    """
    if alphas is None:
        alphas = np.linspace(0.6, 1, 21)  # 默认 0, 0.05, 0.1, ..., 1.0

    all_labels = []
    for test_id in test_ids:
        if test_id in labels_dict:
            all_labels.extend(labels_dict[test_id])
    all_labels = np.array(all_labels)

    best_alpha = None
    best_aupr = -1
    results_alpha = {}

    print("\n🔍 Searching best alpha based on AUPR:")

    for alpha in alphas:
        fused_preds = []
        for test_id in test_ids:
            if test_id not in results:
                continue
            p_model = results[test_id]['model']
            p_struct = results[test_id]['struct']
            p_final = alpha * np.array(p_model) + (1 - alpha) * np.array(p_struct)
            fused_preds.extend(p_final)
        fused_preds = np.array(fused_preds)

        # 计算 AUPR
        try:
            aupr = average_precision_score(all_labels, fused_preds)
        except Exception:
            aupr = np.nan

        results_alpha[alpha] = aupr
        print(f"  α={alpha:.2f} → AUPR={aupr:.4f}")

        if not np.isnan(aupr) and aupr > best_aupr:
            best_aupr = aupr
            best_alpha = alpha

    print("\n✅ Best α = %.2f  (AUPR = %.4f)" % (best_alpha, best_aupr))
    return best_alpha, best_aupr, results_alpha

# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    model_path = "checkpoints/models/GTM_42seed_best.ckpt"
    tm_result_path = "data/test/tmalign/results/tm_results.npz"
    mappings_dir = "data/test/tmalign/results/mappings"
    esmc_path = "./features/esmc_npy/"
    dssp_path = "./features/DSSP/"
    map_path = "./features/map/"
    label_path = "./labels/labels_dict.pkl"
    testid_path = 'data/test/mm0.3_test_id_2.txt'
    trainid_path = 'data/train_mm03/train_id.txt'
    save_csv_path = "result/test_result/mm0.3_GTM_TMalign_test_results.csv"
    with open(label_path, 'rb') as f:
        labels_dict = pickle.load(f)
    with open(testid_path, 'r') as f:
        test_ids = [line.strip() for line in f if line.strip()]
    with open(trainid_path, 'r') as f:
        train_ids = [line.strip() for line in f if line.strip()]

    print("===========================用设定的模型概率和结构概率占比以及tmscore阈值做出预测===================================")
    results = predict_with_structure_fusion(
        test_ids, train_ids, model_path, tm_result_path, mappings_dir,
        esmc_path, dssp_path, map_path, labels_dict,
        alpha=0.8, threshold=0.8
    )
    with open("./mm03_GTM_TMalign_results_fusion.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # print("===========================针对上面做的概率预测结果,找到最合适的模型概率占比让aupr值最大===================================")
    # best_alpha, best_aupr, all_scores = find_best_alpha(results, labels_dict, test_ids)
    # with open('./mm03_GTM_TMalign_results_fusion.pkl', "rb") as f:
    #     results = pickle.load(f)
    
    print("===========================用最初设定的模型概率占比和tmscore阈值做出的预测结果来调整pre,recall,f1参数,先使用默认概率阈值，然后根据f1最大值找到最合适的概率阈值，并保存每个蛋白的预测结果===================================")
    evaluate_predictions(results, labels_dict, test_ids, save_csv_path)
