"""
评估和可视化脚本：评估训练好的模型
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm
import umap

from model import ProteinContrastiveModel, pdb_to_graph
from train import ProteinPairDataset, collate_fn
from torch.utils.data import DataLoader


def compute_embeddings(model, dataset, device):
    """计算所有蛋白质的嵌入向量"""
    model.eval()
    
    embeddings_dict = {}
    
    print("计算嵌入向量...")
    for pdb_id in tqdm(dataset.metadata.index.unique()):
        graph = dataset._load_or_create_graph(pdb_id)
        if graph is not None:
            graph = graph.to(device)
            with torch.no_grad():
                embedding = model.get_embedding(graph)
            embeddings_dict[pdb_id] = embedding.cpu().numpy()
    
    return embeddings_dict


def visualize_embeddings(embeddings_dict, metadata, save_path='results/embeddings_visualization.png', method='tsne'):
    """可视化嵌入空间"""
    
    # 准备数据
    pdb_ids = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[pid] for pid in pdb_ids])
    embeddings = embeddings.squeeze()
    
    categories = [metadata.loc[pid, 'category'] for pid in pdb_ids]
    
    # 降维
    print(f"使用{method.upper()}降维到2D...")
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(pdb_ids)-1))
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # 绘图
    plt.figure(figsize=(12, 8))
    
    # 为每个类别分配颜色
    unique_categories = list(set(categories))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
    color_map = dict(zip(unique_categories, colors))
    
    for category in unique_categories:
        mask = np.array(categories) == category
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color_map[category]],
            label=category,
            s=200,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5
        )
        
        # 标注PDB ID
        for i, (x, y) in enumerate(embeddings_2d[mask]):
            pdb_id = pdb_ids[np.where(mask)[0][i]]
            plt.annotate(
                pdb_id,
                (x, y),
                fontsize=9,
                ha='center',
                va='bottom'
            )
    
    plt.xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    plt.title('Protein Binding Site Embeddings', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存到: {save_path}")
    plt.close()


def evaluate_retrieval(embeddings_dict, test_pairs, k_values=[1, 3, 5]):
    """评估检索性能"""
    
    print("\n评估检索性能...")
    
    results = {f'recall@{k}': [] for k in k_values}
    results['mrr'] = []  # Mean Reciprocal Rank
    
    for _, row in tqdm(test_pairs.iterrows(), total=len(test_pairs)):
        query_id = row['pdb1']
        target_id = row['pdb2']
        is_positive = row['label']
        
        if query_id not in embeddings_dict or target_id not in embeddings_dict:
            continue
        
        query_emb = embeddings_dict[query_id]
        
        # 计算与所有其他蛋白质的相似度
        similarities = []
        pdb_ids = []
        
        for pdb_id, emb in embeddings_dict.items():
            if pdb_id != query_id:
                sim = np.dot(query_emb.squeeze(), emb.squeeze())
                similarities.append(sim)
                pdb_ids.append(pdb_id)
        
        # 排序
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_pdb_ids = [pdb_ids[i] for i in sorted_indices]
        
        # 计算指标
        if is_positive == 1:
            # 找到目标的排名
            if target_id in sorted_pdb_ids:
                rank = sorted_pdb_ids.index(target_id) + 1
                results['mrr'].append(1.0 / rank)
                
                for k in k_values:
                    if target_id in sorted_pdb_ids[:k]:
                        results[f'recall@{k}'].append(1.0)
                    else:
                        results[f'recall@{k}'].append(0.0)
    
    # 计算平均值
    metrics = {}
    for key, values in results.items():
        if len(values) > 0:
            metrics[key] = np.mean(values)
        else:
            metrics[key] = 0.0
    
    return metrics


def plot_similarity_matrix(embeddings_dict, metadata, save_path='results/similarity_matrix.png'):
    """绘制相似度矩阵"""
    
    pdb_ids = sorted(embeddings_dict.keys())
    n = len(pdb_ids)
    
    # 计算相似度矩阵
    sim_matrix = np.zeros((n, n))
    for i, pid1 in enumerate(pdb_ids):
        for j, pid2 in enumerate(pdb_ids):
            emb1 = embeddings_dict[pid1].squeeze()
            emb2 = embeddings_dict[pid2].squeeze()
            sim_matrix[i, j] = np.dot(emb1, emb2)
    
    # 获取类别信息
    categories = [metadata.loc[pid, 'category'] for pid in pdb_ids]
    
    # 绘制热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        sim_matrix,
        xticklabels=pdb_ids,
        yticklabels=pdb_ids,
        cmap='RdYlBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    plt.title('Pairwise Similarity Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('PDB ID', fontsize=12)
    plt.ylabel('PDB ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"相似度矩阵已保存到: {save_path}")
    plt.close()


def evaluate_classification(model, dataloader, device, save_path='results/classification_results.png'):
    """评估分类性能"""
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_scores = []
    
    print("\n评估分类性能...")
    with torch.no_grad():
        for batch_data, pair_labels, true_labels, _, _ in tqdm(dataloader):
            batch_data = batch_data.to(device)
            
            embeddings = model(batch_data)
            
            batch_size = len(true_labels)
            emb1 = embeddings[:batch_size]
            emb2 = embeddings[batch_size:]
            
            # 计算相似度
            similarities = torch.sum(emb1 * emb2, dim=1).cpu().numpy()
            
            # 预测（阈值=0.5）
            predictions = (similarities > 0.5).astype(int)
            
            all_predictions.extend(predictions)
            all_labels.extend(true_labels.numpy())
            all_scores.extend(similarities)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 分类报告
    report = classification_report(all_labels, all_predictions, target_names=['Negative', 'Positive'])
    print("\n分类报告:")
    print(report)
    
    # ROC曲线
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 混淆矩阵
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        ax=axes[0]
    )
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # ROC曲线
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"分类结果已保存到: {save_path}")
    plt.close()
    
    return {
        'accuracy': (all_predictions == all_labels).mean(),
        'auc': roc_auc
    }


def main(args):
    """主评估函数"""
    print("=" * 60)
    print("蛋白质靶点对比学习 - 评估")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    model = ProteinContrastiveModel(
        node_feat_dim=25,
        hidden_dim=128,
        output_dim=256
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载模型: {args.checkpoint}")
    print(f"验证损失: {checkpoint.get('val_loss', 'N/A')}")
    print(f"验证准确率: {checkpoint.get('val_acc', 'N/A')}")
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_dataset = ProteinPairDataset(
        'data/processed/test.csv',
        'data/processed/metadata.csv'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    metadata = pd.read_csv('data/processed/metadata.csv').set_index('pdb_id')
    test_pairs = pd.read_csv('data/processed/test.csv')
    
    # 1. 计算所有蛋白质的嵌入
    embeddings_dict = compute_embeddings(model, test_dataset, device)
    
    # 2. 可视化嵌入空间
    print("\n生成可视化...")
    visualize_embeddings(embeddings_dict, metadata, 'results/embeddings_tsne.png', method='tsne')
    visualize_embeddings(embeddings_dict, metadata, 'results/embeddings_umap.png', method='umap')
    
    # 3. 相似度矩阵
    plot_similarity_matrix(embeddings_dict, metadata)
    
    # 4. 评估检索性能
    retrieval_metrics = evaluate_retrieval(embeddings_dict, test_pairs, k_values=[1, 3, 5])
    print("\n检索性能:")
    for metric, value in retrieval_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 5. 评估分类性能
    classification_metrics = evaluate_classification(model, test_loader, device)
    print("\n分类性能:")
    for metric, value in classification_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 6. 保存结果
    results = {
        **retrieval_metrics,
        **classification_metrics
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('results/evaluation_metrics.csv', index=False)
    print(f"\n评估指标已保存到: results/evaluation_metrics.csv")
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估蛋白质靶点对比学习模型')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt', 
                        help='模型checkpoint路径')
    
    args = parser.parse_args()
    
    main(args)
