"""
训练脚本：训练蛋白质靶点对比学习模型
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from model import ProteinContrastiveModel, pdb_to_graph


class ProteinPairDataset(Dataset):
    """蛋白质对数据集"""
    def __init__(self, pairs_csv, metadata_csv, cache_dir='data/cache'):
        self.pairs = pd.read_csv(pairs_csv)
        self.metadata = pd.read_csv(metadata_csv).set_index('pdb_id')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 预处理所有PDB文件为图
        self.graph_cache = {}
        print("预处理PDB文件...")
        for pdb_id in tqdm(self.metadata.index.unique()):
            self._load_or_create_graph(pdb_id)
    
    def _load_or_create_graph(self, pdb_id):
        """加载或创建图数据"""
        if pdb_id in self.graph_cache:
            return self.graph_cache[pdb_id]
        
        cache_file = self.cache_dir / f"{pdb_id}.pt"
        
        # 尝试从缓存加载
        if cache_file.exists():
            try:
                graph = torch.load(cache_file)
                self.graph_cache[pdb_id] = graph
                return graph
            except:
                pass
        
        # 创建新的图
        try:
            pdb_file = self.metadata.loc[pdb_id, 'pdb_file']
            graph = pdb_to_graph(pdb_file)
            
            if graph is not None:
                torch.save(graph, cache_file)
                self.graph_cache[pdb_id] = graph
                return graph
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
        
        return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        
        pdb1 = row['pdb1']
        pdb2 = row['pdb2']
        label = row['label']
        
        graph1 = self._load_or_create_graph(pdb1)
        graph2 = self._load_or_create_graph(pdb2)
        
        if graph1 is None or graph2 is None:
            # 如果加载失败，返回下一个样本
            return self.__getitem__((idx + 1) % len(self))
        
        return graph1, graph2, label, pdb1, pdb2


def collate_fn(batch):
    """自定义batch整理函数"""
    graphs1, graphs2, labels, pdb1_ids, pdb2_ids = zip(*batch)
    
    # 将两个batch合并成一个大batch
    # 这样可以同时处理所有蛋白质
    all_graphs = list(graphs1) + list(graphs2)
    batch_data = Batch.from_data_list(all_graphs)
    
    # 标签：前半部分和后半部分配对
    # 相同的label表示是正样本对
    labels = torch.tensor(labels, dtype=torch.long)
    
    # 创建配对标签：每个样本对分配一个唯一ID
    batch_size = len(labels)
    pair_labels = torch.arange(batch_size).repeat(2)  # [0,1,2,...,0,1,2,...]
    
    return batch_data, pair_labels, labels, pdb1_ids, pdb2_ids


def train_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_data, pair_labels, true_labels, _, _ in pbar:
        batch_data = batch_data.to(device)
        pair_labels = pair_labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        loss, embeddings = model(batch_data, pair_labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, dataloader, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, pair_labels, true_labels, _, _ in tqdm(dataloader, desc='Validation'):
            batch_data = batch_data.to(device)
            pair_labels = pair_labels.to(device)
            
            loss, embeddings = model(batch_data, pair_labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(true_labels.tolist())
    
    avg_loss = total_loss / num_batches
    
    # 计算准确率：通过余弦相似度判断
    all_embeddings = torch.cat(all_embeddings, dim=0)
    batch_size = len(all_labels)
    
    # 前半部分和后半部分配对
    emb1 = all_embeddings[:batch_size]
    emb2 = all_embeddings[batch_size:]
    
    similarities = torch.sum(emb1 * emb2, dim=1)
    predictions = (similarities > 0.5).long().numpy()
    accuracy = (predictions == np.array(all_labels)).mean()
    
    return avg_loss, accuracy


def plot_training_history(history, save_path='checkpoints/training_history.png'):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"训练历史已保存到 {save_path}")


def main(args):
    """主训练函数"""
    print("=" * 60)
    print("蛋白质靶点对比学习 - 训练")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 加载数据
    print("\n加载数据...")
    train_dataset = ProteinPairDataset(
        'data/processed/train.csv',
        'data/processed/metadata.csv'
    )
    val_dataset = ProteinPairDataset(
        'data/processed/val.csv',
        'data/processed/metadata.csv'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows上设为0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    model = ProteinContrastiveModel(
        node_feat_dim=25,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    ).to(device)
    
    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"✓ 保存最佳模型 (Val Loss: {val_loss:.4f})")
        
        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
    
    # 绘制训练历史
    plot_training_history(history, os.path.join(args.checkpoint_dir, 'training_history.png'))
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练蛋白质靶点对比学习模型')
    
    # 数据参数
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--output_dim', type=int, default=256, help='输出嵌入维度')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    # 其他参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='checkpoint保存目录')
    
    args = parser.parse_args()
    
    main(args)
