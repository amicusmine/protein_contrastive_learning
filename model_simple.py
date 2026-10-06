"""
简化版模型 - 不使用PyTorch Geometric
使用纯PyTorch实现图神经网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')


# 氨基酸字典和性质（与原版相同）
AA_DICT = {
    'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
    'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
    'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
    'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
    'UNK': 20
}

AA_PROPERTIES = {
    'ALA': [1.8, 0, 0, 88.6],    'CYS': [2.5, 0, 1, 108.5],
    'ASP': [-3.5, -1, 1, 111.1], 'GLU': [-3.5, -1, 1, 138.4],
    'PHE': [2.8, 0, 0, 189.9],   'GLY': [-0.4, 0, 0, 60.1],
    'HIS': [-3.2, 0.5, 1, 153.2],'ILE': [4.5, 0, 0, 166.7],
    'LYS': [-3.9, 1, 1, 168.6],  'LEU': [3.8, 0, 0, 166.7],
    'MET': [1.9, 0, 0, 162.9],   'ASN': [-3.5, 0, 1, 114.1],
    'PRO': [-1.6, 0, 0, 112.7],  'GLN': [-3.5, 0, 1, 143.8],
    'ARG': [-4.5, 1, 1, 173.4],  'SER': [-0.8, 0, 1, 89.0],
    'THR': [-0.7, 0, 1, 116.1],  'VAL': [4.2, 0, 0, 140.0],
    'TRP': [-0.9, 0, 0, 227.8],  'TYR': [-1.3, 0, 1, 193.6],
    'UNK': [0, 0, 0, 100]
}


def pdb_to_simple_features(pdb_file, max_residues=100):
    """
    将PDB转换为简单的特征向量（不使用图结构）
    返回固定长度的特征向量
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    residues = []
    coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':
                    try:
                        ca_atom = residue['CA']
                        resname = residue.get_resname()
                        residues.append(resname)
                        coords.append(ca_atom.coord)
                    except:
                        continue
    
    if len(residues) == 0:
        return None
    
    # 截断或填充到固定长度
    coords = np.array(coords)
    
    # 提取特征
    features = []
    for i in range(min(len(residues), max_residues)):
        resname = residues[i]
        
        # one-hot + 物理化学性质
        aa_idx = AA_DICT.get(resname, AA_DICT['UNK'])
        one_hot = np.zeros(21)
        one_hot[aa_idx] = 1
        properties = AA_PROPERTIES.get(resname, AA_PROPERTIES['UNK'])
        
        # 位置编码
        pos_encoding = [i / max_residues]
        
        # 合并特征
        feature = np.concatenate([one_hot, properties, pos_encoding])
        features.append(feature)
    
    # 填充到固定长度
    while len(features) < max_residues:
        features.append(np.zeros(26))  # 21 + 4 + 1
    
    features = np.array(features[:max_residues], dtype=np.float32)
    
    return features


class SimpleProteinEncoder(nn.Module):
    """
    简化的蛋白质编码器 - 使用1D卷积代替GNN
    """
    def __init__(self, input_dim=26, hidden_dim=128, output_dim=256, max_len=100):
        super().__init__()
        
        # 1D卷积层（类似于序列模型）
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 全局池化后的全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        x: [batch_size, max_len, input_dim]
        """
        # 转置为 [batch, channels, length]
        x = x.transpose(1, 2)
        
        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 全局池化
        x_mean = torch.mean(x, dim=2)
        x_max, _ = torch.max(x, dim=2)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 全连接层
        x = self.fc(x)
        
        # L2归一化
        x = F.normalize(x, dim=-1)
        
        return x


class SimpleContrastiveModel(nn.Module):
    """
    简化的对比学习模型
    """
    def __init__(self, input_dim=26, hidden_dim=128, output_dim=256):
        super().__init__()
        
        self.encoder = SimpleProteinEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        self.temperature = 0.07
    
    def forward(self, x, labels=None):
        """
        x: [batch_size, max_len, input_dim]
        labels: [batch_size] 用于对比学习
        """
        embeddings = self.encoder(x)
        
        if labels is not None:
            loss = self.contrastive_loss(embeddings, labels)
            return loss, embeddings
        else:
            return embeddings
    
    def contrastive_loss(self, embeddings, labels):
        """
        InfoNCE损失
        """
        batch_size = embeddings.size(0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 创建正样本mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # InfoNCE损失
        exp_sim = torch.exp(similarity_matrix)
        pos_sim = (exp_sim * mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1) - exp_sim.diag()
        
        loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)
        
        valid_mask = mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device)
        
        return loss
    
    def get_embedding(self, x):
        """获取单个蛋白质的嵌入"""
        self.eval()
        with torch.no_grad():
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            embedding = self.encoder(x)
        return embedding


if __name__ == '__main__':
    print("测试简化版模型...")
    
    # 创建模型
    model = SimpleContrastiveModel(input_dim=26, hidden_dim=64, output_dim=128)
    print(f"✓ 模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 4
    max_len = 100
    input_dim = 26
    
    x = torch.randn(batch_size, max_len, input_dim)
    labels = torch.tensor([0, 0, 1, 1])
    
    loss, embeddings = model(x, labels)
    print(f"✓ 前向传播成功")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {embeddings.shape}")
    print(f"  损失: {loss.item():.4f}")
    
    # 测试推理
    model.eval()
    with torch.no_grad():
        emb = model.get_embedding(x[0])
    print(f"✓ 推理成功")
    print(f"  单个嵌入形状: {emb.shape}")
    
    print("\n✓ 所有测试通过！这个版本不依赖PyTorch Geometric")
