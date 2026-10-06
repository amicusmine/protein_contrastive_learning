"""
对比学习模型：蛋白质结合位点编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')


# 氨基酸字典
AA_DICT = {
    'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
    'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
    'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
    'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
    'UNK': 20  # 未知氨基酸
}

# 氨基酸物理化学性质 (疏水性, 电荷, 极性, 体积)
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


def pdb_to_graph(pdb_file, distance_threshold=10.0):
    """
    将PDB文件转换为图数据结构
    
    Args:
        pdb_file: PDB文件路径
        distance_threshold: 残基间距离阈值（Å）
    
    Returns:
        torch_geometric.data.Data: 图数据
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # 提取残基信息
    residues = []
    coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # 只要标准氨基酸
                    try:
                        ca_atom = residue['CA']
                        resname = residue.get_resname()
                        
                        residues.append(resname)
                        coords.append(ca_atom.coord)
                    except:
                        continue
    
    if len(residues) == 0:
        return None
    
    coords = np.array(coords)
    
    # 节点特征: one-hot编码 + 物理化学性质
    node_features = []
    for resname in residues:
        # one-hot编码
        aa_idx = AA_DICT.get(resname, AA_DICT['UNK'])
        one_hot = np.zeros(21)
        one_hot[aa_idx] = 1
        
        # 物理化学性质
        properties = AA_PROPERTIES.get(resname, AA_PROPERTIES['UNK'])
        
        # 合并特征
        feature = np.concatenate([one_hot, properties])
        node_features.append(feature)
    
    node_features = np.array(node_features, dtype=np.float32)
    
    # 构建边: 距离小于阈值的残基对
    num_nodes = len(residues)
    edge_index = []
    edge_attr = []
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(coords[i] - coords[j])
            
            if distance < distance_threshold:
                # 双向边
                edge_index.append([i, j])
                edge_index.append([j, i])
                
                # 边特征: 距离和方向
                direction = coords[j] - coords[i]
                edge_feature = [distance] + direction.tolist()
                edge_attr.append(edge_feature)
                edge_attr.append(edge_feature)
    
    if len(edge_index) == 0:
        # 如果没有边，创建自环
        edge_index = [[i, i] for i in range(num_nodes)]
        edge_attr = [[0, 0, 0, 0] for _ in range(num_nodes)]
    
    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_attr = np.array(edge_attr, dtype=np.float32)
    
    # 创建PyG Data对象
    data = Data(
        x=torch.FloatTensor(node_features),
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_attr),
        coords=torch.FloatTensor(coords)
    )
    
    return data


class ProteinEncoder(nn.Module):
    """
    蛋白质图神经网络编码器
    使用Graph Attention Network (GAT)
    """
    def __init__(self, node_feat_dim=25, hidden_dim=128, output_dim=256, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        
        # GAT层
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(node_feat_dim, hidden_dim, heads=4, concat=True))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
        
        self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=False))
        
        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * 4) for _ in range(num_layers - 1)
        ] + [nn.BatchNorm1d(hidden_dim)])
        
        # 投影头（对比学习的标准做法）
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, data):
        """
        Args:
            data: PyG Data对象
        Returns:
            嵌入向量 [batch_size, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GAT层
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        # 全局池化：mean + max
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = x_mean + x_max
        
        # 投影到对比学习空间
        x = self.projection(x)
        
        # L2归一化
        x = F.normalize(x, dim=-1)
        
        return x


class ContrastiveLoss(nn.Module):
    """
    InfoNCE对比学习损失函数
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [batch_size, embedding_dim] 归一化后的嵌入
            labels: [batch_size] 样本对的标签（相同标签=正样本对）
        
        Returns:
            loss: 对比学习损失
        """
        batch_size = embeddings.size(0)
        
        # 计算相似度矩阵 [batch_size, batch_size]
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 创建正样本mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 去除对角线（自己和自己）
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # 计算InfoNCE损失
        # 对于每个样本，正样本的相似度应该高于负样本
        exp_sim = torch.exp(similarity_matrix)
        
        # 分子：正样本的相似度
        pos_sim = (exp_sim * mask).sum(dim=1)
        
        # 分母：所有样本的相似度（除了自己）
        all_sim = exp_sim.sum(dim=1) - exp_sim.diag()
        
        # 避免除零
        loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)
        
        # 只计算有正样本的loss
        valid_mask = mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device)
        
        return loss


class ProteinContrastiveModel(nn.Module):
    """
    完整的蛋白质对比学习模型
    """
    def __init__(self, node_feat_dim=25, hidden_dim=128, output_dim=256):
        super().__init__()
        
        self.encoder = ProteinEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
    
    def forward(self, batch_data, labels=None):
        """
        Args:
            batch_data: PyG Batch对象
            labels: [batch_size] 样本标签
        
        Returns:
            如果labels为None: 返回嵌入向量
            否则: 返回损失和嵌入向量
        """
        embeddings = self.encoder(batch_data)
        
        if labels is not None:
            loss = self.contrastive_loss(embeddings, labels)
            return loss, embeddings
        else:
            return embeddings
    
    def get_embedding(self, data):
        """获取单个蛋白质的嵌入向量"""
        self.eval()
        with torch.no_grad():
            embedding = self.encoder(data)
        return embedding


if __name__ == '__main__':
    # 测试代码
    print("测试模型...")
    
    # 创建一个简单的图
    x = torch.randn(10, 25)  # 10个节点，25维特征
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # 创建模型
    model = ProteinContrastiveModel(node_feat_dim=25, hidden_dim=128, output_dim=256)
    
    # 前向传播
    embeddings = model(data)
    print(f"输出嵌入维度: {embeddings.shape}")
    
    # 测试损失
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    batch_data = Batch.from_data_list([data] * 10)
    loss, emb = model(batch_data, labels)
    print(f"对比学习损失: {loss.item():.4f}")
    
    print("模型测试通过！")
