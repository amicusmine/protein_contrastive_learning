# 蛋白质靶点对比学习 - 详细教程

## 📚 目录

1. [项目介绍](#项目介绍)
2. [环境配置](#环境配置)
3. [数据准备](#数据准备)
4. [模型训练](#模型训练)
5. [模型评估](#模型评估)
6. [实际应用](#实际应用)
7. [核心概念解析](#核心概念解析)
8. [常见问题](#常见问题)

---

## 项目介绍

这是一个**从零开始**的蛋白质靶点对比学习项目，帮助你理解如何：

- 将蛋白质结构转换为图数据
- 使用图神经网络（GNN）学习蛋白质表示
- 通过对比学习训练模型识别相似的结合位点
- 检索和匹配蛋白质靶点

### 核心思想

**目标**：学习一个编码器，能够将功能相似的蛋白质靶点映射到相近的嵌入空间。

```
相似的靶点（结合相同配体） → 嵌入向量距离近
不同的靶点（结合不同配体） → 嵌入向量距离远
```

---

## 环境配置

### 方法1：使用pip安装（推荐）

```bash
# 创建虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 方法2：使用conda安装

```bash
conda create -n protein_cl python=3.9
conda activate protein_cl

# 安装PyTorch（根据你的CUDA版本选择）
# CPU版本：
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# GPU版本（CUDA 11.8）：
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装PyTorch Geometric
conda install pyg -c pyg

# 安装其他依赖
pip install biopython pandas matplotlib seaborn scikit-learn umap-learn tqdm
```

### 验证安装

```bash
python -c "import torch; import torch_geometric; from Bio import PDB; print('✓ 所有依赖安装成功')"
```

---

## 数据准备

### 1. 运行数据准备脚本

```bash
python data_preparation.py
```

这个脚本会：
- 从RCSB PDB下载示例蛋白质结构
- 创建正样本对（相同配体的不同蛋白）
- 创建负样本对（不同配体的蛋白）
- 划分训练/验证/测试集

### 2. 数据集说明

示例数据集包含3类蛋白质：

| 类别 | 配体 | 示例PDB | 功能 |
|------|------|---------|------|
| ATP结合 | ATP | 1ATP, 3LZA, 2HCK | 激酶等 |
| ADP结合 | ADP | 1AKE, 4AKE | 腺苷酸激酶 |
| 血红素结合 | HEM | 1MBO, 1HDA | 肌红蛋白、血红蛋白 |

### 3. 数据格式

生成的文件结构：

```
data/
├── raw/                    # 原始PDB文件
│   ├── 1ATP.pdb
│   ├── 1AKE.pdb
│   └── ...
├── processed/
│   ├── metadata.csv       # 蛋白质元数据
│   ├── pairs.csv          # 所有样本对
│   ├── train.csv          # 训练集
│   ├── val.csv            # 验证集
│   └── test.csv           # 测试集
└── cache/                 # 预处理的图数据缓存
    ├── 1ATP.pt
    └── ...
```

---

## 模型训练

### 1. 基础训练

```bash
python train.py --epochs 50 --batch_size 8
```

### 2. 自定义参数

```bash
python train.py \
    --epochs 100 \
    --batch_size 16 \
    --hidden_dim 256 \
    --output_dim 512 \
    --lr 0.001 \
    --weight_decay 0.0001
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 8 | 批次大小 |
| `--hidden_dim` | 128 | GNN隐藏层维度 |
| `--output_dim` | 256 | 嵌入向量维度 |
| `--lr` | 0.001 | 学习率 |
| `--weight_decay` | 0.0001 | L2正则化系数 |

### 4. 训练过程

训练时会看到：

```
Epoch 1/50
Training: 100%|████████| 10/10 [00:05<00:00, loss=0.8234]
Validation: 100%|████████| 3/3 [00:01<00:00]
Train Loss: 0.8234 | Val Loss: 0.7123 | Val Acc: 0.6500
✓ 保存最佳模型 (Val Loss: 0.7123)
```

### 5. 输出文件

```
checkpoints/
├── best_model.pt              # 最佳模型
├── checkpoint_epoch_10.pt     # 定期保存的checkpoint
├── checkpoint_epoch_20.pt
└── training_history.png       # 训练曲线图
```

---

## 模型评估

### 1. 运行评估

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

### 2. 评估指标

#### 检索指标

- **Recall@K**: 在前K个结果中找到正确靶点的比例
- **MRR (Mean Reciprocal Rank)**: 正确结果排名的倒数平均值

#### 分类指标

- **Accuracy**: 分类准确率
- **AUC**: ROC曲线下面积

### 3. 可视化结果

评估会生成多个可视化图：

```
results/
├── embeddings_tsne.png        # t-SNE降维可视化
├── embeddings_umap.png        # UMAP降维可视化
├── similarity_matrix.png      # 相似度矩阵热图
├── classification_results.png # 混淆矩阵和ROC曲线
└── evaluation_metrics.csv     # 数值指标
```

#### 如何解读可视化？

**embeddings_tsne.png**:
- 相同颜色的点=相同类别的蛋白
- 如果训练成功，相同颜色的点应该聚在一起
- 不同颜色的点应该分开

**similarity_matrix.png**:
- 对角线应该是深红色（自己和自己最相似）
- 相同类别的蛋白之间应该是红色（高相似度）
- 不同类别的蛋白之间应该是蓝色（低相似度）

---

## 实际应用

### 示例1：简单检索

```bash
python demo.py
```

这会展示如何：
1. 加载训练好的模型
2. 构建蛋白质数据库
3. 查询最相似的靶点

### 示例2：在代码中使用

```python
from model import ProteinContrastiveModel, pdb_to_graph
import torch

# 1. 加载模型
model = ProteinContrastiveModel()
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. 获取蛋白质嵌入
graph = pdb_to_graph('data/raw/1ATP.pdb')
with torch.no_grad():
    embedding = model.get_embedding(graph)

# 3. 比较两个蛋白质
graph1 = pdb_to_graph('protein1.pdb')
graph2 = pdb_to_graph('protein2.pdb')

with torch.no_grad():
    emb1 = model.get_embedding(graph1)
    emb2 = model.get_embedding(graph2)
    
    # 计算相似度（余弦相似度）
    similarity = torch.sum(emb1 * emb2).item()
    print(f"相似度: {similarity:.4f}")
    
    if similarity > 0.5:
        print("这两个蛋白质的靶点可能相似！")
```

### 示例3：批量检索

```python
import numpy as np

# 假设有一个查询靶点和数据库
query_embedding = ...  # 查询蛋白的嵌入
database = {
    'protein1': emb1,
    'protein2': emb2,
    'protein3': emb3,
    # ...
}

# 计算所有相似度
similarities = []
for name, emb in database.items():
    sim = np.dot(query_embedding, emb)
    similarities.append((name, sim))

# 排序并获取top-10
similarities.sort(key=lambda x: x[1], reverse=True)
top10 = similarities[:10]

for rank, (name, sim) in enumerate(top10, 1):
    print(f"{rank}. {name}: {sim:.4f}")
```

---

## 核心概念解析

### 1. 为什么用图神经网络？

蛋白质是3D结构，残基之间有空间关系：

```
蛋白质结构 → 图表示
- 节点 = 氨基酸残基
- 边 = 空间距离 < 10Å的残基对
- 节点特征 = 氨基酸类型 + 物理化学性质
- 边特征 = 距离 + 方向向量
```

### 2. 什么是对比学习？

对比学习通过比较样本对来学习表示：

```python
# 简化版的对比学习损失
def contrastive_loss(anchor, positive, negatives):
    # 正样本应该相似
    pos_sim = similarity(anchor, positive)
    
    # 负样本应该不相似
    neg_sims = [similarity(anchor, neg) for neg in negatives]
    
    # 损失：让正样本相似度高，负样本相似度低
    loss = -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sims))))
    return loss
```

### 3. InfoNCE损失详解

InfoNCE是对比学习中最常用的损失函数：

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{N} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

其中：
- $z_i, z_j$ 是正样本对的嵌入
- $\tau$ 是温度参数（控制分布的尖锐度）
- 分母包含所有样本（正样本+负样本）

### 4. 模型架构

```
PDB文件
   ↓
图表示（节点特征 + 边）
   ↓
GAT层1（注意力机制）
   ↓
GAT层2
   ↓
GAT层3
   ↓
全局池化（mean + max）
   ↓
投影层（MLP）
   ↓
嵌入向量（256维，L2归一化）
```

---

## 常见问题

### Q1: 训练时内存不足怎么办？

```bash
# 减小batch size
python train.py --batch_size 4

# 或减小模型大小
python train.py --hidden_dim 64 --output_dim 128
```

### Q2: 下载PDB文件失败？

PDB下载依赖网络。如果失败：
1. 检查网络连接
2. 使用代理
3. 手动下载后放入`data/raw/`目录

### Q3: 如何使用自己的数据？

修改`data_preparation.py`中的`sample_data`字典：

```python
sample_data = {
    'your_category': [
        {'pdb': 'YOUR_PDB_ID', 'ligand': 'YOUR_LIGAND', 'name': 'Description'},
        # 添加更多...
    ],
    # 添加更多类别...
}
```

### Q4: 如何提升模型性能？

1. **更多数据**：增加蛋白质数量
2. **数据质量**：确保正负样本定义合理
3. **模型调优**：
   - 增加隐藏层维度
   - 增加GNN层数
   - 调整学习率
4. **Hard Negative Mining**：选择困难的负样本

### Q5: 训练很慢怎么办？

1. 使用GPU（如果有）
2. 减小batch size但增加训练轮数
3. 预处理图数据并缓存（代码已实现）
4. 使用更少的数据先验证流程

### Q6: 如何解释模型的预测？

1. 查看注意力权重（GAT的优势）
2. 可视化嵌入空间
3. 计算梯度显著性图
4. 对比相似和不相似的蛋白对

---

## 进阶话题

### 1. 使用预训练模型

可以使用ESM等预训练的蛋白质语言模型：

```python
import esm

# 加载ESM-2
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

# 获取序列嵌入
data = [("protein1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK")]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33])
    embeddings = results["representations"][33]
```

### 2. 等变神经网络

对于需要旋转不变性的任务，使用SE(3)等变网络：

```python
# 使用e3nn或EGNN
from egnn import EGNN

model = EGNN(
    in_node_nf=25,
    hidden_nf=128,
    out_node_nf=256,
    in_edge_nf=4,
    n_layers=3
)
```

### 3. 多模态融合

结合序列和结构信息：

```python
# 序列编码器 + 结构编码器
sequence_emb = sequence_encoder(sequence)
structure_emb = structure_encoder(graph)

# 融合
combined_emb = torch.cat([sequence_emb, structure_emb], dim=-1)
final_emb = fusion_layer(combined_emb)
```

---

## 参考资料

### 论文
1. GearNet: Zhang et al., 2023
2. ScanNet: Tubiana et al., 2022
3. SimCLR: Chen et al., 2020
4. GAT: Veličković et al., 2018

### 代码库
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Biopython: https://biopython.org/
- ESM: https://github.com/facebookresearch/esm

### 数据库
- RCSB PDB: https://www.rcsb.org/
- PDBbind: http://www.pdbbind.org.cn/
- UniProt: https://www.uniprot.org/

---

## 总结

这个项目展示了**靶点-靶点对比学习**的完整流程：

1. ✅ 数据准备（PDB → 图）
2. ✅ 模型定义（GNN + 对比学习）
3. ✅ 训练流程（InfoNCE损失）
4. ✅ 评估方法（检索 + 分类）
5. ✅ 实际应用（相似靶点检索）

**下一步学习建议**：
1. 运行完整流程，理解每个步骤
2. 尝试修改模型架构
3. 使用自己的数据集
4. 阅读相关论文深入理解
5. 探索等变网络和预训练模型

祝学习顺利！🎉
