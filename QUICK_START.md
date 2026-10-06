# 快速上手指南

欢迎使用**蛋白质靶点对比学习**项目！这是一个完整的、可运行的靶点-靶点对比学习实现。

## 🚀 5分钟快速开始

### 第一步：安装依赖

```bash
# 确保你的Python版本 >= 3.8
python --version

# 安装核心依赖（如果已有PyTorch可跳过）
pip install torch torchvision torchaudio

# 安装PyTorch Geometric
pip install torch-geometric

# 安装其他依赖
pip install biopython numpy pandas matplotlib seaborn scikit-learn tqdm requests
```

**或者一键安装：**

```bash
pip install -r requirements.txt
```

### 第二步：验证安装

```bash
python quick_test.py
```

看到所有 `✓` 就说明安装成功！

### 第三步：运行完整示例

```bash
# 1. 准备数据（会自动下载PDB文件，需要网络连接）
python data_preparation.py

# 2. 训练模型（快速版本：20轮）
python train.py --epochs 20 --batch_size 4

# 3. 评估模型
python evaluate.py --checkpoint checkpoints/best_model.pt

# 4. 查看应用示例
python demo.py
```

---

## 📁 项目结构

```
protein_contrastive_learning/
│
├── README.md              # 项目简介
├── TUTORIAL.md            # 详细教程
├── QUICK_START.md         # 本文件
├── requirements.txt       # 依赖列表
│
├── data_preparation.py    # 数据下载和预处理
├── model.py               # 对比学习模型定义
├── train.py               # 训练脚本
├── evaluate.py            # 评估和可视化
├── demo.py                # 使用示例
│
├── quick_test.py          # 快速测试
├── test_system.py         # 完整系统测试
│
├── data/                  # 数据目录
│   ├── raw/              # 原始PDB文件
│   ├── processed/        # 处理后的数据
│   └── cache/            # 图数据缓存
│
├── checkpoints/           # 模型保存目录
└── results/              # 结果和可视化
```

---

## 🎯 核心概念

### 什么是靶点-靶点对比学习？

**目标**：学习识别功能相似的蛋白质结合位点。

**原理**：
1. **输入**：蛋白质3D结构（PDB文件）
2. **处理**：转换为图（节点=氨基酸，边=空间邻近）
3. **编码**：用图神经网络编码成向量
4. **学习**：相似靶点的向量距离近，不同靶点的向量距离远

**应用**：
- 药物靶点发现
- 蛋白质功能预测
- 虚拟筛选

---

## 💡 使用示例

### 示例1：训练自己的模型

```bash
# 使用默认参数
python train.py

# 自定义参数
python train.py \
    --epochs 100 \
    --batch_size 16 \
    --hidden_dim 256 \
    --output_dim 512 \
    --lr 0.001
```

### 示例2：检索相似靶点

```python
from model import ProteinContrastiveModel, pdb_to_graph
import torch

# 加载模型
model = ProteinContrastiveModel()
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 获取蛋白质嵌入
graph1 = pdb_to_graph('protein1.pdb')
graph2 = pdb_to_graph('protein2.pdb')

with torch.no_grad():
    emb1 = model.get_embedding(graph1)
    emb2 = model.get_embedding(graph2)
    
    # 计算相似度
    similarity = torch.sum(emb1 * emb2).item()
    print(f"相似度: {similarity:.4f}")
```

### 示例3：批量处理

```bash
# 运行demo.py查看完整的批量检索示例
python demo.py
```

---

## 🔧 常见问题

### Q: 我的电脑没有GPU，能运行吗？

**A:** 可以！代码会自动检测并使用CPU。不过训练会慢一些。

### Q: 下载PDB文件很慢怎么办？

**A:** 这取决于网络速度。如果实在太慢：
1. 使用代理
2. 减少样本数量（修改`data_preparation.py`）
3. 手动下载放入`data/raw/`目录

### Q: 安装PyTorch Geometric失败？

**A:** 参考官方文档根据你的PyTorch版本选择：
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

```bash
# 或使用conda安装
conda install pyg -c pyg
```

### Q: 训练时内存不足？

**A:** 减小batch size和模型大小：

```bash
python train.py --batch_size 2 --hidden_dim 64 --output_dim 128
```

### Q: 如何使用自己的数据？

**A:** 修改`data_preparation.py`中的数据定义：

```python
sample_data = {
    'your_category': [
        {'pdb': 'YOUR_PDB_ID', 'ligand': 'LIGAND_NAME', 'name': 'Description'},
    ],
}
```

---

## 📊 理解输出结果

### 训练输出

```
Epoch 1/50
Training: 100%|████████| 10/10 [00:05<00:00, loss=0.8234]
Train Loss: 0.8234 | Val Loss: 0.7123 | Val Acc: 0.6500
✓ 保存最佳模型
```

- **Train Loss**: 训练损失，应该逐渐下降
- **Val Loss**: 验证损失，越低越好
- **Val Acc**: 验证准确率，越高越好（>0.7就不错）

### 评估结果

```
检索性能:
  recall@1: 0.6500    # 前1个结果中找到正确靶点的比例
  recall@5: 0.8500    # 前5个结果中找到的比例
  mrr: 0.7234         # 平均倒数排名

分类性能:
  accuracy: 0.8200    # 分类准确率
  auc: 0.8900         # ROC曲线下面积
```

**好的模型**：
- Recall@5 > 0.8
- Accuracy > 0.75
- AUC > 0.85

### 可视化结果

在`results/`目录下：

1. **embeddings_tsne.png**: 嵌入空间可视化
   - 相同颜色的点应该聚在一起
   
2. **similarity_matrix.png**: 相似度热图
   - 对角线应该是红色（高相似度）
   - 同类别之间应该偏红
   
3. **classification_results.png**: 混淆矩阵和ROC曲线
   - 对角线数字大=分类准确

---

## 🎓 学习路径

### 初学者（0-2周）

1. ✅ 运行`quick_test.py`确保环境正确
2. ✅ 运行完整流程（data → train → evaluate → demo）
3. ✅ 阅读`model.py`理解模型架构
4. ✅ 查看可视化结果，理解嵌入空间

### 进阶（2-4周）

1. 修改模型参数，观察效果
2. 尝试不同的GNN架构（GCN、GAT、GraphSAGE）
3. 实现Hard Negative Mining
4. 添加数据增强

### 高级（4周+）

1. 使用预训练模型（ESM-2）
2. 实现等变神经网络
3. 多模态融合（序列+结构）
4. 在自己的数据集上训练

---

## 📚 推荐阅读

### 必读论文
1. **SimCLR** - 对比学习基础
2. **GearNet** - 蛋白质表示学习
3. **GAT** - 图注意力网络

### 推荐资源
- PyTorch Geometric教程
- Biopython文档
- RCSB PDB数据库

---

## 🤝 获取帮助

1. 查看`TUTORIAL.md`获取详细教程
2. 运行`test_system.py`进行系统诊断
3. 查看代码注释理解实现细节

---

## ✨ 下一步

现在你已经准备好了！运行：

```bash
# 快速测试
python quick_test.py

# 如果测试通过，开始完整流程
python data_preparation.py
python train.py --epochs 20
python evaluate.py
python demo.py
```

**预计时间**：
- 数据准备：5-10分钟（取决于网络）
- 训练（20 epochs）：10-30分钟（取决于CPU/GPU）
- 评估：2-5分钟

祝学习愉快！🎉

---

## 📝 项目特点

✅ **完整可运行** - 从数据到模型到应用的完整流程  
✅ **详细注释** - 每个函数都有清晰的说明  
✅ **教学友好** - 适合学习对比学习和GNN  
✅ **易于扩展** - 模块化设计，方便修改  
✅ **实际应用** - 可直接用于靶点检索任务  

**适合人群**：
- 生物信息学学生/研究者
- 机器学习初学者
- 对蛋白质-药物相互作用感兴趣的人
- 想学习对比学习的开发者
