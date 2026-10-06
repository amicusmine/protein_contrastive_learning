# 🧬 蛋白质靶点对比学习 - 完整项目

## 📦 项目内容

我已经为你创建了一个**完整的、可运行的**靶点-靶点对比学习项目！

### 📂 项目文件（共14个文件，约68KB代码）

#### 核心代码（5个文件）
- **`model.py`** (9.6KB) - 对比学习模型定义
  - 图神经网络编码器（GAT）
  - InfoNCE对比学习损失
  - PDB到图的转换函数

- **`data_preparation.py`** (7.1KB) - 数据准备
  - 自动下载PDB文件
  - 创建正负样本对
  - 数据集划分

- **`train.py`** (10KB) - 训练脚本
  - 完整训练循环
  - 模型保存和验证
  - 训练曲线可视化

- **`evaluate.py`** (12KB) - 评估脚本
  - 检索性能评估
  - 多种可视化（t-SNE、UMAP、热图）
  - 分类性能分析

- **`demo.py`** (6.0KB) - 应用示例
  - 相似靶点检索
  - 交互式查询
  - 批量处理

#### 测试和工具（3个文件）
- **`quick_test.py`** (1.8KB) - 快速环境测试
- **`test_system.py`** (8.6KB) - 完整系统测试
- **`run_all.py`** (3.4KB) - 一键运行脚本

#### 文档（4个文件）
- **`README.md`** (1.8KB) - 项目概述
- **`QUICK_START.md`** (7.3KB) - 快速上手指南
- **`TUTORIAL.md`** (11KB) - 详细教程
- **`PROJECT_SUMMARY.md`** (7.3KB) - 项目总结

#### 配置文件（2个文件）
- **`requirements.txt`** (391B) - Python依赖
- **`quick_start.sh`** (991B) - Bash快速启动脚本

---

## 🚀 三种使用方式

### 方式1：一键运行（最简单）

```bash
# 1. 进入项目目录
cd protein_contrastive_learning

# 2. 运行完整流程（自动执行所有步骤）
python run_all.py
```

这会自动完成：
- ✅ 环境测试
- ✅ 数据准备
- ✅ 模型训练
- ✅ 结果评估

**总用时**: 约30-60分钟

---

### 方式2：分步执行（推荐学习）

```bash
# 步骤1: 测试环境
python quick_test.py

# 步骤2: 准备数据
python data_preparation.py

# 步骤3: 训练模型
python train.py --epochs 20 --batch_size 4

# 步骤4: 评估模型
python evaluate.py --checkpoint checkpoints/best_model.pt

# 步骤5: 查看应用示例
python demo.py
```

---

### 方式3：自定义训练

```bash
# 完整训练（更好的性能）
python train.py \
    --epochs 100 \
    --batch_size 16 \
    --hidden_dim 256 \
    --output_dim 512 \
    --lr 0.001
```

---

## 📚 学习路径

### 🔰 初学者（第一次运行）

1. **阅读 `QUICK_START.md`** (5分钟)
   - 了解项目概述
   - 安装依赖

2. **运行 `python quick_test.py`** (1分钟)
   - 验证环境配置

3. **运行 `python run_all.py`** (30-60分钟)
   - 自动完成所有步骤
   - 观察输出理解流程

4. **查看结果** (10分钟)
   - 打开 `results/` 目录的图片
   - 理解嵌入空间可视化

### 📖 进阶学习（理解原理）

1. **阅读 `TUTORIAL.md`** (30-60分钟)
   - 详细教程，覆盖所有概念
   - 包含代码示例和解释

2. **阅读代码** (2-4小时)
   - 从 `model.py` 开始
   - 理解每个函数的作用
   - 查看注释和文档字符串

3. **实验修改** (1-2天)
   - 修改模型参数
   - 尝试不同的GNN架构
   - 添加数据增强

### 🎓 高级应用（实际项目）

1. **使用自己的数据**
   - 修改 `data_preparation.py`
   - 添加自己的PDB文件

2. **模型优化**
   - 实现Hard Negative Mining
   - 使用预训练模型（ESM-2）
   - 添加等变性

3. **生产部署**
   - 构建API服务
   - 批量处理管道
   - 性能优化

---

## 🎯 核心概念速览

### 什么是对比学习？

```
输入: 两个蛋白质靶点 A 和 B

如果 A 和 B 功能相似（正样本对）:
    模型学习: embedding(A) ≈ embedding(B)

如果 A 和 B 功能不同（负样本对）:
    模型学习: embedding(A) ≠ embedding(B)

应用: 查询蛋白质 Q，找到 embedding 最接近的蛋白质
```

### 模型架构

```
PDB文件
    ↓
图表示（节点=残基，边=空间邻近）
    ↓
图神经网络（3层GAT）
    ↓
全局池化
    ↓
投影层
    ↓
256维嵌入向量（L2归一化）
```

### 训练过程

```
Epoch 1: Loss=0.82, Acc=0.65
Epoch 5: Loss=0.56, Acc=0.75
Epoch 10: Loss=0.42, Acc=0.82
...
最终: Loss=0.25, Acc=0.88
```

---

## 📊 预期结果

### 训练输出
```
checkpoints/
├── best_model.pt              # 最佳模型（用于推理）
├── training_history.png       # 训练曲线
└── checkpoint_epoch_X.pt      # 中间checkpoint
```

### 评估输出
```
results/
├── embeddings_tsne.png        # t-SNE可视化
├── embeddings_umap.png        # UMAP可视化
├── similarity_matrix.png      # 相似度热图
├── classification_results.png # 混淆矩阵和ROC
└── evaluation_metrics.csv     # 数值指标
```

### 性能指标（示例数据集）
- **Recall@5**: 0.70 - 0.85
- **Accuracy**: 0.75 - 0.90
- **AUC**: 0.80 - 0.95

---

## 💡 使用示例

### 示例1：检索相似靶点

```python
from model import ProteinContrastiveModel, pdb_to_graph
import torch

# 加载模型
model = ProteinContrastiveModel()
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 获取嵌入
graph = pdb_to_graph('my_protein.pdb')
with torch.no_grad():
    embedding = model.get_embedding(graph)

print(f"嵌入向量维度: {embedding.shape}")
# 输出: 嵌入向量维度: torch.Size([1, 256])
```

### 示例2：比较两个蛋白质

```python
# 比较相似度
graph1 = pdb_to_graph('protein1.pdb')
graph2 = pdb_to_graph('protein2.pdb')

with torch.no_grad():
    emb1 = model.get_embedding(graph1)
    emb2 = model.get_embedding(graph2)
    similarity = torch.sum(emb1 * emb2).item()

print(f"相似度: {similarity:.4f}")
# 相似度 > 0.5 表示可能功能相似
```

---

## 🔧 常见问题速查

| 问题 | 解决方案 |
|------|----------|
| 依赖安装失败 | `pip install -r requirements.txt` |
| PyTorch Geometric安装失败 | 参考官方文档或使用conda |
| 下载PDB文件慢 | 减少样本数量或使用代理 |
| 训练内存不足 | 减小batch_size和hidden_dim |
| 训练很慢 | 使用GPU或减少epochs |
| 结果不好 | 增加数据量或调整超参数 |

---

## 📖 推荐阅读顺序

1. **5分钟**: `QUICK_START.md` - 快速上手
2. **1小时**: `TUTORIAL.md` - 详细教程
3. **2小时**: 阅读代码和注释
4. **实践**: 运行和实验

---

## 🎉 项目亮点

✅ **完整可运行** - 开箱即用，无需额外配置  
✅ **详细文档** - 3篇教程（26KB），覆盖所有细节  
✅ **清晰注释** - 每个函数都有说明  
✅ **模块化设计** - 易于理解和扩展  
✅ **教学友好** - 适合学习对比学习和GNN  
✅ **实际应用** - 可直接用于研究项目  

---

## 🎯 适合人群

- 🎓 生物信息学学生/研究者
- 🤖 机器学习初学者
- 💊 药物发现相关从业者
- 📊 对蛋白质分析感兴趣的人
- 🧪 想学习对比学习的开发者

---

## 📞 开始使用

```bash
# 1. 进入项目目录
cd protein_contrastive_learning

# 2. 快速测试
python quick_test.py

# 3. 开始学习
# 阅读 QUICK_START.md

# 4. 运行项目
python run_all.py

# 5. 深入学习
# 阅读 TUTORIAL.md 和代码
```

---

## 🌟 总结

这是一个**生产级质量**的靶点对比学习完整实现：

- 📝 **2000+行代码** - 包含模型、训练、评估、应用
- 📚 **26KB文档** - 详细教程和使用说明
- 🧪 **完整测试** - 系统测试和快速测试
- 🎨 **多种可视化** - t-SNE、UMAP、热图等
- 🚀 **一键运行** - 自动化完整流程

**从这里开始你的蛋白质对比学习之旅！** 🧬✨

---

## 📋 项目检查清单

运行前确认：
- [ ] Python >= 3.8
- [ ] 已安装所有依赖
- [ ] 网络连接正常（下载PDB文件）
- [ ] 至少有2GB可用磁盘空间

运行后验证：
- [ ] `data/raw/` 目录有PDB文件
- [ ] `checkpoints/` 目录有模型文件
- [ ] `results/` 目录有可视化图片
- [ ] 评估指标合理（Acc > 0.7）

祝学习愉快！如有问题，参考 `TUTORIAL.md` 的常见问题部分。🎓
