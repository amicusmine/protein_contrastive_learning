#!/bin/bash

# 蛋白质靶点对比学习 - 快速开始脚本

echo "======================================"
echo "蛋白质靶点对比学习 - 快速开始"
echo "======================================"

# 1. 安装依赖
echo ""
echo "[1/4] 安装依赖..."
pip install -q torch torchvision torchaudio
pip install -q torch-geometric
pip install -q biopython numpy pandas scipy matplotlib seaborn plotly scikit-learn umap-learn tqdm requests pyyaml

# 2. 准备数据
echo ""
echo "[2/4] 准备数据..."
python data_preparation.py

# 3. 训练模型（小规模快速训练）
echo ""
echo "[3/4] 训练模型（快速模式：20 epochs）..."
python train.py --epochs 20 --batch_size 4 --hidden_dim 64 --output_dim 128

# 4. 评估模型
echo ""
echo "[4/4] 评估模型..."
python evaluate.py --checkpoint checkpoints/best_model.pt

echo ""
echo "======================================"
echo "完成！查看 results/ 目录获取结果"
echo "======================================"
