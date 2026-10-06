"""
简化版测试脚本
"""

print("=" * 60)
print("蛋白质靶点对比学习 - 快速测试")
print("=" * 60)

# 测试1: 基础导入
print("\n[测试1] 检查Python包...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except:
    print("✗ PyTorch 未安装")

try:
    import torch_geometric
    print(f"✓ PyTorch Geometric")
except:
    print("✗ PyTorch Geometric 未安装")

try:
    from Bio import PDB
    print(f"✓ Biopython")
except:
    print("✗ Biopython 未安装")

try:
    import numpy as np
    import pandas as pd
    print(f"✓ NumPy & Pandas")
except:
    print("✗ NumPy/Pandas 未安装")

# 测试2: 模型基础
print("\n[测试2] 测试模型基础组件...")
try:
    import torch
    import torch.nn as nn
    
    # 简单的神经网络
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    x = torch.randn(2, 10)
    output = model(x)
    
    assert output.shape == (2, 5), "输出形状错误"
    print("✓ PyTorch模型基础正常")
except Exception as e:
    print(f"✗ 模型测试失败: {e}")

# 测试3: 图数据
print("\n[测试3] 测试图数据结构...")
try:
    from torch_geometric.data import Data
    
    x = torch.randn(5, 3)  # 5个节点，3维特征
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    
    print(f"✓ 图数据结构正常")
    print(f"  - 节点数: {data.num_nodes}")
    print(f"  - 边数: {data.num_edges}")
except Exception as e:
    print(f"✗ 图数据测试失败: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
print("\n如果所有测试通过，可以开始使用项目。")
print("如果有失败，请安装缺失的包：")
print("  pip install torch torch-geometric biopython numpy pandas")
