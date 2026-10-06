"""
独立测试脚本 - 不使用PyTorch Geometric
"""

import sys

print("=" * 60)
print("独立测试 - 检查环境问题")
print("=" * 60)

# 测试1: 基础导入
print("\n[测试1] 基础库导入...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except Exception as e:
    print(f"✗ PyTorch导入失败: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except Exception as e:
    print(f"✗ NumPy导入失败: {e}")
    sys.exit(1)

try:
    from Bio import PDB
    print(f"✓ Biopython")
except Exception as e:
    print(f"✗ Biopython导入失败: {e}")
    sys.exit(1)

# 测试2: PyTorch Geometric
print("\n[测试2] PyTorch Geometric...")
try:
    import torch_geometric
    print(f"✓ PyTorch Geometric 已安装")
    
    # 测试基础功能
    from torch_geometric.data import Data
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    data = Data(x=x, edge_index=edge_index)
    print(f"✓ PyTorch Geometric 基础功能正常")
    
except Exception as e:
    print(f"✗ PyTorch Geometric 有问题: {e}")
    print("\n可能的解决方案:")
    print("  1. 重新安装: pip uninstall torch-geometric")
    print("  2. 使用conda: conda install pyg -c pyg")
    print("  3. 或者我可以创建一个不依赖PyG的版本")
    sys.exit(1)

# 测试3: 图神经网络
print("\n[测试3] 图神经网络层...")
try:
    from torch_geometric.nn import GATConv
    conv = GATConv(4, 8, heads=2)
    print(f"✓ GATConv 创建成功")
    
    # 测试前向传播
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    out = conv(x, edge_index)
    print(f"✓ GATConv 前向传播成功，输出形状: {out.shape}")
    
except Exception as e:
    print(f"✗ GNN层测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: Batch操作
print("\n[测试4] Batch操作...")
try:
    from torch_geometric.data import Data, Batch
    
    data_list = []
    for i in range(3):
        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))
    
    batch = Batch.from_data_list(data_list)
    print(f"✓ Batch创建成功")
    print(f"  - 总节点数: {batch.num_nodes}")
    print(f"  - 图数量: {batch.num_graphs}")
    
except Exception as e:
    print(f"✗ Batch操作失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 环境测试全部通过！")
print("=" * 60)
print("\n问题可能出在:")
print("  1. 数据加载过程")
print("  2. PDB文件解析")
print("  3. 内存不足")
print("\n建议:")
print("  - 检查是否有PDB文件: ls data/raw/")
print("  - 查看系统资源: top 或 Activity Monitor")
