"""
简化训练脚本 - 用于调试和快速测试
"""

import os
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("简化训练脚本 - 调试版本")
print("=" * 60)

# 测试1: 导入模块
print("\n[1/6] 测试导入...")
try:
    from model import ProteinContrastiveModel, pdb_to_graph
    from torch_geometric.data import Batch
    print("✓ 模块导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试2: 加载数据
print("\n[2/6] 检查数据文件...")
try:
    import pandas as pd
    
    if not os.path.exists('data/processed/metadata.csv'):
        print("✗ 数据文件不存在，请先运行: python data_preparation.py")
        sys.exit(1)
    
    metadata = pd.read_csv('data/processed/metadata.csv')
    print(f"✓ 找到 {len(metadata)} 个蛋白质")
except Exception as e:
    print(f"✗ 数据加载失败: {e}")
    sys.exit(1)

# 测试3: 图转换
print("\n[3/6] 测试PDB到图的转换...")
try:
    pdb_file = metadata.iloc[0]['pdb_file']
    if not os.path.exists(pdb_file):
        print(f"✗ PDB文件不存在: {pdb_file}")
        sys.exit(1)
    
    graph = pdb_to_graph(pdb_file)
    if graph is None:
        print("✗ 图转换返回None")
        sys.exit(1)
    
    print(f"✓ 图转换成功")
    print(f"  - 节点数: {graph.num_nodes}")
    print(f"  - 边数: {graph.num_edges}")
except Exception as e:
    print(f"✗ 图转换失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 模型创建
print("\n[4/6] 创建模型...")
try:
    model = ProteinContrastiveModel(
        node_feat_dim=25,
        hidden_dim=64,
        output_dim=128
    )
    print("✓ 模型创建成功")
    print(f"  - 参数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 前向传播
print("\n[5/6] 测试前向传播...")
try:
    # 创建一个小batch
    graphs = []
    for i in range(min(2, len(metadata))):
        pdb_file = metadata.iloc[i]['pdb_file']
        if os.path.exists(pdb_file):
            g = pdb_to_graph(pdb_file)
            if g is not None:
                graphs.append(g)
    
    if len(graphs) == 0:
        print("✗ 没有可用的图数据")
        sys.exit(1)
    
    batch_data = Batch.from_data_list(graphs)
    labels = torch.arange(len(graphs))
    
    model.eval()
    with torch.no_grad():
        embeddings = model(batch_data)
    
    print(f"✓ 前向传播成功")
    print(f"  - 输入: {len(graphs)} 个图")
    print(f"  - 输出形状: {embeddings.shape}")
    
except Exception as e:
    print(f"✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试6: 训练一步
print("\n[6/6] 测试训练一步...")
try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 前向传播
    optimizer.zero_grad()
    loss, embeddings = model(batch_data, labels)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    print(f"✓ 训练测试成功")
    print(f"  - 损失值: {loss.item():.4f}")
    
except Exception as e:
    print(f"✗ 训练测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 总结
print("\n" + "=" * 60)
print("✓ 所有测试通过！")
print("=" * 60)
print("\n你现在可以运行完整训练:")
print("  python train.py --epochs 20 --batch_size 2")
