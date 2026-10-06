"""
测试脚本：验证所有组件是否正常工作
"""

import sys
import warnings
warnings.filterwarnings('ignore')


def test_imports():
    """测试所有依赖是否安装"""
    print("=" * 60)
    print("测试1: 检查依赖安装")
    print("=" * 60)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('Bio', 'Biopython'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
    ]
    
    all_installed = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - 未安装")
            all_installed = False
    
    if all_installed:
        print("\n✓ 所有依赖已安装")
        return True
    else:
        print("\n✗ 部分依赖未安装，请运行: pip install -r requirements.txt")
        return False


def test_model():
    """测试模型定义"""
    print("\n" + "=" * 60)
    print("测试2: 模型定义")
    print("=" * 60)
    
    try:
        import torch
        from torch_geometric.data import Data, Batch
        from model import ProteinContrastiveModel, pdb_to_graph
        
        # 创建测试数据
        x = torch.randn(10, 25)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        batch = torch.zeros(10, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, batch=batch)
        
        # 创建模型
        model = ProteinContrastiveModel(node_feat_dim=25, hidden_dim=64, output_dim=128)
        
        # 测试前向传播
        embeddings = model(data)
        assert embeddings.shape == (1, 128), f"输出形状错误: {embeddings.shape}"
        
        # 测试对比学习
        batch_data = Batch.from_data_list([data] * 4)
        labels = torch.tensor([0, 0, 1, 1])
        loss, emb = model(batch_data, labels)
        assert loss.item() >= 0, f"损失值异常: {loss.item()}"
        
        print("✓ 模型定义正确")
        print(f"  - 输出嵌入维度: {embeddings.shape}")
        print(f"  - 测试损失: {loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_conversion():
    """测试PDB到图的转换"""
    print("\n" + "=" * 60)
    print("测试3: PDB到图的转换")
    print("=" * 60)
    
    try:
        import tempfile
        import os
        from model import pdb_to_graph
        
        # 创建一个简单的测试PDB文件
        test_pdb_content = """
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 20.00           C
ATOM      3  C   ALA A   1      11.500  11.000  10.000  1.00 20.00           C
ATOM      4  O   ALA A   1      11.500  11.500  11.000  1.00 20.00           O
ATOM      5  N   GLY A   2      12.000  11.500  11.000  1.00 20.00           N
ATOM      6  CA  GLY A   2      13.000  11.500  11.000  1.00 20.00           C
ATOM      7  C   GLY A   2      13.500  12.500  11.000  1.00 20.00           C
ATOM      8  O   GLY A   2      13.500  13.000  12.000  1.00 20.00           O
END
""".strip()
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(test_pdb_content)
            temp_pdb = f.name
        
        try:
            # 转换为图
            graph = pdb_to_graph(temp_pdb, distance_threshold=10.0)
            
            assert graph is not None, "图转换失败"
            assert graph.x.shape[0] > 0, "节点数为0"
            assert graph.edge_index.shape[1] > 0, "边数为0"
            
            print("✓ PDB到图的转换正常")
            print(f"  - 节点数: {graph.x.shape[0]}")
            print(f"  - 边数: {graph.edge_index.shape[1]}")
            print(f"  - 节点特征维度: {graph.x.shape[1]}")
            
            return True
            
        finally:
            # 清理临时文件
            os.unlink(temp_pdb)
        
    except Exception as e:
        print(f"✗ 图转换测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_preparation():
    """测试数据准备模块"""
    print("\n" + "=" * 60)
    print("测试4: 数据准备模块")
    print("=" * 60)
    
    try:
        from data_preparation import create_positive_pairs, create_negative_pairs
        import pandas as pd
        
        # 创建测试元数据
        test_metadata = pd.DataFrame({
            'pdb_id': ['1ATP', '2ATP', '3ATP', '1HEM', '2HEM'],
            'category': ['ATP', 'ATP', 'ATP', 'HEM', 'HEM'],
            'ligand': ['ATP', 'ATP', 'ATP', 'HEM', 'HEM']
        })
        
        # 测试正样本对生成
        pos_pairs = create_positive_pairs(test_metadata)
        assert len(pos_pairs) > 0, "未生成正样本对"
        
        # 测试负样本对生成
        neg_pairs = create_negative_pairs(test_metadata, num_negatives=5)
        assert len(neg_pairs) > 0, "未生成负样本对"
        
        print("✓ 数据准备模块正常")
        print(f"  - 正样本对数: {len(pos_pairs)}")
        print(f"  - 负样本对数: {len(neg_pairs)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据准备测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_components():
    """测试训练组件"""
    print("\n" + "=" * 60)
    print("测试5: 训练组件")
    print("=" * 60)
    
    try:
        import torch
        from torch_geometric.data import Data, Batch
        from model import ProteinContrastiveModel
        
        # 创建模型和优化器
        model = ProteinContrastiveModel(node_feat_dim=25, hidden_dim=64, output_dim=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 创建测试batch
        data_list = []
        for _ in range(4):
            x = torch.randn(10, 25)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index))
        
        batch_data = Batch.from_data_list(data_list)
        labels = torch.tensor([0, 0, 1, 1])
        
        # 测试训练步骤
        model.train()
        optimizer.zero_grad()
        loss, embeddings = model(batch_data, labels)
        loss.backward()
        optimizer.step()
        
        print("✓ 训练组件正常")
        print(f"  - 损失值: {loss.item():.4f}")
        print(f"  - 嵌入形状: {embeddings.shape}")
        print(f"  - 梯度计算: 正常")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "蛋白质靶点对比学习 - 系统测试" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    tests = [
        ("依赖检查", test_imports),
        ("模型定义", test_model),
        ("图转换", test_graph_conversion),
        ("数据准备", test_data_preparation),
        ("训练组件", test_training_components),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name}测试出现异常: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统准备就绪。")
        print("\n下一步:")
        print("  1. 运行 'python data_preparation.py' 准备数据")
        print("  2. 运行 'python train.py' 训练模型")
        print("  3. 运行 'python evaluate.py' 评估模型")
        print("  4. 运行 'python demo.py' 查看应用示例")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
