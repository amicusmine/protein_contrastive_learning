#!/usr/bin/env python3
"""
一键运行脚本：自动执行完整流程
使用方法：python run_all.py
"""

import os
import sys
import subprocess
import time

def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def run_command(cmd, description):
    """运行命令并显示输出"""
    print(f"▶ {description}...")
    print(f"  命令: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"\n✓ {description} 完成")
        return True
    else:
        print(f"\n✗ {description} 失败")
        return False

def main():
    """主函数"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "蛋白质靶点对比学习 - 一键运行" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    
    start_time = time.time()
    
    # 步骤1: 快速测试
    print_section("步骤 1/4: 环境测试")
    if not run_command("python quick_test.py", "环境测试"):
        print("\n⚠️  环境测试失败，请先安装依赖：pip install -r requirements.txt")
        return 1
    
    input("\n按Enter继续数据准备...")
    
    # 步骤2: 数据准备
    print_section("步骤 2/4: 数据准备")
    print("这将下载约10个PDB文件，需要5-10分钟（取决于网络速度）")
    
    if not run_command("python data_preparation.py", "数据准备"):
        print("\n⚠️  数据准备失败，请检查网络连接")
        return 1
    
    input("\n按Enter继续模型训练...")
    
    # 步骤3: 模型训练
    print_section("步骤 3/4: 模型训练")
    print("快速训练模式：20个epoch，预计10-30分钟")
    
    if not run_command(
        "python train.py --epochs 20 --batch_size 4 --hidden_dim 64 --output_dim 128",
        "模型训练"
    ):
        print("\n⚠️  训练失败")
        return 1
    
    input("\n按Enter继续模型评估...")
    
    # 步骤4: 模型评估
    print_section("步骤 4/4: 模型评估和可视化")
    
    if not run_command(
        "python evaluate.py --checkpoint checkpoints/best_model.pt",
        "模型评估"
    ):
        print("\n⚠️  评估失败")
        return 1
    
    # 完成
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print_section("🎉 所有步骤完成！")
    print(f"总用时: {minutes}分{seconds}秒\n")
    
    print("📁 查看结果:")
    print("  - 训练曲线: checkpoints/training_history.png")
    print("  - 嵌入可视化: results/embeddings_tsne.png")
    print("  - 相似度矩阵: results/similarity_matrix.png")
    print("  - 分类结果: results/classification_results.png")
    print("  - 评估指标: results/evaluation_metrics.csv")
    
    print("\n📝 下一步:")
    print("  - 运行 'python demo.py' 查看应用示例")
    print("  - 阅读 README.md")
    print("  - 修改参数重新训练获得更好的结果")
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
