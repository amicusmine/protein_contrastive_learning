"""
简单示例：如何使用训练好的模型进行靶点检索
"""

import torch
import numpy as np
from model import ProteinContrastiveModel, pdb_to_graph


def load_trained_model(checkpoint_path='checkpoints/best_model.pt'):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ProteinContrastiveModel(
        node_feat_dim=25,
        hidden_dim=128,
        output_dim=256
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device


def get_protein_embedding(model, pdb_file, device):
    """获取蛋白质的嵌入向量"""
    # 将PDB转换为图
    graph = pdb_to_graph(pdb_file)
    if graph is None:
        raise ValueError(f"无法处理PDB文件: {pdb_file}")
    
    graph = graph.to(device)
    
    # 获取嵌入
    with torch.no_grad():
        embedding = model.get_embedding(graph)
    
    return embedding.cpu().numpy()


def find_similar_proteins(query_embedding, database_embeddings, top_k=5):
    """
    找到最相似的蛋白质
    
    Args:
        query_embedding: 查询蛋白的嵌入向量
        database_embeddings: 数据库中所有蛋白的嵌入字典 {pdb_id: embedding}
        top_k: 返回最相似的k个蛋白
    
    Returns:
        [(pdb_id, similarity_score), ...]
    """
    similarities = []
    
    for pdb_id, db_embedding in database_embeddings.items():
        # 计算余弦相似度
        similarity = np.dot(query_embedding.squeeze(), db_embedding.squeeze())
        similarities.append((pdb_id, similarity))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def main():
    """示例使用流程"""
    print("=" * 60)
    print("蛋白质靶点检索示例")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n[1] 加载训练好的模型...")
    model, device = load_trained_model('checkpoints/best_model.pt')
    print("✓ 模型加载成功")
    
    # 2. 构建数据库（预计算所有蛋白的嵌入）
    print("\n[2] 构建蛋白质数据库...")
    import pandas as pd
    metadata = pd.read_csv('data/processed/metadata.csv')
    
    database_embeddings = {}
    for _, row in metadata.iterrows():
        pdb_id = row['pdb_id']
        pdb_file = row['pdb_file']
        
        try:
            embedding = get_protein_embedding(model, pdb_file, device)
            database_embeddings[pdb_id] = embedding
            print(f"  ✓ {pdb_id} ({row['name']})")
        except Exception as e:
            print(f"  ✗ {pdb_id}: {e}")
    
    print(f"\n✓ 数据库构建完成，共 {len(database_embeddings)} 个蛋白")
    
    # 3. 查询示例
    print("\n[3] 查询示例")
    print("-" * 60)
    
    # 选择一个查询蛋白
    query_pdb_id = metadata.iloc[0]['pdb_id']
    query_pdb_file = metadata.iloc[0]['pdb_file']
    query_name = metadata.iloc[0]['name']
    query_category = metadata.iloc[0]['category']
    
    print(f"\n查询蛋白: {query_pdb_id}")
    print(f"名称: {query_name}")
    print(f"类别: {query_category}")
    
    # 获取查询蛋白的嵌入
    query_embedding = get_protein_embedding(model, query_pdb_file, device)
    
    # 从数据库中移除查询蛋白本身
    search_database = {k: v for k, v in database_embeddings.items() if k != query_pdb_id}
    
    # 检索最相似的蛋白
    print(f"\n检索最相似的5个蛋白质靶点:")
    print("-" * 60)
    
    top_results = find_similar_proteins(query_embedding, search_database, top_k=5)
    
    for rank, (pdb_id, similarity) in enumerate(top_results, 1):
        protein_info = metadata[metadata['pdb_id'] == pdb_id].iloc[0]
        print(f"\n{rank}. {pdb_id} (相似度: {similarity:.4f})")
        print(f"   名称: {protein_info['name']}")
        print(f"   类别: {protein_info['category']}")
        print(f"   配体: {protein_info['ligand']}")
        
        # 判断是否检索正确
        if protein_info['category'] == query_category:
            print("   ✓ 正确！（同类别）")
        else:
            print("   ✗ 不同类别")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    
    # 4. 返回结果供其他使用
    return {
        'model': model,
        'device': device,
        'database': database_embeddings,
        'metadata': metadata
    }


def interactive_search(model, device, database_embeddings, metadata):
    """交互式检索"""
    print("\n" + "=" * 60)
    print("交互式蛋白质检索")
    print("=" * 60)
    
    while True:
        print("\n可用的PDB ID:")
        for i, pdb_id in enumerate(metadata['pdb_id'].tolist(), 1):
            print(f"  {i}. {pdb_id}")
        
        choice = input("\n输入PDB ID进行检索 (或输入 'q' 退出): ").strip()
        
        if choice.lower() == 'q':
            break
        
        if choice not in metadata['pdb_id'].tolist():
            print("无效的PDB ID，请重试")
            continue
        
        # 执行检索
        query_info = metadata[metadata['pdb_id'] == choice].iloc[0]
        query_embedding = get_protein_embedding(model, query_info['pdb_file'], device)
        
        search_database = {k: v for k, v in database_embeddings.items() if k != choice}
        top_results = find_similar_proteins(query_embedding, search_database, top_k=3)
        
        print(f"\n查询: {choice} ({query_info['name']})")
        print("最相似的3个靶点:")
        for rank, (pdb_id, similarity) in enumerate(top_results, 1):
            protein_info = metadata[metadata['pdb_id'] == pdb_id].iloc[0]
            print(f"  {rank}. {pdb_id} - {protein_info['name']} (相似度: {similarity:.4f})")


if __name__ == '__main__':
    # 运行示例
    results = main()
    
    # 如果需要交互式检索，取消下面的注释
    # interactive_search(
    #     results['model'],
    #     results['device'],
    #     results['database'],
    #     results['metadata']
    # )
