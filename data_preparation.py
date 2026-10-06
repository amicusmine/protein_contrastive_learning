"""
数据准备脚本：下载和处理蛋白质结合位点数据
"""

import os
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
from Bio.PDB import PDBParser, PDBIO, Select
import warnings
warnings.filterwarnings('ignore')


class BindingSiteExtractor(Select):
    """提取结合位点周围的残基"""
    def __init__(self, center, radius=10.0):
        self.center = np.array(center)
        self.radius = radius
    
    def accept_residue(self, residue):
        """只接受距离中心radius范围内的残基"""
        try:
            ca_atom = residue['CA']
            distance = np.linalg.norm(ca_atom.coord - self.center)
            return distance <= self.radius
        except:
            return False


def download_pdb(pdb_id, save_dir='data/raw'):
    """从RCSB PDB下载蛋白质结构"""
    os.makedirs(save_dir, exist_ok=True)
    pdb_file = os.path.join(save_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(pdb_file):
        return pdb_file
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(pdb_file, 'w') as f:
                f.write(response.text)
            return pdb_file
        else:
            print(f"Failed to download {pdb_id}")
            return None
    except Exception as e:
        print(f"Error downloading {pdb_id}: {e}")
        return None


def extract_binding_site(pdb_file, ligand_name, output_file):
    """提取结合位点区域"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # 找到配体的中心坐标
    ligand_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == ligand_name:
                    for atom in residue:
                        ligand_coords.append(atom.coord)
    
    if not ligand_coords:
        return None
    
    center = np.mean(ligand_coords, axis=0)
    
    # 提取10Å范围内的残基
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file, BindingSiteExtractor(center, radius=10.0))
    
    return output_file


def create_sample_dataset():
    """创建一个小型示例数据集"""
    print("创建示例数据集...")
    
    # 使用一些经典的ATP结合蛋白（正样本对）
    # 和其他不同功能的蛋白（负样本）
    sample_data = {
        'ATP_binding': [
            {'pdb': '1ATP', 'ligand': 'ATP', 'name': 'Protein Kinase'},
            {'pdb': '3LZA', 'ligand': 'ATP', 'name': 'HSP90'},
            {'pdb': '2HCK', 'ligand': 'ATP', 'name': 'Tyrosine Kinase'},
        ],
        'ADP_binding': [
            {'pdb': '1AKE', 'ligand': 'ADP', 'name': 'Adenylate Kinase'},
            {'pdb': '4AKE', 'ligand': 'ADP', 'name': 'Adenylate Kinase variant'},
        ],
        'Heme_binding': [
            {'pdb': '1MBO', 'ligand': 'HEM', 'name': 'Myoglobin'},
            {'pdb': '1HDA', 'ligand': 'HEM', 'name': 'Hemoglobin'},
        ],
        'other': [
            {'pdb': '1ALC', 'ligand': 'BEN', 'name': 'Alcohol Dehydrogenase'},
            {'pdb': '1TRZ', 'ligand': 'NAD', 'name': 'Transaldolase'},
        ]
    }
    
    # 保存元数据
    metadata = []
    os.makedirs('data/processed', exist_ok=True)
    
    print("\n下载PDB文件...")
    for category, proteins in sample_data.items():
        for protein in tqdm(proteins, desc=f"Processing {category}"):
            pdb_id = protein['pdb']
            pdb_file = download_pdb(pdb_id)
            
            if pdb_file:
                metadata.append({
                    'pdb_id': pdb_id,
                    'category': category,
                    'ligand': protein['ligand'],
                    'name': protein['name'],
                    'pdb_file': pdb_file
                })
    
    df = pd.DataFrame(metadata)
    df.to_csv('data/processed/metadata.csv', index=False)
    print(f"\n成功处理 {len(df)} 个蛋白质")
    
    return df


def create_positive_pairs(metadata_df):
    """创建正样本对（相同配体类别）"""
    pairs = []
    
    for category in metadata_df['category'].unique():
        category_proteins = metadata_df[metadata_df['category'] == category]['pdb_id'].tolist()
        
        # 同类别内两两配对
        for i in range(len(category_proteins)):
            for j in range(i + 1, len(category_proteins)):
                pairs.append({
                    'pdb1': category_proteins[i],
                    'pdb2': category_proteins[j],
                    'label': 1,  # 正样本
                    'category': category
                })
    
    return pairs


def create_negative_pairs(metadata_df, num_negatives=20):
    """创建负样本对（不同配体类别）"""
    pairs = []
    categories = metadata_df['category'].unique()
    
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i+1:]:
            proteins_cat1 = metadata_df[metadata_df['category'] == cat1]['pdb_id'].tolist()
            proteins_cat2 = metadata_df[metadata_df['category'] == cat2]['pdb_id'].tolist()
            
            # 随机采样负样本对
            for _ in range(min(num_negatives, len(proteins_cat1) * len(proteins_cat2))):
                p1 = np.random.choice(proteins_cat1)
                p2 = np.random.choice(proteins_cat2)
                pairs.append({
                    'pdb1': p1,
                    'pdb2': p2,
                    'label': 0,  # 负样本
                    'category': f"{cat1}_vs_{cat2}"
                })
    
    return pairs


def main():
    """主函数"""
    print("=" * 50)
    print("蛋白质靶点对比学习 - 数据准备")
    print("=" * 50)
    
    # 创建目录
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # 1. 创建示例数据集
    metadata_df = create_sample_dataset()
    
    # 2. 创建正负样本对
    print("\n创建正负样本对...")
    positive_pairs = create_positive_pairs(metadata_df)
    negative_pairs = create_negative_pairs(metadata_df)
    
    all_pairs = positive_pairs + negative_pairs
    pairs_df = pd.DataFrame(all_pairs)
    pairs_df.to_csv('data/processed/pairs.csv', index=False)
    
    print(f"正样本对: {len(positive_pairs)}")
    print(f"负样本对: {len(negative_pairs)}")
    print(f"总样本对: {len(all_pairs)}")
    
    # 3. 划分训练/验证/测试集
    from sklearn.model_selection import train_test_split
    
    train_val, test = train_test_split(pairs_df, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    
    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    
    print(f"\n训练集: {len(train)} 对")
    print(f"验证集: {len(val)} 对")
    print(f"测试集: {len(test)} 对")
    
    print("\n" + "=" * 50)
    print("数据准备完成！")
    print("=" * 50)
    
    return metadata_df, pairs_df


if __name__ == '__main__':
    main()
