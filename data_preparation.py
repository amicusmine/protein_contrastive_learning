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
    """Write one row per complex, using the ligand code deposited in that PDB."""
    from structures import COMPLEXES

    print("创建示例数据集...")
    metadata = []
    os.makedirs('data/processed', exist_ok=True)

    print("\n下载PDB文件...")
    for pdb_id, ligand, title, split in COMPLEXES:
        pdb_file = download_pdb(pdb_id)
        if pdb_file:
            metadata.append({
                'pdb_id': pdb_id,
                'category': ligand,
                'ligand': ligand,
                'name': title,
                'pdb_file': pdb_file,
                'split': split,
            })

    df = pd.DataFrame(metadata)
    df.to_csv('data/processed/metadata.csv', index=False)
    print(f"\n成功处理 {len(df)} 个蛋白质")
    return df


def pairs_within(metadata_df):
    """Pair proteins only inside one split. The same ligand is a positive pair."""
    pairs = []
    records = metadata_df.to_dict('records')
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            left, right = records[i], records[j]
            same = left['ligand'] == right['ligand']
            pairs.append({
                'pdb1': left['pdb_id'],
                'pdb2': right['pdb_id'],
                'label': int(same),
                'category': left['ligand'] if same else f"{left['ligand']}_vs_{right['ligand']}",
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
    
    # 2. 正负样本对只在同一个 split 内生成，测试蛋白不会进入训练对
    print("\n创建正负样本对...")
    train_proteins = metadata_df[metadata_df['split'] == 'train']
    test_proteins = metadata_df[metadata_df['split'] == 'test']
    train_pairs = pairs_within(train_proteins)
    test_pairs = pairs_within(test_proteins)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(train_pairs))
    n_val = max(1, int(round(0.2 * len(train_pairs))))
    val_index = set(order[:n_val].tolist())
    val_pairs = [pair for i, pair in enumerate(train_pairs) if i in val_index]
    fit_pairs = [pair for i, pair in enumerate(train_pairs) if i not in val_index]

    pairs_df = pd.DataFrame(fit_pairs + val_pairs + test_pairs)
    pairs_df.to_csv('data/processed/pairs.csv', index=False)
    pd.DataFrame(fit_pairs).to_csv('data/processed/train.csv', index=False)
    pd.DataFrame(val_pairs).to_csv('data/processed/val.csv', index=False)
    pd.DataFrame(test_pairs).to_csv('data/processed/test.csv', index=False)

    print(f"训练蛋白对: {len(fit_pairs)}")
    print(f"验证蛋白对: {len(val_pairs)}")
    print(f"测试蛋白对: {len(test_pairs)}")
    print("验证对来自训练蛋白；测试对只含测试蛋白。")
    
    print("\n" + "=" * 50)
    print("数据准备完成！")
    print("=" * 50)
    
    return metadata_df, pairs_df


if __name__ == '__main__':
    main()
