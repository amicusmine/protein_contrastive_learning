# 🚀 快速修复指南

## 当前问题

你的终端显示 `(protein_new)` 但实际还在使用base环境的Python。

## 解决方案

### 方法1：直接在conda环境中安装（推荐）

```bash
# 1. 确保激活了正确的环境
conda activate protein_new

# 2. 在环境中安装依赖
conda install -c conda-forge requests biopython pandas matplotlib seaborn scikit-learn tqdm pyyaml -y

# 3. 验证
python -c "import requests; print('OK')"

# 4. 运行数据准备
python data_preparation.py
```

### 方法2：使用pip安装

```bash
# 确保在protein_new环境
conda activate protein_new

# 安装所有依赖
pip install requests biopython pandas matplotlib seaborn scikit-learn tqdm pyyaml

# 验证
python -c "import requests; print('OK')"

# 运行
python data_preparation.py
```

### 方法3：重新创建环境（如果上面不行）

```bash
# 1. 删除旧环境
conda deactivate
conda env remove -n protein_new

# 2. 创建新环境并直接安装所有包
conda create -n protein_new python=3.9 requests biopython pandas matplotlib seaborn scikit-learn tqdm pyyaml -y

# 3. 激活环境
conda activate protein_new

# 4. 安装PyTorch
conda install pytorch cpuonly -c pytorch -y

# 5. 验证
python -c "import requests; import torch; print('All OK!')"

# 6. 运行
cd /Users/amicus4ever/protein_contrastive_learning
python data_preparation.py
```

## 检查环境是否正确

运行这些命令检查：

```bash
# 查看当前环境
conda env list

# 查看当前Python路径（应该包含protein_new）
which python

# 查看已安装的包
pip list | grep requests

# 测试导入
python -c "import requests; import Bio; import torch; print('✓ 所有包可用')"
```

## 如果还有问题

尝试这个一键命令：

```bash
conda activate protein_new && conda install -c conda-forge -y requests biopython pandas matplotlib seaborn scikit-learn tqdm pyyaml && python data_preparation.py
```

## 预期结果

成功后你应该看到：

```
==================================================
蛋白质靶点对比学习 - 数据准备
==================================================

创建示例数据集...

下载PDB文件...
Processing ATP_binding: 100%|████████| 3/3
Processing ADP_binding: 100%|████████| 2/2
...

成功处理 7 个蛋白质
创建正负样本对...
正样本对: 15
负样本对: 20
总样本对: 35
```

---

**提示**: 如果conda环境一直有问题，可以使用虚拟环境（venv）：

```bash
python3 -m venv venv_protein
source venv_protein/bin/activate
pip install torch requests biopython pandas matplotlib seaborn scikit-learn tqdm pyyaml
python data_preparation.py
```
