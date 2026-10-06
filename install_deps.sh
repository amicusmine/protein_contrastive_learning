#!/bin/bash

echo "=========================================="
echo "安装依赖到 protein_new 环境"
echo "=========================================="

# 确保在正确的环境
if [[ "$CONDA_DEFAULT_ENV" != "protein_new" ]]; then
    echo "错误: 请先激活环境："
    echo "  conda activate protein_new"
    exit 1
fi

echo "当前环境: $CONDA_DEFAULT_ENV"
echo "Python路径: $(which python)"
echo ""

# 安装依赖
echo "安装依赖包..."
pip install requests biopython pandas matplotlib seaborn scikit-learn tqdm pyyaml

echo ""
echo "=========================================="
echo "验证安装..."
echo "=========================================="

python -c "import requests; print('✓ requests')"
python -c "import Bio; print('✓ biopython')"
python -c "import pandas; print('✓ pandas')"
python -c "import matplotlib; print('✓ matplotlib')"
python -c "import seaborn; print('✓ seaborn')"
python -c "import sklearn; print('✓ scikit-learn')"
python -c "import tqdm; print('✓ tqdm')"

echo ""
echo "=========================================="
echo "安装完成！现在可以运行："
echo "  python data_preparation.py"
echo "=========================================="
