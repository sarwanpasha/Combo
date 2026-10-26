#!/bin/bash
# One-time environment setup (Ubuntu 22.04, CUDA 12 driver, Python 3.10+).
# Usage: bash setup_env.sh   (run from the repository root)
set -euo pipefail
cd "$(dirname "$0")"
WD=$(pwd)
python3 -m venv venv
cat > activate.sh << EOF
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export TMPDIR=$WD/tmp
mkdir -p $WD/tmp
source $WD/venv/bin/activate
EOF
source activate.sh
pip install -q --upgrade pip wheel
pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -q torch_geometric xxhash aiohttp psutil requests
pip install -q torch-scatter torch-sparse pyg-lib -f https://data.pyg.org/whl/torch-2.5.1+cu121.html || echo "compiled PyG companions skipped"
pip install -q cell-gears==0.1.2 scanpy anndata "numpy<2" pandas scipy scikit-learn matplotlib \
    numba llvmlite tqdm h5py natsort pyyaml networkx seaborn dcor
# Forward-port patches for GEARS 0.1.2 (recent SciPy and PyTorch Geometric)
GD=$(python -c "import gears,os;print(os.path.dirname(gears.__file__))")
sed -i "s/self.adata.obs.condition == 'ctrl'\]/(self.adata.obs.condition == 'ctrl').values]/" $GD/gears.py
sed -i "s/from torch_geometric.data import DataLoader/from torch_geometric.loader import DataLoader/" $GD/gears.py $GD/pertdata.py
grep -n "condition == 'ctrl'" $GD/gears.py | head -3
python -c "import torch,gears;from gears import PertData,GEARS;print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('gears ok')"
