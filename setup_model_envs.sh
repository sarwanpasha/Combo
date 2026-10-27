#!/bin/bash
# Extra environments for external models: CPA (torch 2.0.1) and scGPT, plus the AttentionPert source.
set -uo pipefail
cd "$(dirname "$0")"
python3 -m venv venv_cpa && source venv_cpa/bin/activate && pip install -q --upgrade pip wheel && \
  pip install -q torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118 && \
  pip install -q cpa-tools==0.8.8 scanpy "pyarrow<15" && python -c "import cpa, torch; print('cpa ok', torch.cuda.is_available())"; deactivate
python3 -m venv venv_scgpt && source venv_scgpt/bin/activate && pip install -q --upgrade pip wheel && \
  pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
  pip install -q "scgpt @ git+https://github.com/bowang-lab/scGPT@cebd6fae655b9c585a4807daa3ac31bb764f06b4" gdown scanpy ipython "numpy<2" && \
  python -c "import scgpt, torch; print('scgpt ok', torch.cuda.is_available())"; deactivate
[ -d AttentionPert ] || git clone -q https://github.com/BaiDing1234/AttentionPert.git ./AttentionPert
# same forward-port patches as for GEARS 0.1.2 (recent SciPy and PyTorch Geometric)
find AttentionPert -name "*.py" -exec sed -i "s/self.adata.obs.condition == 'ctrl'\]/(self.adata.obs.condition == 'ctrl').values]/; s/from torch_geometric.data import DataLoader/from torch_geometric.loader import DataLoader/" {} +
echo MODEL_ENVS_DONE
