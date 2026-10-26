#!/bin/bash
# Downloads Norman 2019 (via GEARS) and Wessels/Satija 2023 CaRPool-seq (scPerturb, Zenodo 13350497).
set -euo pipefail
cd "$(dirname "$0")"; source activate.sh
python - << 'PY'
from gears import PertData
pd_ = PertData("./data"); pd_.load(data_name="norman")
print("Norman loaded:", pd_.adata.shape)
PY
mkdir -p data_wessels
if [ ! -s data_wessels/WesselsSatija2023.h5ad ]; then
  wget -q -O data_wessels/WesselsSatija2023.h5ad "https://zenodo.org/records/13350497/files/WesselsSatija2023.h5ad?download=1"
fi
ls -la data_wessels
