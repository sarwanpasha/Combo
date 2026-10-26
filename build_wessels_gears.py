# Build a GEARS PertData object for Wessels (same gene panel as pseudobulk_wessels.npz)
import os, scanpy as sc, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from gears import PertData
a = sc.read_h5ad("data_wessels/WesselsSatija2023.h5ad")
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
pb = np.load("pseudobulk_wessels.npz", allow_pickle=True)
panel = list(map(str, pb["gene_names"]))
a = a[:, [g for g in panel if g in a.var.index]].copy()
def to_cond(p):
    if p == "control": return "ctrl"
    if "_" in p: g1, g2 = p.split("_", 1); return f"{g1}+{g2}"
    return f"{p}+ctrl"
a.obs["condition"] = a.obs["perturbation"].astype(str).map(to_cond).astype("category")
a.obs["cell_type"] = "THP1"
a.var["gene_name"] = a.var.index.astype(str); a.var_names = a.var.index.astype(str)
os.makedirs("data_wessels_gears", exist_ok=True)
print("adata:", a.shape, "conditions:", a.obs["condition"].nunique())
pd_ = PertData("./data_wessels_gears")
pd_.new_data_process(dataset_name="wessels", adata=a)
print("PertData built for wessels")
