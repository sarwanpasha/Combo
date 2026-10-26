# Norman et al. (2019) pseudobulk arrays: control mean, single-gene effects, pair profiles and GEARS split labels.
import os, numpy as np, pandas as pd
from gears import PertData
os.chdir(os.path.dirname(os.path.abspath(__file__)))
pd_ = PertData("./data"); pd_.load(data_name="norman")
pd_.prepare_split(split="simulation", seed=1)
adata = pd_.adata
cond = adata.obs["condition"].values
X = adata.X
G = X.shape[1]
gene_names = adata.var["gene_name"].values
def parts(c): return [p for p in c.split("+") if p != "ctrl"]
def is_single(c): return c.count("+") == 1 and "ctrl" in c
def is_combo(c): return c.count("+") == 1 and "ctrl" not in c
def mean_for(c):
    idx = np.where(cond == c)[0]
    return np.asarray(X[idx].mean(axis=0)).ravel(), len(idx)
conds = list(adata.obs["condition"].unique())
ctrl_mean, n_ctrl = mean_for("ctrl")
all_pert_genes = set()
for c in conds:
    if c != "ctrl": all_pert_genes.update(parts(c))
pert_gene_list = sorted(all_pert_genes)
gidx = {g: i for i, g in enumerate(pert_gene_list)}
P = len(pert_gene_list)
print(f"perturbable genes P={P}, output genes G={G}")
alpha = np.full((P, G), np.nan, dtype=np.float32)
single_n = np.zeros(P, dtype=int)
for c in conds:
    if is_single(c):
        g = parts(c)[0]
        m, n = mean_for(c)
        alpha[gidx[g]] = (m - ctrl_mean).astype(np.float32)
        single_n[gidx[g]] = n
print(f"singles observed: {(~np.isnan(alpha[:,0])).sum()} / {P}")
combo_names, combo_pairs, combo_truth, combo_n = [], [], [], []
for c in conds:
    if is_combo(c):
        gs = parts(c); m, n = mean_for(c)
        combo_names.append(c); combo_pairs.append((gidx[gs[0]], gidx[gs[1]]))
        combo_truth.append(m.astype(np.float32)); combo_n.append(n)
combo_pairs = np.array(combo_pairs); combo_truth = np.vstack(combo_truth); combo_n = np.array(combo_n)
print(f"combos: {len(combo_names)}")
set2c = pd_.set2conditions; sub = pd_.subgroup
def split_of(c):
    for s in ["train", "val", "test"]:
        if c in set2c[s]: return s
    return "none"
def tier_of(c):
    for sp in ["test_subgroup", "val_subgroup"]:
        for t, lst in sub[sp].items():
            if c in lst: return t
    return "train" if split_of(c) == "train" else "none"
combo_split = np.array([split_of(c) for c in combo_names])
combo_tier = np.array([tier_of(c) for c in combo_names])
print("combo split counts:", pd.Series(combo_split).value_counts().to_dict())
print("combo tier counts :", pd.Series(combo_tier).value_counts().to_dict())
single_split = {parts(c)[0]: split_of(c) for c in conds if is_single(c)}
single_in_train = np.array([single_split.get(g, "none") == "train" for g in pert_gene_list])
print(f"singles in train: {single_in_train.sum()} / {P}")
np.savez_compressed("pseudobulk.npz", ctrl_mean=ctrl_mean.astype(np.float32), alpha=alpha,
    single_n=single_n, single_in_train=single_in_train, pert_gene_list=np.array(pert_gene_list),
    combo_names=np.array(combo_names), combo_pairs=combo_pairs, combo_truth=combo_truth,
    combo_n=combo_n, combo_split=combo_split, combo_tier=combo_tier, gene_names=gene_names)
print("saved pseudobulk.npz")
