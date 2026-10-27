# Wessels et al. (2023) CaRPool-seq pseudobulk arrays: normalization, HVG selection, single-gene effects and pair profiles.
import os, scanpy as sc, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
a = sc.read_h5ad("data_wessels/WesselsSatija2023.h5ad")
print("raw:", a.shape)
# The matrix also stores transgene, guide-array and reporter features after the endogenous genes (starting at eGFP);
# keep only endogenous genes so that guide counts cannot enter the expression profiles.
a = a[:, :list(a.var.index).index("eGFP")].copy()
assert not any(":" in g for g in a.var.index)
print("endogenous genes:", a.shape)
a.layers["counts"] = a.X.copy()
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=5000, flavor="seurat")
pert_genes_all = [g for g in a.obs["perturbation"].astype(str).unique() if "_" not in g and g != "control"]
keep = a.var["highly_variable"].values.copy()
gi = {g: i for i, g in enumerate(a.var.index)}
for g in pert_genes_all:
    if g in gi: keep[gi[g]] = True
a = a[:, keep].copy()
print("after HVG+perturbed:", a.shape)
gene_names = list(a.var.index); G = len(gene_names)
pert = a.obs["perturbation"].astype(str).values
X = a.X
if not isinstance(X, np.ndarray): X = X.toarray()
def mean_of(label):
    m = pert == label
    return X[m].mean(0) if m.sum() > 0 else None
ctrl_mean = mean_of("control"); assert ctrl_mean is not None
singles = sorted([g for g in np.unique(pert) if "_" not in g and g != "control"])
P = len(singles); sgidx = {g: i for i, g in enumerate(singles)}
alpha = np.zeros((P, G), np.float32)
for g in singles:
    mu = mean_of(g); alpha[sgidx[g]] = (mu - ctrl_mean) if mu is not None else 0.0
combo_names, combo_pairs, combo_truth = [], [], []
for label in np.unique(pert):
    if "_" not in label: continue
    g1, g2 = label.split("_", 1)
    if g1 in sgidx and g2 in sgidx:
        mu = mean_of(label)
        if mu is not None:
            combo_names.append(label); combo_pairs.append([sgidx[g1], sgidx[g2]]); combo_truth.append(mu)
combo_pairs = np.array(combo_pairs); combo_truth = np.vstack(combo_truth).astype(np.float32)
print(f"singles(P)={P} combos(M)={len(combo_names)} genes(G)={G}")
np.savez_compressed("pseudobulk_wessels.npz", ctrl_mean=ctrl_mean.astype(np.float32), alpha=alpha,
    combo_pairs=combo_pairs, combo_truth=combo_truth, combo_names=np.array(combo_names),
    pert_gene_list=np.array(singles), gene_names=np.array(gene_names))
print("saved pseudobulk_wessels.npz")
