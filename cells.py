# Shared helpers: load cell-level expression for each screen, aligned with the pseudobulk arrays.
import numpy as np, scanpy as sc, scipy.sparse as sp

def load_cells(ds):
    """Returns (X csr log-normalized, cond index dict {'ctrl':idx, single_gene:idx, pair_name:idx}, pb npz)."""
    if ds == "norman":
        pb = np.load("pseudobulk.npz", allow_pickle=True)
        a = sc.read_h5ad("data/norman/perturb_processed.h5ad")
        X = sp.csr_matrix(a.X)
        cond = a.obs["condition"].astype(str).values
        conds = list(dict.fromkeys(cond))  # order of first appearance, as in adata.obs.condition.unique()
        idx = {}
        for c in conds:
            ii = np.where(cond == c)[0]
            if c == "ctrl":
                idx["ctrl"] = ii
            elif c.count("+") == 1 and "ctrl" in c:
                g = [p for p in c.split("+") if p != "ctrl"][0]
                idx["S:" + g] = ii  # last orientation wins, as in build_pseudobulk.py
            else:
                idx["P:" + c] = ii
        genes = list(map(str, pb["gene_names"]))
        assert list(a.var["gene_name"].astype(str)) == genes
    else:
        pb = np.load("pseudobulk_wessels.npz", allow_pickle=True)
        a = sc.read_h5ad("data_wessels/WesselsSatija2023.h5ad")
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
        genes = list(map(str, pb["gene_names"]))
        a = a[:, genes].copy()
        X = sp.csr_matrix(a.X)
        pert = a.obs["perturbation"].astype(str).values
        idx = {}
        for c in np.unique(pert):
            ii = np.where(pert == c)[0]
            if c == "control": idx["ctrl"] = ii
            elif "_" in c: idx["P:" + c] = ii
            else: idx["S:" + c] = ii
    return X, idx, pb, genes

def mean_rows(X, ii):
    return np.asarray(X[ii].mean(axis=0)).ravel()
