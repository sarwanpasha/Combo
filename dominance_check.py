# Is the gene-level dominance hierarchy explained by the strength of the single-gene effects?
# Compares leave-one-pair-out predictions of the signed dominance from (i) the gene hierarchy and (ii) the log ratio of
# single-effect norms, and fits the hierarchy on the residual after regressing out the norm ratio.
import os, csv, numpy as np
from scipy.stats import spearmanr
os.chdir(os.path.dirname(os.path.abspath(__file__)))
pb = np.load("pseudobulk.npz", allow_pickle=True); pairs = pb["combo_pairs"]; pg = list(map(str, pb["pert_gene_list"]))
alpha = np.nan_to_num(pb["alpha"]); nrm = np.linalg.norm(alpha, axis=1)
bio = list(csv.DictReader(open("bio_norman.tsv"), delimiter="\t"))
c1 = np.array([float(r["c1"]) for r in bio]); c2 = np.array([float(r["c2"]) for r in bio])
d = np.log2(np.maximum(np.abs(c1), 1e-3) / np.maximum(np.abs(c2), 1e-3))
lr = np.log2(nrm[pairs[:, 0]] / nrm[pairs[:, 1]])
M, P = len(d), len(pg)
X = np.zeros((M, P))
for m in range(M): X[m, pairs[m, 0]] += 1; X[m, pairs[m, 1]] -= 1
print("spearman(signed dominance, log2 norm ratio)", spearmanr(d, lr))
s, *_ = np.linalg.lstsq(X, d, rcond=None); used = np.unique(pairs.flatten())
print("spearman(gene score, log norm)", spearmanr(s[used], np.log(nrm[used])))
pred_h, pred_n, pred_r, ok = np.zeros(M), np.zeros(M), np.zeros(M), np.zeros(M, bool)
for m in range(M):
    k = np.arange(M) != m; g1, g2 = pairs[m]
    if not (np.isin(g1, pairs[k]).any() and np.isin(g2, pairs[k]).any()): continue
    ok[m] = True
    sc, *_ = np.linalg.lstsq(X[k], d[k], rcond=None); pred_h[m] = sc[g1] - sc[g2]
    b = (d[k] * lr[k]).sum() / (lr[k] ** 2).sum(); pred_n[m] = b * lr[m]
    res = d[k] - b * lr[k]; sc2, *_ = np.linalg.lstsq(X[k], res, rcond=None); pred_r[m] = b * lr[m] + sc2[g1] - sc2[g2]
for nm, p in [("hierarchy", pred_h), ("norm ratio", pred_n), ("norm ratio + hierarchy", pred_r)]:
    print(nm, "n", ok.sum(), "sign acc %.3f" % (np.sign(p[ok]) == np.sign(d[ok])).mean(), "spearman %.3f" % spearmanr(p[ok], d[ok])[0])
for g in ["CEBPA", "KLF1", "UBASH3A", "BCL2L11", "FOXL2", "TGFBR2"]:
    i = pg.index(g); print(g, "score %.2f" % s[i], "norm %.2f" % nrm[i], "norm rank %d/%d" % ((nrm[used] > nrm[i]).sum() + 1, len(used)))
