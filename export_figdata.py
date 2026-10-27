# Compact text export of the per-pair and per-gene quantities used in the figures (figdata.txt).
import os, csv, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
L = []
import re
def rd(x, k=2): return re.sub(r"^(-?)0\.", r"\1.", f"{float(x):.{k}f}")
for ds, pbf in [("norman", "pseudobulk.npz"), ("wessels", "pseudobulk_wessels.npz")]:
    pb = np.load(pbf, allow_pickle=True); pairs = pb["combo_pairs"]; pg = list(map(str, pb["pert_gene_list"]))
    ctrl = pb["ctrl_mean"]; alpha = np.nan_to_num(pb["alpha"]); truth = pb["combo_truth"]
    delta = truth - ctrl[None] - alpha[pairs[:, 0]] - alpha[pairs[:, 1]]
    S = np.linalg.svd(delta, compute_uv=False); cv = np.cumsum(S ** 2) / (S ** 2).sum()
    L.append(f"#spectrum {ds} " + " ".join(rd(v, 3) for v in cv[:40]))
    bio = list(csv.DictReader(open(f"bio_{ds}.tsv"), delimiter="\t"))
    nz = np.load(f"noise_{ds}.npz", allow_pickle=True)
    prog = {r["pair"]: r for r in csv.DictReader(open("bio_programs.tsv"), delimiter="\t")} if ds == "norman" else {}
    cols = ["pair", "c1", "c2", "magnitude", "dominance", "fit_r2", "single_corr", "rescaling_frac", "share", "class", "ridge_pair_F"]
    L.append(f"#pairs {ds} " + " ".join(cols + ["noise_share", "rel", "ncell", "ery_int", "ery_add", "mye_int", "mye_add"]))
    for m, r in enumerate(bio):
        v = [r["pair"]] + [rd(r[c]) for c in cols[1:9]] + [r["class"][:3], rd(r["ridge_pair_F"])]
        v += [rd(nz["N"][m] / nz["D"][m]), rd(nz["rel"][m]), str(int(nz["ncell"][m]))]
        p = prog.get(r["pair"])
        v += [rd(p["erythroid_int"]), rd(p["erythroid_add"]), rd(p["myeloid_int"]), rd(p["myeloid_add"])] if p else []
        L.append(" ".join(v))
    # per-gene dominance scores (antisymmetric additive model on log2 |c1/c2|) and leave-one-pair-out predictions
    c1 = np.array([float(r["c1"]) for r in bio]); c2 = np.array([float(r["c2"]) for r in bio])
    d = np.log2(np.maximum(np.abs(c1), 1e-3) / np.maximum(np.abs(c2), 1e-3)); P = len(pg); M = len(bio)
    X = np.zeros((M, P))
    for m in range(M): X[m, pairs[m, 0]] += 1; X[m, pairs[m, 1]] -= 1
    s, *_ = np.linalg.lstsq(X, d, rcond=None)
    cnt = np.bincount(pairs.flatten(), minlength=P)
    L.append(f"#genedom {ds} " + " ".join(f"{pg[g]}:{rd(s[g], 2)}:{cnt[g]}" for g in np.argsort(-s) if cnt[g] >= 2))
    loo = []
    for m in range(M):
        keep = np.arange(M) != m; g1, g2 = pairs[m]
        if not (np.isin(g1, pairs[keep]).any() and np.isin(g2, pairs[keep]).any()): continue
        sc, *_ = np.linalg.lstsq(X[keep], d[keep], rcond=None); loo.append(f"{rd(sc[g1] - sc[g2], 2)}:{rd(d[m], 2)}")
    L.append(f"#domloo {ds} " + " ".join(loo))
open("figdata.txt", "w").write("\n".join(L) + "\n")
print("wrote figdata.txt", sum(len(x) for x in L))
