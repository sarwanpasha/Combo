# Comprehensive scorer.
# Splits: random (custom_split*), within-group pair holdout (gd_split*; formerly called gene-disjoint),
#         held-out gene (generated here: 20 seeds; 25% of genes held out; training pairs contain no held-out gene;
#         test pairs contain one ("one_unseen") or two ("two_unseen") held-out genes).
# In-script methods: additive, ridge (lambda=10), ridge (lambda tuned by inner 5-fold CV), MLP, shared-gene mean,
#         kNN pairs (k=5), oracle (rank 16).
# External predictions (random and within-group splits): any npz family found on disk, e.g. gears_fold*.npz,
#         gearsS{seed}_*, cpa_*, attnpert_*, scgpt_*, linpairA/G_*, written in the common format.
# Metrics: pooled F_int and R2_tot, noise-corrected F_int (noise_{ds}.npz), per-pair F median/IQR, per-gene-weighted F,
#          GEARS-style own-backbone F for models that also saved singles, and the common-subset comparison.
import os, glob, re, argparse, numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import kendalltau
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--heldout_seeds", type=int, default=20); ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--out", default="scores_all.tsv"); ap.add_argument("--no_mlp", action="store_true"); args = ap.parse_args()
R_ = args.rank
if not args.no_mlp:
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
else:
    nn = type("nn", (), {"Module": object})

class CoordMap(nn.Module):
    def __init__(s, P, r, emb=32, hid=64):
        super().__init__(); s.emb = nn.Embedding(P, emb); nn.init.normal_(s.emb.weight, std=0.1)
        s.net = nn.Sequential(nn.Linear(emb, hid), nn.ReLU(), nn.Linear(hid, r))
    def forward(s, i, j): return s.net(s.emb(i) + s.emb(j))

def fit_methods(pairs, add, delta, tr, te, P, seeds=3):
    """Returns dict method -> predicted interaction for test rows."""
    _, _, Vt = np.linalg.svd(delta[tr], full_matrices=False); B = Vt[:R_]; Z = delta[tr] @ B.T
    out = {"Additive": np.zeros((len(te), delta.shape[1])), "Oracle": delta[te] @ B.T @ B}
    out["Ridge"] = Ridge(alpha=10.0).fit(add[tr], Z).predict(add[te]) @ B
    # global shrinkage of the additive prediction: delta_hat = c * (alpha_i + alpha_j), c fitted on training pairs
    c = float((delta[tr] * add[tr]).sum() / (add[tr] ** 2).sum()); out["Shrink"] = c * add[te]
    # inner CV for lambda
    lams = [0.1, 1, 10, 100, 1000]; trs = np.array(tr); rng = np.random.default_rng(0); fid = rng.permutation(len(trs)) % 5; cv = []
    for lam in lams:
        e = 0.0
        for k in range(5):
            a_, b_ = trs[fid != k], trs[fid == k]
            _, _, V2 = np.linalg.svd(delta[a_], full_matrices=False); B2 = V2[:R_]
            p = Ridge(alpha=lam).fit(add[a_], delta[a_] @ B2.T).predict(add[b_]) @ B2
            e += ((p - delta[b_]) ** 2).sum()
        cv.append(e)
    out["Ridge-CV"] = Ridge(alpha=lams[int(np.argmin(cv))]).fit(add[tr], Z).predict(add[te]) @ B
    if args.no_mlp: preds = None
    else:
     preds = []
     ti = torch.tensor(pairs[tr, 0], device=dev); tj = torch.tensor(pairs[tr, 1], device=dev); ty = torch.tensor(Z, dtype=torch.float32, device=dev)
     for sd in range(seeds):
        torch.manual_seed(sd); m = CoordMap(P, R_).to(dev); opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=1e-3)
        for _ in range(3000):
            opt.zero_grad(); ((m(ti, tj) - ty) ** 2).mean().backward(); opt.step()
        with torch.no_grad():
            preds.append(m(torch.tensor(pairs[te, 0], device=dev), torch.tensor(pairs[te, 1], device=dev)).cpu().numpy() @ B)
     out["MLP"] = np.mean(preds, 0)
    sg = []
    for m_ in te:
        nb = [t for t in tr if set(pairs[t]) & set(pairs[m_])]
        sg.append(delta[nb].mean(0) if nb else np.zeros(delta.shape[1]))
    out["SharedGene"] = np.array(sg)
    Xtr = add[tr] / (np.linalg.norm(add[tr], axis=1, keepdims=True) + 1e-9); Xte = add[te] / (np.linalg.norm(add[te], axis=1, keepdims=True) + 1e-9)
    nn_ = np.argsort(-(Xte @ Xtr.T), axis=1)[:, :5]
    out["kNN"] = np.array([delta[np.array(tr)[r]].mean(0) for r in nn_])
    return out

def scores(PI, TI, PT, TT, Nn=None, gw=None):
    E = ((PI - TI) ** 2).sum(); D = (TI ** 2).sum()
    s = dict(F=1 - E / D, R2=1 - ((PT - TT) ** 2).sum() / ((TT - TT.mean(0)) ** 2).sum())
    if Nn is not None:
        Ns = Nn.sum(); s["F_noise_corr"] = 1 - (E - Ns) / (D - Ns); s["F_max"] = 1 - Ns / D
    pp = 1 - ((PI - TI) ** 2).sum(1) / (TI ** 2).sum(1)
    s["pairF_median"] = float(np.median(pp)); s["pairF_q25"] = float(np.percentile(pp, 25)); s["pairF_q75"] = float(np.percentile(pp, 75))
    if gw is not None:  # per-gene normalization: each gene's interaction variance weighted equally
        s["F_gene_weighted"] = float(np.mean(1 - ((PI - TI) ** 2).sum(0) / np.maximum((TI ** 2).sum(0), 1e-12)))
    return s

rows = []
EXT = {}
for ds, pbf, splits in [("norman", "pseudobulk.npz", {"random": ("custom_split.npz", ""), "within_group": ("gd_split.npz", "gd_")}),
                        ("wessels", "pseudobulk_wessels.npz", {"random": ("custom_split_wessels.npz", "wessels_rand_"), "within_group": ("gd_split_wessels.npz", "wessels_gd_")})]:
    pb = np.load(pbf, allow_pickle=True); ctrl = pb["ctrl_mean"]; alpha = np.nan_to_num(pb["alpha"]); truth = pb["combo_truth"]
    pairs = pb["combo_pairs"]; names = list(map(str, pb["combo_names"])); pg = list(map(str, pb["pert_gene_list"])); P = len(pg)
    add = alpha[pairs[:, 0]] + alpha[pairs[:, 1]]; A = ctrl[None] + add; delta = truth - A
    norm = (lambda c: c.replace("+", "_")) if ds == "wessels" else (lambda c: c)
    ci = {c: k for k, c in enumerate(names)}
    Nn = np.load(f"noise_{ds}.npz")["N"] if os.path.exists(f"noise_{ds}.npz") else None
    for split, (sfile, tag) in splits.items():
        fold = np.load(sfile)["fold_of"]; K = int(np.load(sfile)["K"])
        P_ = {}; idx_all = []
        for f in range(K):
            te = np.where(fold == f)[0]
            if len(te) == 0: continue
            tr = list(np.where(fold != f)[0]); o = fit_methods(pairs, add, delta, tr, te, P)
            for k, v in o.items(): P_.setdefault(k, []).append(v)
            idx_all.append(te)
        idx_all = np.concatenate(idx_all)
        preds = {k: np.vstack(v) for k, v in P_.items()}
        # external model families
        pat = re.compile(r"^(.*)_" + tag + r"fold(\d+)\.npz$") if tag else re.compile(r"^(.*)_fold(\d+)\.npz$")
        fam = {}
        for fn in glob.glob("*.npz"):
            mm = pat.match(fn)
            if mm and "_gd_" not in mm.group(1) + "_" and "_hog" not in mm.group(1) and not mm.group(1).endswith("wessels") and "wessels" not in mm.group(1):
                fam.setdefault(mm.group(1), []).append(fn)
        for model, files in sorted(fam.items()):
            cp = np.full_like(delta, np.nan); sp_ = {}
            for fn in files:
                z = np.load(fn, allow_pickle=True); f = int(re.search(r"fold(\d+)", fn).group(1))
                cidx = z["combo_idx"] if "combo_idx" in z.files else [ci.get(norm(str(n)), -1) for n in z["combo_names"]]
                for k, v in zip(cidx, z["combo_pred"]):
                    if k >= 0: cp[k] = v
                for n, v in zip(z["single_names"], z["single_pred"]): sp_[(f, str(n))] = v
            EXT[(ds, split, model)] = (cp, sp_)
        # common subset: pairs predicted by every external model
        common = np.ones(len(idx_all), bool)
        for (d_, s_, mdl), (cp, _) in EXT.items():
            if d_ == ds and s_ == split: common &= ~np.isnan(cp[idx_all, 0])
        for subset_name, mask in [("all", np.ones(len(idx_all), bool)), ("common", common)]:
            if subset_name == "common" and mask.all(): continue
            ii = idx_all[mask]; TI = delta[ii]; TT = truth[ii]; Nsub = Nn[ii] if Nn is not None else None
            for k, v in preds.items():
                s = scores(v[mask], TI, A[ii] + v[mask], TT, Nsub, True)
                rows.append(dict(dataset=ds, split=split, subset=subset_name, method=k, n=len(ii), **s))
            for (d_, s_, mdl), (cp, sp_) in EXT.items():
                if d_ != ds or s_ != split: continue
                ok = ~np.isnan(cp[ii, 0]); jj = ii[ok]
                if ok.sum() < 2: continue
                s = scores(cp[jj] - A[jj], delta[jj], cp[jj], truth[jj], Nn[jj] if Nn is not None else None, True)
                # own backbone, if singles available
                own_p, own_t = [], []
                for k in jj:
                    f = int(fold[k]); ga = sp_.get((f, pg[pairs[k, 0]])); gb = sp_.get((f, pg[pairs[k, 1]]))
                    if ga is not None and gb is not None:
                        ah = ga + gb - ctrl; own_p.append(cp[k] - ah); own_t.append(truth[k] - ah)
                if own_p:
                    own_p, own_t = np.array(own_p), np.array(own_t)
                    s["F_own"] = 1 - ((own_p - own_t) ** 2).sum() / (own_t ** 2).sum()
                    s["Down_over_Demp"] = (own_t ** 2).sum() / (delta[jj] ** 2).sum()
                rows.append(dict(dataset=ds, split=split, subset=subset_name, method=mdl, n=int(ok.sum()), **s))
    # held-out gene split
    genes_in = np.unique(pairs.flatten())
    for seed in range(args.heldout_seeds):
        rng = np.random.default_rng(100 + seed); S = set(rng.choice(genes_in, size=int(round(0.25 * len(genes_in))), replace=False))
        nin = np.array([(pairs[m, 0] in S) + (pairs[m, 1] in S) for m in range(len(pairs))])
        tr = list(np.where(nin == 0)[0]); te = np.where(nin > 0)[0]
        o = fit_methods(pairs, add, delta, tr, te, P, seeds=1)
        for cat, cm in [("one_unseen", nin[te] == 1), ("two_unseen", nin[te] == 2), ("any_unseen", nin[te] > 0)]:
            if cm.sum() < 2: continue
            ii = te[cm]
            for k, v in o.items():
                s = scores(v[cm], delta[ii], A[ii] + v[cm], truth[ii], Nn[ii] if Nn is not None else None)
                rows.append(dict(dataset=ds, split=f"heldout_gene_{cat}", subset=f"seed{seed}", method=k, n=len(ii), **s))
        # external models trained on the same held-out-gene split (files <model>_hog_fold<seed>.npz)
        htag = "hog_" if ds == "norman" else "wessels_hog_"
        for fn in sorted(glob.glob(f"*_{htag}fold{seed}.npz")):
            if ds == "norman" and "wessels" in fn: continue
            mdl = fn[: -len(f"_{htag}fold{seed}.npz")]
            z = np.load(fn, allow_pickle=True); cp = np.full_like(delta, np.nan)
            for k, v in zip(z["combo_idx"], z["combo_pred"]): cp[k] = v
            for cat, cm in [("one_unseen", nin[te] == 1), ("two_unseen", nin[te] == 2), ("any_unseen", nin[te] > 0)]:
                ii = te[cm]; ii = ii[~np.isnan(cp[ii, 0])]
                if len(ii) < 2: continue
                s = scores(cp[ii] - A[ii], delta[ii], cp[ii], truth[ii], Nn[ii] if Nn is not None else None)
                rows.append(dict(dataset=ds, split=f"heldout_gene_{cat}", subset=f"seed{seed}", method=mdl, n=len(ii), **s))
        print(ds, "heldout seed", seed, "train", len(tr), "test", len(te), flush=True)
keys = sorted({k for r in rows for k in r}, key=lambda k: ["dataset", "split", "subset", "method", "n"].index(k) if k in ["dataset", "split", "subset", "method", "n"] else 99)
with open(args.out, "w") as fh:
    fh.write("\t".join(keys) + "\n")
    for r in rows: fh.write("\t".join(f"{r[k]:.4f}" if isinstance(r.get(k), (float, np.floating)) else str(r.get(k, "")) for k in keys) + "\n")
# summary of held-out gene results (mean +/- sd over seeds)
import collections
agg = collections.defaultdict(list)
for r in rows:
    if r["split"].startswith("heldout_gene"): agg[(r["dataset"], r["split"], r["method"])].append((r["F"], r.get("F_noise_corr", np.nan), r["n"]))
for k, v in sorted(agg.items()):
    v = np.array(v); print(k, f"F={v[:,0].mean():.3f}+/-{v[:,0].std():.3f} Fnc={np.nanmean(v[:,1]):.3f} n_mean={v[:,2].mean():.1f} seeds={len(v)}")
for r in rows:
    if not r["split"].startswith("heldout"):
        print(r["dataset"], r["split"], r["subset"], f"{r['method']:14s}", r["n"], " ".join(f"{k}={r[k]:.3f}" for k in r if k not in ("dataset", "split", "subset", "method", "n")))
# ranking sensitivity: pooled F vs median per-pair F (random split, all)
for ds in ["norman", "wessels"]:
    sub = [r for r in rows if r["dataset"] == ds and r["split"] == "random" and r["subset"] == "all" and r["method"] not in ("Oracle", "Additive")]
    if len(sub) > 2:
        t = kendalltau([r["F"] for r in sub], [r["pairF_median"] for r in sub]); print(ds, "Kendall tau pooled vs per-pair median:", round(t[0], 3))
print("wrote", args.out)
