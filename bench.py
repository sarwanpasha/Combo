# Unified CombiPert-Bench scorer (port of finalize.py / finalize2.py / finalize_wessels.py).
# Methods: additive, ridge (alpha=10, rank-r SVD coords), ours (CoordMap, 3-seed), oracle (rank-r projection),
# GEARS (empirical backbone + own backbone) when gears_*.npz checkpoints exist.
# Usage: python bench.py [--ranks 16] [--datasets norman wessels]
import os, argparse, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import Ridge
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--ranks", type=int, nargs="+", default=[16])
ap.add_argument("--datasets", nargs="+", default=["norman", "wessels"])
ap.add_argument("--no_ours", action="store_true")
ap.add_argument("--out", default="bench_results.tsv")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

def intfrac(p, t): return 1 - ((p - t) ** 2).sum() / (t ** 2).sum()
def r2(p, t): return 1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum()

class CoordMap(nn.Module):
    def __init__(s, P, r, emb=32, hid=64):
        super().__init__(); s.emb = nn.Embedding(P, emb); nn.init.normal_(s.emb.weight, std=0.1)
        s.net = nn.Sequential(nn.Linear(emb, hid), nn.ReLU(), nn.Linear(hid, r))
    def forward(s, i, j): return s.net(s.emb(i) + s.emb(j))

def load(ds):
    if ds == "norman":
        d = np.load("pseudobulk.npz", allow_pickle=True)
        splits = {"random": ("custom_split.npz", "gears_fold"), "gene-disjoint": ("gd_split.npz", "gears_gd_fold")}
        norm = lambda c: c
    else:
        d = np.load("pseudobulk_wessels.npz", allow_pickle=True)
        splits = {"random": ("custom_split_wessels.npz", "gears_wessels_rand_fold"),
                  "gene-disjoint": ("gd_split_wessels.npz", "gears_wessels_gd_fold")}
        norm = lambda c: c.replace("+", "_")
    return d, splits, norm

rows = []
for ds in args.datasets:
    d, splits, norm = load(ds)
    ctrl = d["ctrl_mean"]; alpha = np.nan_to_num(d["alpha"]); G = ctrl.shape[0]
    genes = list(d["pert_gene_list"]); P = len(genes)
    pairs = d["combo_pairs"]; truth = d["combo_truth"]; names = list(map(str, d["combo_names"]))
    ci = {c: k for k, c in enumerate(names)}
    add_all = ctrl[None, :] + alpha[pairs[:, 0]] + alpha[pairs[:, 1]]
    true_int = truth - add_all
    share = np.linalg.norm(true_int, axis=1) / np.linalg.norm(truth - ctrl[None, :], axis=1)
    s = np.linalg.svd(true_int, compute_uv=False); cv = np.cumsum(s ** 2) / np.sum(s ** 2)
    print(f"\n##### {ds}: P={P} M={len(names)} G={G} | interaction share mean={share.mean():.3f} "
          f"median={np.median(share):.3f} min={share.min():.3f} | cumvar r1={cv[0]:.2f} r10={cv[9]:.2f} r30={cv[min(29,len(cv)-1)]:.2f}")
    for split, (sfile, gpref) in splits.items():
        sp = np.load(sfile, allow_pickle=True); fold_of = sp["fold_of"]; K = int(sp["K"])
        # GEARS predictions (if present)
        g_combo = {}; g_single = {}
        for f in range(K):
            fn = f"{gpref}{f}.npz"
            if not os.path.exists(fn): continue
            z = np.load(fn, allow_pickle=True)
            for nm, v in zip(z["combo_names"], z["combo_pred"]): g_combo[ci[norm(str(nm))]] = v
            for nm, v in zip(z["single_names"], z["single_pred"]): g_single[(f, str(nm))] = v
        for RANK in args.ranks:
            acc = {m: {"pi": [], "pt": [], "ti": [], "tt": [], "fold": []} for m in ["additive", "ridge", "ours", "oracle"]}
            gacc = {"emp_pi": [], "emp_ti": [], "pt": [], "tt": [], "own_pi": [], "own_ti": [], "fold": []}
            for f in range(K):
                test = fold_of == f
                if test.sum() < 1: continue
                train = (~test) if split == "random" else (fold_of != f)
                _, _, Vt = np.linalg.svd(true_int[train], full_matrices=False); B = Vt[:RANK]
                Ztr = true_int[train] @ B.T
                reg = Ridge(alpha=10.0).fit(alpha[pairs[train, 0]] + alpha[pairs[train, 1]], Ztr)
                tidx = np.where(test)[0]
                ridge_pi = reg.predict(alpha[pairs[tidx, 0]] + alpha[pairs[tidx, 1]]) @ B
                if args.no_ours: ours_pi = np.zeros((len(tidx), G))
                else:
                    preds = []
                    ti = torch.tensor(pairs[train, 0], device=dev); tj = torch.tensor(pairs[train, 1], device=dev)
                    ty = torch.tensor(Ztr, dtype=torch.float32, device=dev)
                    for sd in range(3):
                        torch.manual_seed(sd); m = CoordMap(P, RANK).to(dev)
                        opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=1e-3)
                        for e in range(3000):
                            opt.zero_grad(); loss = ((m(ti, tj) - ty) ** 2).mean(); loss.backward(); opt.step()
                        with torch.no_grad():
                            preds.append(m(torch.tensor(pairs[tidx, 0], device=dev), torch.tensor(pairs[tidx, 1], device=dev)).cpu().numpy() @ B)
                    ours_pi = np.mean(preds, 0)
                oracle_pi = true_int[tidx] @ B.T @ B
                for name, PI in [("additive", np.zeros((len(tidx), G))), ("ridge", ridge_pi), ("ours", ours_pi), ("oracle", oracle_pi)]:
                    acc[name]["pi"].append(PI); acc[name]["pt"].append(add_all[tidx] + PI)
                    acc[name]["ti"].append(true_int[tidx]); acc[name]["tt"].append(truth[tidx]); acc[name]["fold"] += [f] * len(tidx)
                for k in tidx:
                    if k not in g_combo: continue
                    gacc["emp_pi"].append(g_combo[k] - add_all[k]); gacc["emp_ti"].append(true_int[k])
                    gacc["pt"].append(g_combo[k]); gacc["tt"].append(truth[k]); gacc["fold"].append(f)
                    ga = g_single.get((f, genes[pairs[k, 0]])); gb = g_single.get((f, genes[pairs[k, 1]]))
                    if ga is not None and gb is not None:
                        gadd = ga + gb - ctrl
                        gacc["own_pi"].append(g_combo[k] - gadd); gacc["own_ti"].append(truth[k] - gadd)
            def perfold(pi, ti, folds):
                folds = np.array(folds); v = [intfrac(pi[folds == f], ti[folds == f]) for f in np.unique(folds) if (folds == f).sum() >= 2]
                return np.mean(v), np.std(v)
            for name in acc:
                PI = np.vstack(acc[name]["pi"]); TI = np.vstack(acc[name]["ti"])
                PT = np.vstack(acc[name]["pt"]); TT = np.vstack(acc[name]["tt"])
                mu, sd = perfold(PI, TI, acc[name]["fold"])
                rows.append((ds, split, RANK, name, len(TI), intfrac(PI, TI), r2(PT, TT), mu, sd))
            if gacc["emp_pi"]:
                PI = np.vstack(gacc["emp_pi"]); TI = np.vstack(gacc["emp_ti"])
                mu, sd = perfold(PI, TI, gacc["fold"])
                rows.append((ds, split, RANK, "GEARS(emp)", len(TI), intfrac(PI, TI), r2(np.vstack(gacc["pt"]), np.vstack(gacc["tt"])), mu, sd))
                if gacc["own_pi"]:
                    OP = np.vstack(gacc["own_pi"]); OT = np.vstack(gacc["own_ti"])
                    rows.append((ds, split, RANK, "GEARS(own)", len(OT), intfrac(OP, OT), np.nan, np.nan, np.nan))
                    print(f"   [GEARS own-backbone target norm / empirical target norm] = "
                          f"{np.linalg.norm(OT,axis=1).mean():.3f} / {np.linalg.norm(TI,axis=1).mean():.3f}")
            print(f"\n=== {ds} | {split} | rank {RANK} ===")
            print(f"{'method':12s} {'n':>4s} {'intFrac':>8s} {'totExpR2':>9s} {'perfold':>16s}")
            for r in rows:
                if r[0] == ds and r[1] == split and r[2] == RANK:
                    print(f"{r[3]:12s} {r[4]:>4d} {r[5]:>8.4f} {r[6]:>9.4f}   {r[7]:.3f}+/-{r[8]:.3f}", flush=True)
with open(args.out, "w") as fh:
    fh.write("dataset\tsplit\trank\tmethod\tn\tintFrac\ttotExpR2\tperfold_mean\tperfold_std\n")
    for r in rows: fh.write("\t".join(str(round(x, 4)) if isinstance(x, float) else str(x) for x in r) + "\n")
print("\nwrote", args.out)
