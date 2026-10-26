# Pair-density experiment on both datasets.
# For each random-split fold the test pairs are fixed and the model is trained on a random subset of n
# of the remaining pairs. Reports ridge, MLP (one seed) and oracle interaction fraction pooled over
# folds (mean and standard deviation over repeats) together with the training pairs per gene.
import os, argparse, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import Ridge
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=10); ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--out", default="density_subsample.tsv"); args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
def intfrac(p, t): return 1 - ((p - t) ** 2).sum() / (t ** 2).sum()
class CoordMap(nn.Module):
    def __init__(s, P, r, emb=32, hid=64):
        super().__init__(); s.emb = nn.Embedding(P, emb); nn.init.normal_(s.emb.weight, std=0.1)
        s.net = nn.Sequential(nn.Linear(emb, hid), nn.ReLU(), nn.Linear(hid, r))
    def forward(s, i, j): return s.net(s.emb(i) + s.emb(j))
def ours(P, pairs_tr, Z, pairs_te, B, seed):
    torch.manual_seed(seed); m = CoordMap(P, B.shape[0]).to(dev)
    ti = torch.tensor(pairs_tr[:, 0], device=dev); tj = torch.tensor(pairs_tr[:, 1], device=dev)
    ty = torch.tensor(Z, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=1e-3)
    for e in range(3000):
        opt.zero_grad(); ((m(ti, tj) - ty) ** 2).mean().backward(); opt.step()
    with torch.no_grad():
        return m(torch.tensor(pairs_te[:, 0], device=dev), torch.tensor(pairs_te[:, 1], device=dev)).cpu().numpy() @ B
rows = []
for ds, pb, sf in [("norman", "pseudobulk.npz", "custom_split.npz"), ("wessels", "pseudobulk_wessels.npz", "custom_split_wessels.npz")]:
    d = np.load(pb, allow_pickle=True); ctrl = d["ctrl_mean"]; alpha = np.nan_to_num(d["alpha"])
    pairs = d["combo_pairs"]; truth = d["combo_truth"]; P = len(d["pert_gene_list"])
    TI = truth - (ctrl[None] + alpha[pairs[:, 0]] + alpha[pairs[:, 1]])
    fold_of = np.load(sf)["fold_of"]; K = int(np.load(sf)["K"])
    ngenes = len(set(pairs.flatten()))
    pool_min = min(int((fold_of != f).sum()) for f in range(K))
    grid = sorted(set([n for n in [20, 30, 40, 60, 80, 100, 126] if n <= pool_min] + [pool_min]))
    for n in grid:
        for rep in range(args.reps):
            rng = np.random.default_rng(1000 * rep + n)
            acc = {m: ([], []) for m in ["ridge", "ours", "oracle"]}; ppg = []
            for f in range(K):
                te = np.where(fold_of == f)[0]; pool = np.where(fold_of != f)[0]
                tr = rng.choice(pool, size=n, replace=False)
                ppg.append(n / len(set(pairs[tr].flatten())))
                _, _, Vt = np.linalg.svd(TI[tr], full_matrices=False); B = Vt[:args.rank]
                Z = TI[tr] @ B.T
                reg = Ridge(alpha=10.0).fit(alpha[pairs[tr, 0]] + alpha[pairs[tr, 1]], Z)
                preds = {"ridge": reg.predict(alpha[pairs[te, 0]] + alpha[pairs[te, 1]]) @ B,
                         "ours": ours(P, pairs[tr], Z, pairs[te], B, rep),
                         "oracle": TI[te] @ B.T @ B}
                for m, p in preds.items(): acc[m][0].append(p); acc[m][1].append(TI[te])
            res = {m: intfrac(np.vstack(a), np.vstack(b)) for m, (a, b) in acc.items()}
            rows.append((ds, n, rep, float(np.mean(ppg)), res["ridge"], res["ours"], res["oracle"]))
        r = np.array([x[3:] for x in rows if x[0] == ds and x[1] == n])
        print(f"{ds:8s} n_train={n:4d} pairs/gene={r[:,0].mean():.2f} | ridge {r[:,1].mean():.3f}+/-{r[:,1].std():.3f} "
              f"ours {r[:,2].mean():.3f}+/-{r[:,2].std():.3f} oracle {r[:,3].mean():.3f}+/-{r[:,3].std():.3f} "
              f"gap(oracle-ridge) {(r[:,3]-r[:,1]).mean():.3f}", flush=True)
with open(args.out, "w") as fh:
    fh.write("dataset\tn_train\trep\tpairs_per_gene\tridge\tours\toracle\n")
    for x in rows: fh.write("\t".join(str(round(v, 4)) if isinstance(v, float) else str(v) for v in x) + "\n")
print("wrote", args.out)
