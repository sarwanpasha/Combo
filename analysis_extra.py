# Supplementary analyses for CombiPert-Bench:
#   (1) leakage-free bootstrap of the oracle ceiling and of the spectrum,
#   (2) MLP coordinate map with Gene Ontology (GO) spectral initialization,
#   (3) GO enrichment of the leading interaction components (Norman),
#   (4) per-combination interaction share and spectrum dumps for plotting.
# Usage: python analysis_extra.py [--boot 1000]
import os, json, argparse, pickle, urllib.request
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import hypergeom
from sklearn.linear_model import Ridge
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=1000); args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
RANK = 16
def intfrac(p, t): return 1 - ((p - t) ** 2).sum() / (t ** 2).sum()
def r2(p, t): return 1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum()
DS = {"norman": ("pseudobulk.npz", "custom_split.npz", "gd_split.npz"),
      "wessels": ("pseudobulk_wessels.npz", "custom_split_wessels.npz", "gd_split_wessels.npz")}
def load(ds):
    d = np.load(DS[ds][0], allow_pickle=True)
    ctrl = d["ctrl_mean"]; alpha = np.nan_to_num(d["alpha"]); pairs = d["combo_pairs"]; truth = d["combo_truth"]
    add = ctrl[None] + alpha[pairs[:, 0]] + alpha[pairs[:, 1]]
    return d, ctrl, alpha, pairs, truth, add, truth - add
out = {}
# ---------------- (1) bootstrap ----------------
for ds in DS:
    d, ctrl, alpha, pairs, truth, add, TI = load(ds)
    fold = np.load(DS[ds][1])["fold_of"]; K = int(np.load(DS[ds][1])["K"])
    rng = np.random.default_rng(0); ceil = []; cv10 = []; cv30 = []
    for b in range(args.boot):
        P_, T_ = [], []
        for f in range(K):
            te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
            trb = rng.choice(tr, size=len(tr), replace=True)
            _, _, Vt = np.linalg.svd(TI[trb], full_matrices=False); B = Vt[:RANK]
            P_.append(TI[te] @ B.T @ B); T_.append(TI[te])
        ceil.append(intfrac(np.vstack(P_), np.vstack(T_)))
        idx = rng.choice(len(TI), size=len(TI), replace=True)
        s = np.linalg.svd(TI[idx], compute_uv=False); c = np.cumsum(s ** 2) / np.sum(s ** 2)
        cv10.append(c[9]); cv30.append(c[29])
    q = lambda x: [float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out[f"boot_{ds}"] = {"ceiling": q(ceil), "cumvar10": q(cv10), "cumvar30": q(cv30)}
    print(ds, "bootstrap ceiling mean/CI", q(ceil), "cumvar10", q(cv10), "cumvar30", q(cv30), flush=True)
# ---------------- (2) GO-initialized MLP ----------------
go = pd.read_csv("data/go_essential_all/go_essential_all.csv")
class CoordMap(nn.Module):
    def __init__(s, P, r, init):
        super().__init__(); s.emb = nn.Embedding.from_pretrained(torch.tensor(init, dtype=torch.float32), freeze=False)
        s.net = nn.Sequential(nn.Linear(init.shape[1], 64), nn.ReLU(), nn.Linear(64, r))
    def forward(s, i, j): return s.net(s.emb(i) + s.emb(j))
for ds in DS:
    d, ctrl, alpha, pairs, truth, add, TI = load(ds)
    genes = list(map(str, d["pert_gene_list"])); P = len(genes); gi = {g: i for i, g in enumerate(genes)}
    A = np.zeros((P, P))
    sub = go[go.source.isin(gi) & go.target.isin(gi)]
    for s_, t_, w in sub[["source", "target", "importance"]].itertuples(index=False):
        A[gi[s_], gi[t_]] = max(A[gi[s_], gi[t_]], w); A[gi[t_], gi[s_]] = A[gi[s_], gi[t_]]
    np.fill_diagonal(A, 0); deg = A.sum(1); dinv = np.where(deg > 0, 1 / np.sqrt(np.maximum(deg, 1e-12)), 0)
    L = dinv[:, None] * A * dinv[None, :]; w_, v_ = np.linalg.eigh(L)
    dim = min(32, P); emb = v_[:, np.argsort(-w_)[:dim]] * 0.1 * np.sqrt(P)
    cov = float((deg > 0).mean())
    for split, sfile in [("random", DS[ds][1]), ("gene-disjoint", DS[ds][2])]:
        fold = np.load(sfile)["fold_of"]; K = int(np.load(sfile)["K"]); PI, TT, TIs, PT = [], [], [], []
        for f in range(K):
            te = np.where(fold == f)[0]
            if len(te) == 0: continue
            tr = np.where(fold != f)[0]
            _, _, Vt = np.linalg.svd(TI[tr], full_matrices=False); B = Vt[:RANK]; Z = TI[tr] @ B.T
            preds = []
            for sd in range(3):
                torch.manual_seed(sd); m = CoordMap(P, RANK, emb).to(dev)
                opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=1e-3)
                ti = torch.tensor(pairs[tr, 0], device=dev); tj = torch.tensor(pairs[tr, 1], device=dev)
                ty = torch.tensor(Z, dtype=torch.float32, device=dev)
                for e in range(3000):
                    opt.zero_grad(); ((m(ti, tj) - ty) ** 2).mean().backward(); opt.step()
                with torch.no_grad():
                    preds.append(m(torch.tensor(pairs[te, 0], device=dev), torch.tensor(pairs[te, 1], device=dev)).cpu().numpy() @ B)
            pi = np.mean(preds, 0); PI.append(pi); TIs.append(TI[te]); PT.append(add[te] + pi); TT.append(truth[te])
        PI, TIs, PT, TT = map(np.vstack, (PI, TIs, PT, TT))
        out[f"go_{ds}_{split}"] = {"intFrac": float(intfrac(PI, TIs)), "totExpR2": float(r2(PT, TT)), "n": len(TIs),
                                   "go_dim": dim, "go_coverage": cov, "go_edges": int((A > 0).sum() // 2)}
        print(ds, split, "MLP+GO", out[f"go_{ds}_{split}"], flush=True)
# ---------------- (3) GO enrichment (Norman) ----------------
d, ctrl, alpha, pairs, truth, add, TI = load("norman")
gnames = list(map(str, d["gene_names"])); g2go = pickle.load(open("data/gene2go_all.pkl", "rb"))
bg = [g for g in gnames if g in g2go]; term2genes = {}
for g in bg:
    for t in g2go[g]: term2genes.setdefault(t, set()).add(g)
terms = {t: s for t, s in term2genes.items() if 5 <= len(s) <= 500}
_, _, Vt = np.linalg.svd(TI, full_matrices=False); N = len(bg); enr = []
for k in range(5):
    top = [gnames[i] for i in np.argsort(-np.abs(Vt[k]))[:50] if gnames[i] in g2go]
    res = []
    for t, s in terms.items():
        x = len(s.intersection(top))
        if x >= 2: res.append((t, x, len(s), hypergeom.sf(x - 1, N, len(s), len(top))))
    res.sort(key=lambda r: r[3]); m_ = len(terms)
    for t, x, K_, p in res[:3]:
        enr.append({"component": k + 1, "term": t, "overlap": x, "term_size": K_, "n_top": len(top), "p": float(p),
                    "p_bonf": float(min(1, p * m_))})
names = {}
for e in enr:
    try:
        u = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{e['term'].replace(':', '%3A')}"
        r = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"Accept": "application/json"}), timeout=20).read())
        e["name"] = r["results"][0]["name"]; e["aspect"] = r["results"][0].get("aspect", "")
    except Exception as ex:
        e["name"] = ""
out["go_enrichment"] = enr
for e in enr: print("GO", e, flush=True)
# ---------------- (4) figure data ----------------
for ds in DS:
    d, ctrl, alpha, pairs, truth, add, TI = load(ds)
    share = np.linalg.norm(TI, axis=1) / np.linalg.norm(truth - ctrl[None], axis=1)
    s = np.linalg.svd(TI, compute_uv=False); c = np.cumsum(s ** 2) / np.sum(s ** 2)
    out[f"fig_{ds}"] = {"share": np.round(share, 4).tolist(), "cumvar": np.round(c[:60], 4).tolist(),
                        "n_genes_in_pairs": len(set(pairs.flatten())), "M": len(TI)}
json.dump(out, open("analysis_extra.json", "w"))
print("wrote analysis_extra.json")
