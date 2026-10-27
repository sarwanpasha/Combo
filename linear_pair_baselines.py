"""
linear_pair_baselines.py -- CPU baselines on the pseudobulk npz files (no GEARS, no GPU).

Install: numpy only (python >= 3.8).

All baselines predict the pair as  ctrl + alpha_i + alpha_j + I_hat(i,j)  (additive + predicted
interaction), where the training interactions are I_m = truth_m - (ctrl + alpha_a + alpha_b)
for training pairs m (fold_of != f).  single_pred = measured singles (ctrl + alpha).

(a) "linpair": adaptation of the Ahlmann-Eltze et al. (2025, Nat Methods) linear model
    Y ~ G W P^T + b  (solve_y_axb in run_linear_pretrained_model.R: ridge-regularised
    closed form W = (G'G + lam I)^-1 G' Y P (P'P + lam I)^-1, b = row means of Y).
    - Y   = training-pair INTERACTIONS (genes x pairs), b = mean training interaction
    - G   = gene embedding: top-K PCs (K=10) of the training pseudobulk matrix
            [ctrl, all singles, training pairs] (genes as rows), as in the paper
    - P   = pair embedding built from the two perturbations' embeddings e_i, e_j:
            [e_i + e_j, e_i * e_j]  (symmetric in i,j).  Two variants of e:
              linpairA:  e = projection of alpha_i onto the K gene PCs (always available)
              linpairG:  e = the PCA row of the perturbed gene itself (the paper's trick for
                         unseen perturbations; falls back to linpairA's e if not measured)
    ridge lambda = 0.1 (paper default), K = PCA_DIM env (default 10).
(b) "sharedmean": I_hat(i,j) = mean interaction of training pairs that contain i or j (0 if none).
(c) "knnpair": I_hat = mean interaction of the k=5 training pairs whose single-effect
    sum alpha_a + alpha_b is most cosine-similar to alpha_i + alpha_j.
Env: DATASET=norman|wessels|all, SPLIT=rand|gd|all, FOLD (optional), PCA_DIM=10, RIDGE=0.1, KNN=5
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
FOLD = os.environ.get("FOLD")
PCA_DIM = int(os.environ.get("PCA_DIM", "10"))
RIDGE = float(os.environ.get("RIDGE", "0.1"))
KNN = int(os.environ.get("KNN", "5"))


def out_name(model, ds, split, f):
    if ds == "norman":
        return f"{model}_fold{f}.npz" if split == "rand" else f"{model}_gd_fold{f}.npz"
    return f"{model}_wessels_{split}_fold{f}.npz"


def load_bench(ds, split):
    sfx = "" if ds == "norman" else "_wessels"
    pb = dict(np.load(f"pseudobulk{sfx}.npz", allow_pickle=True))
    sp = np.load(f"{'custom' if split == 'rand' else 'gd'}_split{sfx}.npz", allow_pickle=True)
    fold_of = sp["fold_of"].astype(int)
    K = int(sp["K"]) if "K" in sp.files else int(fold_of.max() + 1)
    return pb, fold_of, K


def solve_y_axb(Y, A, B, lam):
    """Ahlmann-Eltze closed form: Y (g x n) ~ A W B + center ; A (g x k), B (k' x n)."""
    center = Y.mean(1, keepdims=True)
    Yc = Y - center
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Yc @ B.T) @ \
        np.linalg.inv(B @ B.T + lam * np.eye(B.shape[0]))
    return W, center


def pair_emb(E, pairs):
    a, b = E[pairs[:, 0]], E[pairs[:, 1]]
    return np.concatenate([a + b, a * b], 1).T          # (2k x n)


def run(ds, split):
    pb, fold_of, K = load_bench(ds, split)
    ctrl = pb["ctrl_mean"].astype(np.float64); alpha = pb["alpha"].astype(np.float64)
    pairs = pb["combo_pairs"].astype(int); truth = pb["combo_truth"].astype(np.float64)
    pg = pb["pert_gene_list"].astype(str); gnames = list(pb["gene_names"].astype(str))
    gpos = {g: i for i, g in enumerate(gnames)}
    additive = ctrl[None] + alpha[pairs[:, 0]] + alpha[pairs[:, 1]]
    inter = truth - additive
    names = np.array([f"{pg[a]}+{pg[b]}" for a, b in pairs], dtype=str)
    folds = [int(FOLD)] if FOLD is not None else list(range(K))
    for f in folds:
        te = np.where(fold_of == f)[0]; tr = np.where(fold_of != f)[0]
        Itr = inter[tr]
        preds = {}
        # ---- (a) linear model on interactions
        M = np.vstack([ctrl[None], ctrl[None] + alpha, truth[tr]]).T          # genes x conditions
        Mc = M - M.mean(1, keepdims=True)
        U, S, _ = np.linalg.svd(Mc, full_matrices=False)
        Gemb = U[:, :PCA_DIM] * S[:PCA_DIM]                                    # gene embedding
        eA = alpha @ U[:, :PCA_DIM]                                            # alpha projection
        eA = eA / (np.linalg.norm(eA, axis=1, keepdims=True).mean() + 1e-12)
        eG = eA.copy()
        sc = np.linalg.norm(Gemb, axis=1).mean() + 1e-12
        for p, g in enumerate(pg):
            if g in gpos:
                eG[p] = Gemb[gpos[g]] / sc
        for tag, E in (("linpairA", eA), ("linpairG", eG)):
            W, center = solve_y_axb(Itr.T, Gemb, pair_emb(E, pairs[tr]), RIDGE)
            preds[tag] = (Gemb @ W @ pair_emb(E, pairs[te]) + center).T
        # ---- (b) shared-gene mean
        sm = np.zeros((len(te), truth.shape[1]))
        for r, m in enumerate(te):
            i, j = pairs[m]
            sel = (pairs[tr, 0] == i) | (pairs[tr, 1] == i) | (pairs[tr, 0] == j) | (pairs[tr, 1] == j)
            if sel.any():
                sm[r] = Itr[sel].mean(0)
        preds["sharedmean"] = sm
        # ---- (c) kNN in single-effect space (cosine on alpha_i + alpha_j)
        feat = alpha[pairs[:, 0]] + alpha[pairs[:, 1]]
        feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12)
        sim = feat[te] @ feat[tr].T
        nn_ = np.argsort(-sim, axis=1)[:, :min(KNN, len(tr))]
        preds["knnpair"] = Itr[nn_].mean(1)

        genes_te = sorted({int(x) for m in te for x in pairs[m]})
        s_names = np.array([pg[p] for p in genes_te], dtype=str)
        s_pred = (ctrl[None] + alpha[genes_te]).astype(np.float32)
        for tag, I_hat in preds.items():
            out = out_name(tag, ds, split, f)
            np.savez(out, combo_names=names[te], combo_pred=(additive[te] + I_hat).astype(np.float32),
                     combo_idx=te, single_names=s_names, single_pred=s_pred, fold=f)
            err = np.mean((additive[te] + I_hat - truth[te]) ** 2)
            add_err = np.mean((additive[te] - truth[te]) ** 2)
            print(f"{ds}/{split} fold {f} {tag:10s} MSE {err:.5f} (additive {add_err:.5f}) -> {out}")


if __name__ == "__main__":
    DS = os.environ.get("DATASET", "all").lower(); SPLIT = os.environ.get("SPLIT", "all").lower()
    for ds in (["norman", "wessels"] if DS == "all" else [DS]):
        for sp_ in (["rand", "gd"] if SPLIT == "all" else [SPLIT]):
            run(ds, sp_)
