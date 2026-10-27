# Cross-half evaluation. Cells of every condition are split at random into halves A and B. Models are trained and
# applied on half-A quantities (additive profile and interaction both estimated from half A) and scored against
#   (i) the half-A interaction of the test pairs, which shares the estimation noise of the single-gene profiles
#       with the model input, and
#   (ii) the half-B interaction, whose noise is independent of everything the model saw.
# The gap between (i) and (ii) measures how much of the interaction fraction comes from shared estimation noise.
# Also reports the cross-half reproducibility of the interaction itself (delta_A used as a prediction of delta_B)
# and the entry-level buffering slope with and without independent halves (regression dilution).
import os, numpy as np
from sklearn.linear_model import Ridge
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from cells import load_cells, mean_rows
REPS, R_ = 5, 16
def F(P, T): return 1 - ((P - T) ** 2).sum() / (T ** 2).sum()
for ds, sf in [("norman", "custom_split.npz"), ("wessels", "custom_split_wessels.npz")]:
    X, idx, pb, genes = load_cells(ds)
    names = list(map(str, pb["combo_names"])); pg = list(map(str, pb["pert_gene_list"])); pairs = pb["combo_pairs"]
    fold = np.load(sf)["fold_of"]; K = int(np.load(sf)["K"]); M = len(names)
    expressed = pb["ctrl_mean"] > 0.5
    rng = np.random.default_rng(1); res = {}
    def add(k, v): res.setdefault(k, []).append(v)
    for r in range(REPS):
        H = [{}, {}]
        for k, ii in idx.items():
            if k == "ctrl" or k.startswith("S:") or k[2:] in names:
                ii = rng.permutation(ii); h = len(ii) // 2
                H[0][k] = mean_rows(X, ii[:h]); H[1][k] = mean_rows(X, ii[h:2 * h])
        ad, de = [], []
        for h in (0, 1):
            c0 = H[h]["ctrl"]
            al = {g: H[h]["S:" + g] - c0 for g in pg if "S:" + g in H[h]}
            a_ = np.array([al[pg[i]] + al[pg[j]] for i, j in pairs]); y = np.array([H[h]["P:" + n] for n in names])
            ad.append(a_); de.append(y - c0[None] - a_)
        aA, aB = ad; dA, dB = de
        add("F_rep_AtoB", F(dA, dB))
        # signal part of the interaction: sum <dA, dB> is unbiased for the noise-free sum ||delta||^2
        Dt = float((dA * dB).sum()); NB = float((dB ** 2).sum()) - Dt
        add("signal_share_half", Dt / (Dt + NB)); add("signal_share_full_est", Dt / (Dt + NB / 2))
        # entry-level slope of (pair - ctrl) on (alpha_i + alpha_j), expressed genes, same half vs independent halves
        yA = dA + aA
        add("slope_same", float((yA[:, expressed] * aA[:, expressed]).sum() / (aA[:, expressed] ** 2).sum()))
        add("slope_cross", float((yA[:, expressed] * aB[:, expressed]).sum() / (aB[:, expressed] * aA[:, expressed]).sum()))
        # reproducibility of pair-level interaction descriptors between independent halves
        from scipy.stats import spearmanr
        al_h = []
        for h in (0, 1):
            c0 = H[h]["ctrl"]; al_h.append({g: H[h]["S:" + g] - c0 for g in pg if "S:" + g in H[h]})
        desc = []
        for h in (0, 1):
            mag, dom, rf, nrm = [], [], [], []
            for m_, (i, j) in enumerate(pairs):
                A2 = np.stack([al_h[h][pg[i]], al_h[h][pg[j]]], 1)[expressed]
                yv = (de[h][m_] + ad[h][m_])[expressed]
                c, *_ = np.linalg.lstsq(A2, yv, rcond=None)
                mag.append(np.hypot(*c)); dom.append(np.log10(abs(c[0]) + 1e-6) - np.log10(abs(c[1]) + 1e-6))
                Q, _ = np.linalg.qr(np.stack([al_h[h][pg[i]], al_h[h][pg[j]]], 1)); d = de[h][m_]
                rf.append(((Q.T @ d) ** 2).sum() / (d ** 2).sum()); nrm.append(np.linalg.norm(d))
            desc.append([np.array(x) for x in (mag, dom, rf, nrm)])
        for nm, k in [("magnitude", 0), ("dominance", 1), ("rescaling_frac", 2), ("delta_norm", 3)]:
            add(f"rep_spearman_{nm}", float(spearmanr(desc[0][k], desc[1][k])[0]))
        P = {m: np.zeros_like(dA) for m in ["Additive", "Shrink", "Ridge"]}
        for f in range(K):
            te = fold == f; tr = ~te
            _, _, Vt = np.linalg.svd(dA[tr], full_matrices=False); B = Vt[:R_]
            P["Ridge"][te] = Ridge(alpha=10.0).fit(aA[tr], dA[tr] @ B.T).predict(aA[te]) @ B
            c = (dA[tr] * aA[tr]).sum() / (aA[tr] ** 2).sum(); P["Shrink"][te] = c * aA[te]
        for m, p in P.items():
            add(f"{m}_vs_A", F(p, dA)); add(f"{m}_vs_B", F(p, dB))
            # noise-free interaction fraction of a half-A model: E||p - dB||^2 = ||p - delta||^2 + N_B
            add(f"{m}_true", 1 - (((p - dB) ** 2).sum() - NB) / Dt)
            # same, for the oracle-free reference "use the half-A interaction itself"
        add("dA_true", 1 - (((dA - dB) ** 2).sum() - NB) / Dt)
        print(ds, "rep", r, {k: round(v[-1], 3) for k, v in res.items()}, flush=True)
    print(f"\n#### {ds} cross-half summary (mean, sd over {REPS} splits)")
    for k, v in res.items(): print(f"{ds} {k} {np.mean(v):.3f} {np.std(v):.3f}")
    np.savez(f"crosshalf_{ds}.npz", **{k: np.array(v) for k, v in res.items()})
