# On-target perturbation strength in singles versus pairs.
# For every perturbed gene that is measured, compares its expression (linear normalized scale) in control cells,
# in its single-perturbation cells and in the pair cells that contain it, and reports the ratio of the on-target
# effect in pairs relative to singles. Pairs where either target is perturbed much more weakly or strongly
# than in its single are flagged. Writes target_check_{ds}.tsv (per pair) and prints summaries.
import os, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from cells import load_cells
from scipy.stats import spearmanr, wilcoxon
for ds in ["norman", "wessels"]:
    X, idx, pb, genes = load_cells(ds)
    gi = {g: i for i, g in enumerate(genes)}
    names = list(map(str, pb["combo_names"])); pg = list(map(str, pb["pert_gene_list"])); pairs = pb["combo_pairs"]
    ctrl = pb["ctrl_mean"]; alpha = np.nan_to_num(pb["alpha"]); truth = pb["combo_truth"]
    delta = truth - (ctrl[None] + alpha[pairs[:, 0]] + alpha[pairs[:, 1]])
    share = np.linalg.norm(delta, axis=1) / np.linalg.norm(truth - ctrl[None], axis=1)
    def lin_mean(ii, g):
        v = X[ii, gi[g]].toarray().ravel(); return float(np.expm1(v).mean())
    c0 = {g: lin_mean(idx["ctrl"], g) for g in pg if g in gi}
    rows = []
    for m, n in enumerate(names):
        rec = [n]
        effs = []
        for pos in (0, 1):
            g = pg[pairs[m, pos]]
            if g not in gi or c0[g] <= 0:
                rec += [np.nan, np.nan, np.nan]; continue
            fs = lin_mean(idx["S:" + g], g) / c0[g]; fp = lin_mean(idx["P:" + n], g) / c0[g]
            # on-target effect as log2 fold change vs control; ratio of pair to single effect
            ls, lp = np.log2(max(fs, 1e-3)), np.log2(max(fp, 1e-3))
            r = lp / ls if abs(ls) > 0.2 else np.nan
            rec += [ls, lp, r]; effs.append(r)
        rec += [share[m]]
        rows.append(rec)
    A = np.array([r[1:] for r in rows], float)
    with open(f"target_check_{ds}.tsv", "w") as fh:
        fh.write("pair\tlog2fc_single_g1\tlog2fc_pair_g1\tratio_g1\tlog2fc_single_g2\tlog2fc_pair_g2\tratio_g2\tinteraction_share\n")
        for r in rows: fh.write("\t".join([r[0]] + [f"{x:.4f}" for x in r[1:]]) + "\n")
    ratios = np.concatenate([A[:, 2], A[:, 5]]); ratios = ratios[np.isfinite(ratios)]
    ls_all = np.concatenate([A[:, 0], A[:, 3]]); lp_all = np.concatenate([A[:, 1], A[:, 4]]); ok = np.isfinite(ls_all) & np.isfinite(lp_all) & (np.abs(ls_all) > 0.2)
    print(f"\n{ds}: targets measured in {np.isfinite(A[:,0]).sum()}+{np.isfinite(A[:,3]).sum()} pair positions; usable ratios n={len(ratios)}")
    print(f"{ds}: on-target log2FC single median={np.median(ls_all[ok]):.2f}, in pair median={np.median(lp_all[ok]):.2f}, Wilcoxon p={wilcoxon(ls_all[ok], lp_all[ok]).pvalue:.2e}")
    print(f"{ds}: ratio pair/single median={np.median(ratios):.2f} IQR=({np.percentile(ratios,25):.2f},{np.percentile(ratios,75):.2f})")
    for pos, col in ((0, 2), (1, 5)):
        v = A[:, col]; v = v[np.isfinite(v)]
        if len(v): print(f"{ds}: position {pos+1}: median ratio {np.median(v):.2f} (n={len(v)})")
    flag = np.array([(np.isfinite(r[3]) and (r[3] < 0.5 or r[3] > 2)) or (np.isfinite(r[6]) and (r[6] < 0.5 or r[6] > 2)) for r in rows])
    minr = np.nanmin(np.abs(np.log2(np.clip(A[:, [2, 5]], 1e-3, None))), axis=1) if np.isfinite(A[:, [2, 5]]).any() else None
    print(f"{ds}: pairs with an on-target effect <0.5x or >2x of its single: {flag.sum()} / {len(rows)}")
    dev = np.nanmax(np.abs(np.log2(np.clip(A[:, [2, 5]], 1e-3, None))), axis=1)
    okd = np.isfinite(dev)
    if okd.sum() > 5:
        rho, p = spearmanr(dev[okd], A[okd, 6]); print(f"{ds}: Spearman(|log2 on-target ratio|, interaction share) rho={rho:.3f} p={p:.2e} n={okd.sum()}")
    np.save(f"target_flag_{ds}.npy", flag)
