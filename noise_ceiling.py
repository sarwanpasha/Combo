# Measurement-noise ceiling for the interaction fraction.
# Each condition's cells are split at random into two halves; the interaction of every pair is computed
# separately from each half (pair, both singles and control all re-estimated), and
#   N_ij = ||delta_A - delta_B||^2 / 4
# estimates the sampling variance of the full-data interaction (sum over genes). Averaged over R splits.
# Outputs noise_{ds}.npz with per-pair N, D=||delta||^2, split-half reliability, cell counts and per-gene noise,
# and prints the noise-limited ceiling F_max = 1 - sum N / sum D.
import os, argparse, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from cells import load_cells, mean_rows
ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=20); args = ap.parse_args()
for ds in ["norman", "wessels"]:
    X, idx, pb, genes = load_cells(ds)
    names = list(map(str, pb["combo_names"])); pg = list(map(str, pb["pert_gene_list"])); pairs = pb["combo_pairs"]
    ctrl = pb["ctrl_mean"]; alpha = np.nan_to_num(pb["alpha"]); truth = pb["combo_truth"]
    delta = truth - (ctrl[None] + alpha[pairs[:, 0]] + alpha[pairs[:, 1]])
    keys = ["ctrl"] + sorted({"S:" + pg[g] for g in pairs.flatten()}) + ["P:" + n for n in names]
    # sanity: full-data means reproduce pseudobulk
    chk = mean_rows(X, idx["P:" + names[0]]) - truth[0]
    print(ds, "pseudobulk check max abs diff", float(np.abs(chk).max()), flush=True)
    M, G = delta.shape
    N = np.zeros(M); rel = np.zeros(M); gene_noise = np.zeros(G)
    rng = np.random.default_rng(0)
    for r in range(args.reps):
        hA, hB = {}, {}
        for k in keys:
            ii = rng.permutation(idx[k]); h = len(ii) // 2
            hA[k] = mean_rows(X, ii[:h]); hB[k] = mean_rows(X, ii[h:2 * h])
        for m in range(M):
            a, b = pg[pairs[m, 0]], pg[pairs[m, 1]]
            dA = hA["P:" + names[m]] - hA["S:" + a] - hA["S:" + b] + hA["ctrl"]
            dB = hB["P:" + names[m]] - hB["S:" + a] - hB["S:" + b] + hB["ctrl"]
            diff = dA - dB
            N[m] += (diff ** 2).sum() / 4; gene_noise += diff ** 2 / 4
            rel[m] += np.corrcoef(dA, dB)[0, 1]
        print(ds, "rep", r, flush=True)
    N /= args.reps; rel /= args.reps; gene_noise /= args.reps
    D = (delta ** 2).sum(1)
    ncell = np.array([len(idx["P:" + n]) for n in names])
    np.savez("noise_%s.npz" % ds, N=N, D=D, rel=rel, ncell=ncell, gene_noise=gene_noise, combo_names=np.array(names))
    print(f"\n{ds}: pairs={M} cells/pair median={np.median(ncell):.0f} min={ncell.min()}")
    print(f"{ds}: noise share of interaction variance (pooled) = {N.sum()/D.sum():.3f} -> F_max = {1-N.sum()/D.sum():.3f}")
    print(f"{ds}: per-pair noise share median={np.median(N/D):.3f} IQR=({np.percentile(N/D,25):.3f},{np.percentile(N/D,75):.3f}) max={np.max(N/D):.3f}")
    print(f"{ds}: split-half reliability of delta median={np.median(rel):.3f} min={rel.min():.3f}")
    print(f"{ds}: Spearman(noise share, cells/pair) =", float(np.corrcoef(np.argsort(np.argsort(N/D)), np.argsort(np.argsort(ncell)))[0, 1]))
