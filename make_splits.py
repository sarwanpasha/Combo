# Random 5-fold (seed 0) and gene-disjoint (seed 7) splits for both datasets.
# Norman expected: gene-disjoint test counts [7,0,7,5,7]; Wessels random ~31-32/fold.
import os, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
K = 5
for pb, rnd, gd in [("pseudobulk.npz", "custom_split.npz", "gd_split.npz"),
                    ("pseudobulk_wessels.npz", "custom_split_wessels.npz", "gd_split_wessels.npz")]:
    if not os.path.exists(pb): print("missing", pb); continue
    pairs = np.load(pb, allow_pickle=True)["combo_pairs"]; M = len(pairs)
    rng = np.random.default_rng(0); perm = rng.permutation(M); fold_of = np.zeros(M, int)
    for i, k in enumerate(perm): fold_of[k] = i % K
    np.savez(rnd, fold_of=fold_of, K=K)
    rng2 = np.random.default_rng(7); ug = sorted(set(pairs.flatten())); rng2.shuffle(ug)
    gf = {g: i % K for i, g in enumerate(ug)}
    fgd = np.array([gf[pairs[k, 0]] if gf[pairs[k, 0]] == gf[pairs[k, 1]] else -1 for k in range(M)])
    np.savez(gd, fold_of=fgd, K=K)
    print(pb, "M=", M, "random:", [int((fold_of == f).sum()) for f in range(K)],
          "| gene-disjoint test:", [int((fgd == f).sum()) for f in range(K)], "always-train:", int((fgd == -1).sum()))
