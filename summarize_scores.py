# Compact summary of scores_all.tsv: one line per dataset, split, subset and method (random and group-holdout splits),
# and mean +/- sd over gene draws for the held-out-gene split.
import csv, sys, numpy as np
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "scores_all.tsv"), delimiter="\t"))
def g(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", "nan", None) else np.nan
for r in rows:
    if r["split"].startswith("heldout"): continue
    print(r["dataset"][:1], r["split"][:4], r["subset"][:3], r["method"], r["n"],
          " ".join(f"{g(r, k):.3f}" for k in ["F", "R2", "pairF_median", "F_noise_corr", "F_own", "Down_over_Demp"]))
agg = defaultdict(list)
for r in rows:
    if r["split"].startswith("heldout"): agg[(r["dataset"][:1], r["split"].replace("heldout_gene_", "hog_"), r["method"])].append((g(r, "F"), g(r, "n")))
for k, v in sorted(agg.items()):
    v = np.array(v); print(*k, f"{v[:, 0].mean():.3f} {v[:, 0].std():.3f} n{v[:, 1].mean():.1f} d{len(v)}")
