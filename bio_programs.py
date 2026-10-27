# Interaction effects on erythroid and myeloid programs of K562 cells (Norman screen).
# For every pair: program scores (mean over program genes) of the observed response y - mu0, of the additive
# expectation alpha_i + alpha_j and of the interaction delta. Also: per-gene consistency of buffering and synergy
# across pairs (which readout genes are most often pulled toward or pushed beyond the additive expectation).
import os, json, numpy as np
from scipy.stats import spearmanr, wilcoxon
os.chdir(os.path.dirname(os.path.abspath(__file__)))
pb = np.load("pseudobulk.npz", allow_pickle=True)
genes = list(map(str, pb["gene_names"])); gi = {g: k for k, g in enumerate(genes)}
ctrl = pb["ctrl_mean"]; alpha = np.nan_to_num(pb["alpha"]); truth = pb["combo_truth"]; pairs = pb["combo_pairs"]
names = list(map(str, pb["combo_names"])); pg = list(map(str, pb["pert_gene_list"]))
add = alpha[pairs[:, 0]] + alpha[pairs[:, 1]]; resp = truth - ctrl[None]; delta = resp - add
PROG = {
    "erythroid": ["HBG1", "HBG2", "HBZ", "HBA1", "HBA2", "HBE1", "HBD", "GYPA", "GYPB", "GYPC", "ALAS2", "SLC25A37",
                  "BLVRB", "KLF1", "GATA1", "TFRC", "EPOR", "SLC4A1", "ANK1", "SPTA1", "FECH", "UROD"],
    "myeloid": ["LST1", "AIF1", "TYROBP", "FCER1G", "LYZ", "S100A8", "S100A9", "ITGAM", "CSF3R", "MPO", "ELANE",
                "CD14", "CD33", "SPI1", "CEBPB", "CEBPE", "LGALS1", "CTSG", "PRTN3", "AZU1", "SRGN", "VIM", "CFD"],
}
out = {}
rows = []
for p, gl in PROG.items():
    # perturbed genes are excluded so that on-target activation does not enter the program scores
    ix = [gi[g] for g in gl if g in gi and g not in set(pg)]; out[f"{p}_genes_measured"] = [genes[k] for k in ix]
    obs, ad, de = resp[:, ix].mean(1), add[:, ix].mean(1), delta[:, ix].mean(1)
    for m in range(len(names)):
        if len(rows) <= m: rows.append({"pair": names[m]})
        rows[m].update({f"{p}_obs": obs[m], f"{p}_add": ad[m], f"{p}_int": de[m]})
    # does the interaction oppose the additive program shift (buffering) or extend it (synergy)?
    big = np.abs(ad) > np.percentile(np.abs(ad), 50)
    same = np.sign(de[big]) == np.sign(ad[big])
    out[f"{p}_frac_extend_among_large_additive"] = float(same.mean())
    out[f"{p}_slope_obs_on_add"] = float((obs * ad).sum() / (ad ** 2).sum())
    out[f"{p}_spearman_obs_add"] = [float(x) for x in spearmanr(obs, ad)]
    out[f"{p}_wilcoxon_abs_obs_vs_abs_add"] = [float(np.median(np.abs(obs) - np.abs(ad))), float(wilcoxon(np.abs(obs), np.abs(ad)).pvalue)]
    # pairs whose interaction flips the sign of the program response
    flip = (np.sign(obs) != np.sign(ad)) & (np.abs(obs) > 0.05) & (np.abs(ad) > 0.05)
    out[f"{p}_sign_flip_pairs"] = [names[m] for m in np.where(flip)[0]]
    order = np.argsort(de)
    out[f"{p}_most_negative_interaction"] = [(names[m], round(float(de[m]), 3), round(float(ad[m]), 3)) for m in order[:8]]
    out[f"{p}_most_positive_interaction"] = [(names[m], round(float(de[m]), 3), round(float(ad[m]), 3)) for m in order[::-1][:8]]
# relation between the two programs in the interaction
e = np.array([r["erythroid_int"] for r in rows]); y = np.array([r["myeloid_int"] for r in rows])
out["spearman_erythroid_int_vs_myeloid_int"] = [float(x) for x in spearmanr(e, y)]
# per readout gene: across pairs with a sizeable additive change, fraction of pairs where the interaction opposes it
mask = np.abs(add) > 0.1
opp = ((np.sign(delta) != np.sign(add)) & mask).sum(0); cnt = mask.sum(0)
frac = np.where(cnt >= 20, opp / np.maximum(cnt, 1), np.nan)
ok = ~np.isnan(frac)
out["readout_genes_tested"] = int(ok.sum())
out["readout_buffered_median_frac"] = float(np.nanmedian(frac))
o = np.argsort(np.where(ok, frac, -1))[::-1]
out["most_buffered_readouts"] = [(genes[k], round(float(frac[k]), 3), int(cnt[k])) for k in o[:15]]
o2 = np.argsort(np.where(ok, frac, 2))
out["most_synergistic_readouts"] = [(genes[k], round(float(frac[k]), 3), int(cnt[k])) for k in o2[:15]]
with open("bio_programs.json", "w") as fh: json.dump(out, fh, indent=1)
with open("bio_programs.tsv", "w") as fh:
    ks = list(rows[0].keys()); fh.write("\t".join(ks) + "\n")
    for r in rows: fh.write("\t".join(str(r[k]) if k == "pair" else f"{r[k]:.4f}" for k in ks) + "\n")
for k, v in out.items(): print(k, v)
