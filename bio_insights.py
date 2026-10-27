# Biological characterization of pairwise interactions in both screens.
#  (1) Norman-style genetic-interaction (GI) model per pair: (y - mu0) ~ c1*alpha_i + c2*alpha_j (Theil-Sen, no intercept,
#      on genes expressed in control), giving magnitude, dominance, model fit and similarity of the singles.
#  (2) Rescaling versus novel interaction: projection of delta onto span{alpha_i, alpha_j}.
#  (3) Gene-level direction of non-additivity (buffering / synergy / opposite), as in Ahlmann-Eltze et al.
#  (4) Similar single effects and shared annotation versus interaction type.
#  (5) Cross-validated ridge predictions per pair: what makes an interaction predictable, and prioritization of
#      strongly interacting pairs.
#  (6) Interaction components: top pairs and expression/variance-matched GO enrichment of top-loading genes.
# Outputs bio_{ds}.tsv (per pair), bio_{ds}_components.tsv, bio_summary.json and prints a report.
import os, json, pickle, numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import TheilSenRegressor, Ridge
os.chdir(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)
g2go = pickle.load(open("data/gene2go_all.pkl", "rb"))
NAMED = {"CBL+CNN1": "synergy", "KLF1+CEBPA": "epistasis (CEBPA dominant)", "CEBPA+KLF1": "epistasis (CEBPA dominant)",
         "PTPN12+SNAI1": "suppression", "SNAI1+PTPN12": "suppression", "FEV+CBFA2T3": "potentiation/neomorphic",
         "CBFA2T3+FEV": "potentiation/neomorphic", "DUSP9+ETS2": "DUSP9 dominant", "ETS2+DUSP9": "DUSP9 dominant",
         "DUSP9+MAPK1": "mutual suppression", "MAPK1+DUSP9": "mutual suppression", "ETS2+MAPK1": "synergy", "MAPK1+ETS2": "synergy",
         "PLK4+STIL": "neomorphic", "STIL+PLK4": "neomorphic"}
summary = {}

def gi_fit(y, a, b):
    X = np.stack([a, b], 1)
    ts = TheilSenRegressor(fit_intercept=False, max_subpopulation=10000, random_state=1000).fit(X, y)
    c1, c2 = ts.coef_
    fit = X @ ts.coef_
    r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return c1, c2, r2

def ranks(x): return np.argsort(np.argsort(x))

for ds, pbf, sf in [("norman", "pseudobulk.npz", "custom_split.npz"), ("wessels", "pseudobulk_wessels.npz", "custom_split_wessels.npz")]:
    pb = np.load(pbf, allow_pickle=True)
    ctrl = pb["ctrl_mean"]; alpha = np.nan_to_num(pb["alpha"]); truth = pb["combo_truth"]; pairs = pb["combo_pairs"]
    names = list(map(str, pb["combo_names"])); pg = list(map(str, pb["pert_gene_list"])); genes = np.array(list(map(str, pb["gene_names"])))
    M, G = truth.shape
    resp = truth - ctrl[None]; add = alpha[pairs[:, 0]] + alpha[pairs[:, 1]]; delta = resp - add
    pert_set = set(pg)
    sel = (np.expm1(ctrl) > 0.5) & np.array([g not in pert_set for g in genes])
    out = {}
    # ---- (1) GI model and (2) rescaling decomposition ----
    rows = []
    for m in range(M):
        a, b, y = alpha[pairs[m, 0]], alpha[pairs[m, 1]], resp[m]
        c1, c2, r2 = gi_fit(y[sel], a[sel], b[sel])
        A2 = np.stack([a, b], 1); coef, *_ = np.linalg.lstsq(A2, delta[m], rcond=None)
        resc = A2 @ coef; phi = (resc ** 2).sum() / (delta[m] ** 2).sum()
        sim = np.corrcoef(a, b)[0, 1]
        mag = np.hypot(c1, c2); dom = abs(np.log10(max(abs(c1), 1e-3) / max(abs(c2), 1e-3)))
        share = np.linalg.norm(delta[m]) / np.linalg.norm(resp[m])
        rows.append(dict(pair=names[m], c1=c1, c2=c2, magnitude=mag, dominance=dom, fit_r2=r2, single_corr=sim,
                         rescaling_frac=phi, share=share, delta_norm=np.linalg.norm(delta[m])))
    R = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "pair"}
    # operational classes (our rules, stated in the paper): neomorphic if the linear model fits poorly,
    # otherwise epistatic if one coefficient dominates, synergistic if the magnitude exceeds that of an additive pair
    # (c1 = c2 = 1, magnitude sqrt(2)), buffering otherwise
    cls = np.where(R["fit_r2"] < 0.5, "neomorphic", np.where(R["dominance"] > 0.5, "epistatic",
                   np.where(R["magnitude"] > np.sqrt(2), "synergistic", "buffering")))
    for k, c in zip(rows, cls): k["class"] = c
    out["class_counts"] = {c: int((cls == c).sum()) for c in np.unique(cls)}
    out["magnitude_median"] = float(np.median(R["magnitude"])); out["frac_magnitude_lt_additive"] = float((R["magnitude"] < np.sqrt(2)).mean())
    out["fit_r2_median"] = float(np.median(R["fit_r2"]))
    w = (delta ** 2).sum(1)
    out["rescaling_frac_pooled"] = float((R["rescaling_frac"] * w).sum() / w.sum()); out["rescaling_frac_median"] = float(np.median(R["rescaling_frac"]))
    named = {n: (rows[names.index(n)]["c1"], rows[names.index(n)]["c2"], rows[names.index(n)]["class"], NAMED[n]) for n in names if n in NAMED}
    out["named_pairs"] = {n: [round(v[0], 2), round(v[1], 2), v[2], v[3]] for n, v in named.items()}
    # ---- (3) gene-level direction of non-additivity (entries with both singles in same direction and |add|>0.2) ----
    a1, a2 = alpha[pairs[:, 0]], alpha[pairs[:, 1]]
    same = (np.sign(a1) == np.sign(a2)) & (np.abs(add) > 0.2) & (np.abs(a1) > 0.05) & (np.abs(a2) > 0.05)
    r_, ad_ = resp[same], add[same]
    buff = ((np.sign(r_) == np.sign(ad_)) & (np.abs(r_) < np.abs(ad_))).mean()
    syn = ((np.sign(r_) == np.sign(ad_)) & (np.abs(r_) > np.abs(ad_))).mean()
    opp = (np.sign(r_) != np.sign(ad_)).mean()
    slope = float((r_ * ad_).sum() / (ad_ ** 2).sum())
    out["entry_direction"] = dict(n_entries=int(same.sum()), buffering=float(buff), synergistic=float(syn), opposite=float(opp), pooled_slope=slope)
    # ---- (4) similarity / shared annotation ----
    def go_set(g): return {t for t in g2go.get(g, set())}
    shared = np.array([len(go_set(pg[i]) & go_set(pg[j])) > 0 for i, j in pairs])
    jac = np.array([len(go_set(pg[i]) & go_set(pg[j])) / max(1, len(go_set(pg[i]) | go_set(pg[j]))) for i, j in pairs])
    out["spearman_singlecorr_magnitude"] = [float(x) for x in spearmanr(R["single_corr"], R["magnitude"])]
    out["spearman_singlecorr_share"] = [float(x) for x in spearmanr(R["single_corr"], R["share"])]
    out["spearman_GOjaccard_magnitude"] = [float(x) for x in spearmanr(jac, R["magnitude"])]
    hi = R["single_corr"] > 0.5
    if 3 <= hi.sum() <= M - 3:
        out["similar_vs_dissimilar_magnitude"] = [float(np.median(R["magnitude"][hi])), float(np.median(R["magnitude"][~hi])),
                                                  float(mannwhitneyu(R["magnitude"][hi], R["magnitude"][~hi]).pvalue), int(hi.sum())]
        out["similar_vs_dissimilar_share"] = [float(np.median(R["share"][hi])), float(np.median(R["share"][~hi])),
                                              float(mannwhitneyu(R["share"][hi], R["share"][~hi]).pvalue)]
    # ---- (5) per-pair predictability (5-fold random split, ridge rank 16) and prioritization ----
    fold = np.load(sf)["fold_of"]; K = int(np.load(sf)["K"]); pred = np.zeros_like(delta)
    for f in range(K):
        te = fold == f; tr = ~te
        _, _, Vt = np.linalg.svd(delta[tr], full_matrices=False); B = Vt[:16]
        reg = Ridge(alpha=10.0).fit(add[tr], delta[tr] @ B.T)
        pred[te] = reg.predict(add[te]) @ B
    err = ((pred - delta) ** 2).sum(1); pf = 1 - err / w
    for k, v in zip(rows, pf): k["ridge_pair_F"] = v
    for k in ["magnitude", "dominance", "fit_r2", "single_corr", "rescaling_frac", "share"]:
        out[f"spearman_pairF_{k}"] = [float(x) for x in spearmanr(pf, R[k])]
    # recovery of rescaling vs novel parts (pooled)
    Er = En = Dr = Dn = 0.0
    for m in range(M):
        A2 = np.stack([alpha[pairs[m, 0]], alpha[pairs[m, 1]]], 1)
        Q, _ = np.linalg.qr(A2)
        dr, dn = Q @ (Q.T @ delta[m]), delta[m] - Q @ (Q.T @ delta[m])
        pr, pn = Q @ (Q.T @ pred[m]), pred[m] - Q @ (Q.T @ pred[m])
        Er += ((pr - dr) ** 2).sum(); En += ((pn - dn) ** 2).sum(); Dr += (dr ** 2).sum(); Dn += (dn ** 2).sum()
    out["ridge_F_rescaling_part"] = float(1 - Er / Dr); out["ridge_F_novel_part"] = float(1 - En / Dn)
    out["rescaling_share_of_D"] = float(Dr / (Dr + Dn))
    # by class
    out["ridge_pairF_by_class"] = {c: float(np.median(pf[cls == c])) for c in np.unique(cls)}
    # prioritization: rank pairs by predicted interaction norm; precision@top-quartile for strongest interactions
    pn_norm = np.linalg.norm(pred, axis=1); tn = np.linalg.norm(delta, axis=1)
    k = max(1, M // 4); top_true = set(np.argsort(-tn)[:k]); top_pred = set(np.argsort(-pn_norm)[:k])
    out["prioritize_spearman_norm"] = [float(x) for x in spearmanr(pn_norm, tn)]
    out["prioritize_precision_topquartile"] = len(top_true & top_pred) / k
    # baseline from training-pair means of each gene ("shared-gene" heuristic)
    sg = np.zeros(M)
    for f in range(K):
        te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
        for m in te:
            nb = [t for t in tr if set(pairs[t]) & set(pairs[m])]
            sg[m] = np.mean(tn[nb]) if nb else np.mean(tn[tr])
    top_sg = set(np.argsort(-sg)[:k])
    out["prioritize_sharedgene_spearman"] = [float(x) for x in spearmanr(sg, tn)]
    out["prioritize_sharedgene_precision"] = len(top_true & top_sg) / k
    # ---- (5b) gene-intrinsic structure of interaction phenotypes (leave-one-pair-out) ----
    def loo_additive(yv, sign_model):
        # sign_model: y_m ~ s_g1 - s_g2 (antisymmetric, dominance) ; else y_m ~ mu + u_g1 + u_g2 (symmetric)
        preds = np.full(M, np.nan)
        for m in range(M):
            keep = np.arange(M) != m
            g1, g2 = pairs[m]
            if not (np.isin(g1, pairs[keep]).any() and np.isin(g2, pairs[keep]).any()): continue
            Xd = np.zeros((keep.sum(), P_ + (0 if sign_model else 1)))
            for r_, mm in enumerate(np.where(keep)[0]):
                a_, b_ = pairs[mm]
                if sign_model: Xd[r_, a_] += 1; Xd[r_, b_] -= 1
                else: Xd[r_, a_] += 1; Xd[r_, b_] += 1; Xd[r_, -1] = 1
            coef_, *_ = np.linalg.lstsq(Xd, yv[keep], rcond=None)
            x = np.zeros(Xd.shape[1])
            if sign_model: x[g1] += 1; x[g2] -= 1
            else: x[g1] += 1; x[g2] += 1; x[-1] = 1
            preds[m] = x @ coef_
        ok = np.isfinite(preds)
        return preds, ok
    P_ = len(pg)
    dlog = np.log2(np.maximum(np.abs(R["c1"]), 1e-3) / np.maximum(np.abs(R["c2"]), 1e-3))
    pr, ok = loo_additive(dlog, True)
    out["dominance_hierarchy_loo"] = dict(n=int(ok.sum()), sign_accuracy=float((np.sign(pr[ok]) == np.sign(dlog[ok])).mean()),
                                          spearman=float(spearmanr(pr[ok], dlog[ok])[0]))
    for nm_, yv in [("log_magnitude", np.log(R["magnitude"])), ("log_share", np.log(R["share"])), ("rescaling_frac", R["rescaling_frac"])]:
        pr, ok = loo_additive(yv, False)
        out[f"gene_additive_{nm_}_loo"] = dict(n=int(ok.sum()), spearman=float(spearmanr(pr[ok], yv[ok])[0]),
                                              r2=float(1 - ((pr[ok] - yv[ok]) ** 2).sum() / ((yv[ok] - yv[ok].mean()) ** 2).sum()))
    # per-gene dominance score (full fit) for reporting
    Xd = np.zeros((M, P_))
    for m in range(M): Xd[m, pairs[m, 0]] += 1; Xd[m, pairs[m, 1]] -= 1
    sc_, *_ = np.linalg.lstsq(Xd, dlog, rcond=None); used = np.unique(pairs.flatten())
    order_ = used[np.argsort(-sc_[used])]
    out["dominance_rank_top"] = [(pg[g], round(float(sc_[g]), 2)) for g in order_[:8]]
    out["dominance_rank_bottom"] = [(pg[g], round(float(sc_[g]), 2)) for g in order_[-8:]]
    # ---- (6) components: top pairs and matched GO enrichment ----
    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    var_g = resp.var(0); mean_g = ctrl
    bins_m = np.digitize(mean_g, np.quantile(mean_g, np.linspace(0, 1, 11)[1:-1]))
    bins_v = np.digitize(var_g, np.quantile(var_g, np.linspace(0, 1, 11)[1:-1]))
    cell = bins_m * 10 + bins_v
    annotated = np.array([g in g2go for g in genes])
    term2genes = {}
    for gi_, g in enumerate(genes):
        if annotated[gi_]:
            for t in g2go[g]: term2genes.setdefault(t, set()).add(gi_)
    terms = {t: s for t, s in term2genes.items() if 5 <= len(s) <= 500}
    tlist = list(terms); tmat = np.zeros((len(tlist), G), bool)
    for ti, t in enumerate(tlist): tmat[ti, list(terms[t])] = True
    comp_rows = []
    nB = 2000
    for kc in range(5):
        load = Vt[kc] * np.sign(Vt[kc][np.argmax(np.abs(Vt[kc]))])
        top = np.argsort(-np.abs(load))[:50]
        top = top[annotated[top]]
        obs = tmat[:, top].sum(1)
        # matched null: same number of genes from the same mean x variance cells, annotated only
        null_ge = np.zeros(len(tlist)); null_mean = np.zeros(len(tlist))
        pools = {c: np.where((cell == c) & annotated)[0] for c in np.unique(cell[top])}
        need = {c: int((cell[top] == c).sum()) for c in pools}
        for bb in range(nB):
            samp = np.concatenate([rng.choice(pools[c], size=min(need[c], len(pools[c])), replace=False) for c in pools])
            nb_ = tmat[:, samp].sum(1); null_ge += nb_ >= obs; null_mean += nb_
        p_emp = (1 + null_ge) / (1 + nB); fold_enr = obs / np.maximum(null_mean / nB, 1e-9)
        cand = np.where(obs >= 3)[0]
        order = cand[np.lexsort((-fold_enr[cand], p_emp[cand]))][:3]
        # BH over candidate terms
        pc = p_emp[cand]; o = np.argsort(pc); bh = np.empty_like(pc); mcount = len(pc)
        bh[o] = np.minimum.accumulate((pc[o] * mcount / np.arange(1, mcount + 1))[::-1])[::-1] if mcount else pc
        bhd = dict(zip(cand, np.minimum(bh, 1)))
        tp = np.argsort(-np.abs(U[:, kc] * S[kc]))[:5]
        comp_rows.append(dict(component=kc + 1, var_frac=float(S[kc] ** 2 / (S ** 2).sum()),
                              top_pairs=[f"{names[t]}({cls[t]})" for t in tp],
                              top_genes=[str(genes[g]) for g in np.argsort(-np.abs(load))[:12]],
                              terms=[dict(term=tlist[t], overlap=int(obs[t]), size=int(tmat[t].sum()), fold=float(fold_enr[t]),
                                          p_matched=float(p_emp[t]), q_bh=float(bhd[t])) for t in order]))
    out["components"] = comp_rows
    # ---- write ----
    keys = list(rows[0].keys())
    with open(f"bio_{ds}.tsv", "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows: fh.write("\t".join(str(round(r[k], 4)) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    summary[ds] = out
    print(f"\n######## {ds} ########")
    for k_, v in out.items():
        if k_ != "components": print(k_, v)
    for c in comp_rows:
        print(f"component {c['component']} ({c['var_frac']:.3f}) pairs={c['top_pairs']} genes={c['top_genes']}")
        for t in c["terms"]: print("    ", t)
json.dump(summary, open("bio_summary.json", "w"), indent=1, default=float)
print("wrote bio_summary.json")
# ---- GO term names for reported terms (QuickGO) ----
import urllib.request
ids = sorted({t["term"] for ds in summary for c in summary[ds]["components"] for t in c["terms"]})
names_go = {}
for t in ids:
    try:
        u = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/" + t.replace(":", "%3A")
        r = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"Accept": "application/json"}), timeout=20).read())
        names_go[t] = r["results"][0]["name"]
    except Exception:
        names_go[t] = ""
for ds in summary:
    for c in summary[ds]["components"]:
        for t in c["terms"]: t["name"] = names_go.get(t["term"], "")
json.dump(summary, open("bio_summary.json", "w"), indent=1, default=float)
for ds in summary:
    for c in summary[ds]["components"]:
        print(ds, c["component"], [(t["name"], t["overlap"], t["size"], round(t["fold"], 1), t["p_matched"], round(t["q_bh"], 3)) for t in c["terms"]])
