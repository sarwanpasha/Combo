"""
run_attentionpert.py -- AttentionPert (Bai et al., Bioinformatics 2024) on the two-gene folds.

Install (AttentionPert is NOT on PyPI; it vendors a modified GEARS as package `attnpert`):
    git clone https://github.com/BaiDing1234/AttentionPert.git ./AttentionPert   (or ATTNPERT_DIR=...)
    # works in the GEARS env (torch 2.5.1+cu121, torch_geometric 2.6.x, scanpy, networkx);
    # the repo's environment.yml is py3.7/cu11.7 but the code only needs torch + PyG.
    The model needs:
      * Gene2Vec embeddings: gene2vec_dim_200_iter_9_w2v.txt (auto-downloaded from
        github.com/jingcheng-du/Gene2vec, ~57 MB) -> <data>/gene2vec.npy (per gene list)
      * a GO gene-gene graph <data>/go.csv: built here from ./data/gene2go_all.pkl (the file GEARS
        downloads) with the same Jaccard>0.1 rule as attnpert.utils.get_go_auto, vectorised.
        (The paper used "reduced GO graphs" shared via SharePoint; drop them in to use those.)
Data: attnpert caches its own data_pyg/cell_graphs.pkl with a DIFFERENT format than GEARS 0.1.2,
so we use a separate dir data_attnpert/<ds>/ holding a symlink to perturb_processed.h5ad.

Hyper-parameters = official run_attnpert.py: PL_PW_non_add_Model, hidden 64, batch 128,
epochs 20, lr 1e-3, pert_local_min_weight .75, K=1, 2 heads x 64, act softmax, beta 0.05.
Validation = random 10% of training pairs (seeded); test pairs are never in train/val.
attnpert has no predict(): we build GEARS-style input graphs (300 random control cells,
pert flag column) and average best_model outputs over cells.
Env: DATASET, SPLIT, FOLD, SEED=1, EPOCHS=20, BATCH=128, ATTNPERT_DIR=./AttentionPert
"""
import os, sys, pickle, random, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
DS = os.environ.get("DATASET", "norman").lower()
SPLIT = os.environ.get("SPLIT", "rand").lower()
FOLD = os.environ.get("FOLD")
SEED = int(os.environ.get("SEED", "1"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH = int(os.environ.get("BATCH", "128"))
N_CTRL = int(os.environ.get("N_CTRL_PRED", "300"))
sys.path.insert(0, os.path.abspath(os.environ.get("ATTNPERT_DIR", "./AttentionPert")))
SRC_H5AD = "data/norman/perturb_processed.h5ad" if DS == "norman" else \
    "data_wessels_gears/wessels/perturb_processed.h5ad"
AP_DIR = f"data_attnpert/{DS}"
G2V_URL = ("https://raw.githubusercontent.com/jingcheng-du/Gene2vec/master/"
           "pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt")
assert DS in ("norman", "wessels") and SPLIT in ("rand", "gd")


# ----------------------------------------------------------------- shared helpers
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
    genes = pb["pert_gene_list"].astype(str)
    pb["pair_genes"] = [(genes[a], genes[b]) for a, b in pb["combo_pairs"].astype(int)]
    return pb, fold_of, K


def condition_maps(conditions):
    pair, single = {}, {}
    for c in conditions:
        g = [x for x in str(c).split("+") if x != "ctrl"]
        if len(g) == 1:
            single[g[0]] = c
        elif len(g) == 2 and g[0] != g[1]:
            pair[frozenset(g)] = c
    return pair, single


def split_pairs(fold_of, f, seed, frac=0.1):
    test = np.where(fold_of == f)[0]
    train = np.where(fold_of != f)[0]
    rng = np.random.default_rng([seed, f])
    val = np.sort(rng.choice(train, max(1, int(round(frac * len(train)))), replace=False))
    return np.setdiff1d(train, val), val, test


# ----------------------------------------------------------------- resources
def build_go_csv(gene_list, path):
    """Vectorised version of attnpert.utils.get_go_auto (Jaccard of GO term sets, keep > 0.1)."""
    import pandas as pd, scipy.sparse as sp
    with open("data/gene2go_all.pkl", "rb") as fh:
        gene2go = pickle.load(fh)
    genes = [g for g in dict.fromkeys(gene_list) if g in gene2go]
    terms = {t: i for i, t in enumerate(sorted({t for g in genes for t in gene2go[g]}))}
    rows = [i for i, g in enumerate(genes) for t in set(gene2go[g])]
    cols = [terms[t] for g in genes for t in set(gene2go[g])]
    M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(genes), len(terms)))
    inter = (M @ M.T).tocoo()
    size = np.asarray(M.sum(1)).ravel()
    jac = inter.data / (size[inter.row] + size[inter.col] - inter.data)
    k = jac > 0.1
    gl = np.array(genes)
    pd.DataFrame({"source": gl[inter.row[k]], "target": gl[inter.col[k]],
                  "importance": jac[k]}).to_csv(path, index=False)
    print(f"wrote {path}: {k.sum()} edges over {len(genes)} genes")


def build_gene2vec(gene_names, path, seed):
    txt = "data/gene2vec_dim_200_iter_9_w2v.txt"
    if not os.path.exists(txt):
        print("downloading Gene2Vec ..."); urllib.request.urlretrieve(G2V_URL, txt)
    vec = {}
    with open(txt) as fh:
        next(fh)
        for line in fh:
            p = line.split()
            if len(p) >= 201:
                vec[p[0]] = np.array(p[1:201], dtype=float)
    allv = np.stack(list(vec.values()))
    rng = np.random.default_rng(seed)   # missing genes ~ N(mean, cov) as in gene2vec_example.py
    out = np.stack([vec[g] if g in vec else rng.multivariate_normal(allv.mean(0), np.cov(allv.T))
                    for g in gene_names])
    np.save(path, out)
    print(f"wrote {path}: {out.shape}, {sum(g not in vec for g in gene_names)} genes w/o Gene2Vec")


# ----------------------------------------------------------------- main
def main():
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from attnpert import PertData
    from attnpert.attnpert import ATTNPERT_RECORD_TRAIN
    from attnpert.model import PL_PW_non_add_Model

    os.makedirs(AP_DIR, exist_ok=True)
    dst = os.path.join(AP_DIR, "perturb_processed.h5ad")
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(SRC_H5AD), dst)
    pb, fold_of, K = load_bench(DS, SPLIT)
    pd_ = PertData("./data_attnpert")
    pd_.load(data_path=AP_DIR)                    # builds/loads data_attnpert/<ds>/data_pyg
    processed = pd_.dataset_processed             # get_dataloader() deletes it -> keep a handle
    pd_.seed, pd_.subgroup, pd_.train_gene_set_size = SEED, None, 0.75
    conds = list(pd_.adata.obs["condition"].astype(str).unique())
    pair_c, single_c = condition_maps(conds)
    real_genes = list(pd_.adata.var["gene_name"].astype(str))
    pos = {g: i for i, g in enumerate(real_genes)}
    gidx = np.array([pos[g] for g in pb["gene_names"].astype(str)])
    ctrl_X = pd_.adata[pd_.adata.obs["condition"] == "ctrl"].X
    ctrl_X = ctrl_X.toarray() if hasattr(ctrl_X, "toarray") else np.asarray(ctrl_X)

    folds = [int(FOLD)] if FOLD is not None else list(range(K))
    for f in folds:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        tr, va, te = split_pairs(fold_of, f, SEED)
        cond = lambda idx: [pair_c[frozenset(pb["pair_genes"][m])] for m in idx
                            if frozenset(pb["pair_genes"][m]) in pair_c]
        pd_.dataset_processed = processed
        pd_.set2conditions = {"train": ["ctrl"] + sorted(set(single_c.values())) + cond(tr),
                              "val": cond(va), "test": cond(te)}
        pd_.split = "no_test"                     # train/val loaders only; we predict ourselves
        pd_.get_dataloader(batch_size=BATCH, test_batch_size=BATCH)
        gene_names = [str(g) for g in pd_.gene_names]   # expression genes + unmeasured pert genes
        g2v = os.path.join(AP_DIR, "gene2vec.npy")
        if not os.path.exists(g2v):
            build_gene2vec(gene_names, g2v, seed=0)
        if not os.path.exists(os.path.join(AP_DIR, "go.csv")):
            build_go_csv(gene_names, os.path.join(AP_DIR, "go.csv"))

        m = ATTNPERT_RECORD_TRAIN(pd_, device="cuda", weight_bias_track=False,
                                  exp_name=f"AP_{DS}_{SPLIT}_f{f}_s{SEED}")
        m.model_initialize(hidden_size=64, model_class=PL_PW_non_add_Model,
                           exp_name=f"AP_{DS}_{SPLIT}_f{f}_s{SEED}",
                           gene2vec_args={"gene2vec_file": g2v}, pert_local_min_weight=0.75,
                           pert_local_conv_K=1, pert_weight_heads=2, pert_weight_head_dim=64,
                           pert_weight_act="softmax", non_add_beta=5e-2, record_pred=False)
        m.train(epochs=EPOCHS, valid_every=1)
        model = m.best_model.eval()
        node = {g: i for i, g in enumerate(gene_names)}
        N, R = len(gene_names), len(real_genes)

        def predict(genes):
            rng = np.random.default_rng([SEED, f, len(genes)] + [node[g] for g in genes])
            idx = rng.integers(0, ctrl_X.shape[0], N_CTRL)
            graphs = []
            for i in idx:
                x = np.zeros((N, 2), np.float32); x[:R, 0] = ctrl_X[i]
                for g in genes:
                    x[node[g], 1] = 1.0
                graphs.append(Data(x=torch.from_numpy(x), pert="+".join(genes)))
            outs = []
            with torch.no_grad():
                for b in DataLoader(graphs, batch_size=64, shuffle=False):
                    outs.append(model(b.to("cuda"))[:, :R].float().cpu().numpy())
            return np.concatenate(outs).mean(0)[gidx]

        names, preds, keep = [], [], []
        for mi in te:
            c = pair_c.get(frozenset(pb["pair_genes"][mi]))
            if c is None:
                print(f"[warn] {pb['pair_genes'][mi]} not in adata -> skipped"); continue
            names.append(c); preds.append(predict(c.split("+"))); keep.append(mi)
        s_names = sorted({x for c in names for x in c.split("+")})
        s_pred = [predict([g]) for g in s_names]
        G = len(gidx)
        out = out_name("attnpert", DS, SPLIT, f)
        np.savez(out, combo_names=np.array(names, dtype=str),
                 combo_pred=np.array(preds, np.float32).reshape(-1, G), combo_idx=np.array(keep, int),
                 single_names=np.array(s_names, dtype=str),
                 single_pred=np.array(s_pred, np.float32).reshape(-1, G),
                 train_loss=np.asarray(m.train_losses, np.float32),
                 val_de_mse=np.asarray(m.val_de_mses, np.float32), fold=f, seed=SEED)
        print(f"[fold {f}] wrote {out}: {len(names)} pairs")
        del m, model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
