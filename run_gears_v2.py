"""
run_gears_v2.py -- GEARS fold runner (v2) for the two-gene benchmark.

Install (same env as the existing GEARS runner):
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    pip install torch_geometric==2.6.1          # pure-python PyG is enough for GEARS
    pip install cell-gears==0.1.2 scanpy anndata
    (your "patched" GEARS 0.1.2 must be importable as `gears`)

Differences from the original runner:
  (a) validation = random 10% of the TRAINING pairs (seeded by SEED and fold), never test pairs
  (b) SEED seeds python / numpy / torch (CPU + CUDA)
  (c) EPOCHS env (default 20)
  (d) outputs named via out_name(f"gearsS{SEED}", ...)
  (e) also saves predictions for ALL measured single perturbations that GEARS can
      predict (keys all_single_names / all_single_pred) and the training-loss history
      (loss_steps: [epoch, step, loss]; epoch_metrics: [epoch, train_mse, val_mse,
      train_mse_de, val_mse_de]) captured from GEARS' own log lines.

Env: DATASET=norman|wessels  SPLIT=rand|gd  FOLD=<int, optional>  SEED=1  EPOCHS=20
     BATCH=32  HIDDEN=64  DEVICE=cuda
Run from the project dir (the script chdirs to its own directory).
"""
import os, re, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
DS = os.environ.get("DATASET", "norman").lower()
SPLIT = os.environ.get("SPLIT", "rand").lower()
FOLD = os.environ.get("FOLD")
SEED = int(os.environ.get("SEED", "1"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH = int(os.environ.get("BATCH", "32"))
HIDDEN = int(os.environ.get("HIDDEN", "64"))
DEVICE = os.environ.get("DEVICE", "cuda")
assert DS in ("norman", "wessels") and SPLIT in ("rand", "gd", "hog")


# ----------------------------------------------------------------- shared helpers
def out_name(model, ds, split, f):
    if ds == "norman":
        return f"{model}_fold{f}.npz" if split == "rand" else f"{model}_{split}_fold{f}.npz"
    return f"{model}_wessels_{split}_fold{f}.npz"


def load_bench(ds, split):
    sfx = "" if ds == "norman" else "_wessels"
    pb = dict(np.load(f"pseudobulk{sfx}.npz", allow_pickle=True))
    genes = pb["pert_gene_list"].astype(str)
    if split == "hog":  # held-out genes: same construction as score_all.py (seed s -> rng 100+s, 25% of paired genes)
        s = int(FOLD); pairs = pb["combo_pairs"].astype(int); genes_in = np.unique(pairs.flatten())
        rng = np.random.default_rng(100 + s)
        S = set(rng.choice(genes_in, size=int(round(0.25 * len(genes_in))), replace=False))
        nin = np.array([(a in S) + (b in S) for a, b in pairs])
        fold_of = np.where(nin > 0, s, -1)
        pb["pair_genes"] = [(genes[a], genes[b]) for a, b in pairs]
        return pb, fold_of, s + 1
    sp = np.load(f"{'custom' if split == 'rand' else 'gd'}_split{sfx}.npz", allow_pickle=True)
    fold_of = sp["fold_of"].astype(int)
    K = int(sp["K"]) if "K" in sp.files else int(fold_of.max() + 1)
    pb["pair_genes"] = [(genes[a], genes[b]) for a, b in pb["combo_pairs"].astype(int)]
    return pb, fold_of, K


def condition_maps(conditions):
    """frozenset({a,b}) -> 'a+b' condition string ; gene -> 'g+ctrl' / 'ctrl+g'."""
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


def seed_all(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------- main
def main():
    import torch
    import gears.gears as gears_mod
    from gears import PertData, GEARS

    pb, fold_of, K = load_bench(DS, SPLIT)
    pd_ = PertData("./data") if DS == "norman" else PertData("./data_wessels_gears")
    if DS == "norman":
        pd_.load(data_name="norman")
    else:
        pd_.load(data_path="./data_wessels_gears/wessels")
    conds = list(pd_.adata.obs["condition"].astype(str).unique())
    pair_c, single_c = condition_maps(conds)
    var_genes = list(pd_.adata.var["gene_name"].astype(str))
    pos = {g: i for i, g in enumerate(var_genes)}
    gidx = np.array([pos[g] for g in pb["gene_names"].astype(str)])  # reorder to pseudobulk gene order

    # Capture GEARS' log lines (print_sys) to recover the loss history.
    log_lines = []
    _orig_print = gears_mod.print_sys
    def _capture(s):
        log_lines.append(str(s)); _orig_print(s)
    gears_mod.print_sys = _capture

    folds = [int(FOLD)] if FOLD is not None else list(range(K))
    for f in folds:
        seed_all(SEED)
        tr, va, te = split_pairs(fold_of, f, SEED)
        to_c = lambda idx: [pair_c[frozenset(pb["pair_genes"][m])] for m in idx
                            if frozenset(pb["pair_genes"][m]) in pair_c]
        train_c = ["ctrl"] + sorted(set(single_c.values())) + to_c(tr)
        val_c, test_c = to_c(va), to_c(te)
        missing = len(te) - len(test_c)
        if missing:
            print(f"[warn] fold {f}: {missing} test pairs not present in GEARS adata -> skipped")
        # 'no_test' => GEARS trains with train/val loaders only and skips its own test analysis
        pd_.set2conditions = {"train": train_c, "val": val_c, "test": test_c}
        pd_.split = "no_test"
        # unique tag so GEARS caches a separate co-expression graph per fold (built from that fold's training cells)
        pd_.seed = f"v2f{f}s{SEED}"
        pd_.get_dataloader(batch_size=BATCH, test_batch_size=128)

        log_lines.clear()
        g = GEARS(pd_, device=DEVICE)
        g.model_initialize(hidden_size=HIDDEN)
        g.train(epochs=EPOCHS, lr=1e-3)

        # ---- predictions for test pairs
        names, preds, keep = [], [], []
        for m in te:
            c = pair_c.get(frozenset(pb["pair_genes"][m]))
            if c is None:
                continue
            a, b = c.split("+")
            if a not in g.pert_list or b not in g.pert_list:
                print(f"[warn] {c}: gene not in GEARS pert_list -> skipped"); continue
            p = g.predict([[a, b]])[f"{a}_{b}"]
            names.append(c); preds.append(np.asarray(p).ravel()[gidx]); keep.append(m)
        # ---- singles: genes of the test pairs, and all measured predictable singles
        def pred_single(gn):
            return np.asarray(g.predict([[gn]])[gn]).ravel()[gidx]
        test_genes = sorted({x for c in names for x in c.split("+")})
        s_names = [x for x in test_genes if x in g.pert_list]
        s_pred = [pred_single(x) for x in s_names]
        all_s = sorted(x for x in single_c if x in g.pert_list)
        all_pred = [pred_single(x) for x in all_s]

        # ---- loss history parsed from the captured log (lines end with a period, so numbers are matched strictly)
        NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
        txt = "\n".join(map(str, log_lines))
        try:
            steps = [tuple(map(float, r)) for r in re.findall(rf"Epoch (\d+) Step (\d+) Train Loss: {NUM}", txt)]
            ep = re.findall(rf"Epoch (\d+): Train Overall MSE: {NUM} Validation Overall MSE: {NUM}", txt)
            de = re.findall(rf"Train Top 20 DE MSE: {NUM} Validation Top 20 DE MSE: {NUM}", txt)
            ep_metrics = [tuple(map(float, a + b)) for a, b in zip(ep, de)]
        except Exception as e:
            print("[warn] could not parse training log:", e); steps, ep_metrics = [], []
        G = len(gidx)
        out = out_name(f"gearsS{SEED}", DS, SPLIT, f)
        np.savez(out,
                 combo_names=np.array(names, dtype=str),
                 combo_pred=np.array(preds, dtype=np.float32).reshape(-1, G),
                 combo_idx=np.array(keep, dtype=int),
                 single_names=np.array(s_names, dtype=str),
                 single_pred=np.array(s_pred, dtype=np.float32).reshape(-1, G),
                 all_single_names=np.array(all_s, dtype=str),
                 all_single_pred=np.array(all_pred, dtype=np.float32).reshape(-1, G),
                 val_pairs=np.array(val_c, dtype=str),
                 loss_steps=np.array(steps, dtype=np.float32).reshape(-1, 3),
                 epoch_metrics=np.array(ep_metrics, dtype=np.float32).reshape(-1, 5),
                 fold=f, seed=SEED, epochs=EPOCHS)
        print(f"[fold {f}] wrote {out}: {len(names)} pairs, {len(all_s)} singles")
        del g; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
