"""
run_cpa.py -- Compositional Perturbation Autoencoder (cpa-tools) on the two-gene folds.

Install (SEPARATE env: cpa-tools 0.8.8 pins torch<=2.0.1, anndata<0.10, numpy<1.24,
scvi-tools<1.0, lightning 2.2.x, python 3.9/3.10). This mirrors the env Ahlmann-Eltze et al.
used for CPA (their cpa_env3.yml):
    conda create -n cpa python=3.10 -y && conda activate cpa
    pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    pip install cpa-tools==0.8.8 scanpy
This script does NOT import GEARS: it reads the GEARS-processed h5ad directly
(data/norman/perturb_processed.h5ad, data_wessels_gears/wessels/perturb_processed.h5ad;
override with H5AD=...). adata.X is log-normalised, so we use recon_loss="gauss",
is_count_data=False (user choice; the CPA Norman tutorial itself uses counts + "nb").

Model/trainer hyper-parameters are copied from the official CPA Norman tutorial
(https://cpa-tools.readthedocs.io/en/latest/tutorials/Norman.html) except recon_loss.

Per fold:
  training cells = control + all singles + training pairs (90%); validation = cells of a
  random 10% of training pairs (seeded); "ood" rows = control cells relabelled with each
  test pair / test-pair single gene (counterfactual inputs, never trained on).
  Condition labels follow the tutorial: "A+B" with dose "1.0+1.0", single "A" dose "1.0".
  Predictions = model.predict() on the ood rows, averaged to pseudobulk (full gene space).
Env: DATASET, SPLIT, FOLD, SEED (default 8206 = tutorial seed), MAX_EPOCHS=2000,
     N_CTRL_PRED=500 (control cells per predicted condition), BATCH=2048
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
DS = os.environ.get("DATASET", "norman").lower()
SPLIT = os.environ.get("SPLIT", "rand").lower()
FOLD = os.environ.get("FOLD")
SEED = int(os.environ.get("SEED", "8206"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "2000"))
N_CTRL_PRED = int(os.environ.get("N_CTRL_PRED", "500"))
BATCH = int(os.environ.get("BATCH", "2048"))
assert DS in ("norman", "wessels") and SPLIT in ("rand", "gd")
H5AD = os.environ.get("H5AD", "data/norman/perturb_processed.h5ad" if DS == "norman"
                      else "data_wessels_gears/wessels/perturb_processed.h5ad")

MODEL_PARAMS = {  # CPA Norman tutorial, recon_loss changed nb -> gauss (log data)
    "n_latent": 32, "recon_loss": "gauss", "doser_type": "linear",
    "n_hidden_encoder": 256, "n_layers_encoder": 4, "n_hidden_decoder": 256, "n_layers_decoder": 2,
    "use_batch_norm_encoder": True, "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False, "use_layer_norm_decoder": False,
    "dropout_rate_encoder": 0.2, "dropout_rate_decoder": 0.0, "variational": False, "seed": SEED}
TRAINER_PARAMS = {
    "n_epochs_kl_warmup": None, "n_epochs_adv_warmup": 50, "n_epochs_mixup_warmup": 10,
    "n_epochs_pretrain_ae": 10, "mixup_alpha": 0.1, "lr": 0.0001, "wd": 3.2170178270865573e-06,
    "adv_steps": 3, "reg_adv": 10.0, "pen_adv": 20.0, "adv_lr": 0.0001, "adv_wd": 7.051355554517135e-06,
    "n_layers_adv": 2, "n_hidden_adv": 128, "use_batch_norm_adv": True, "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3, "step_size_lr": 25, "do_clip_grad": False, "adv_loss": "cce",
    "gradient_clip_value": 5.0}


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


# ----------------------------------------------------------------- main
def main():
    import anndata as ad
    import scipy.sparse as sp
    import torch
    import cpa

    pb, fold_of, K = load_bench(DS, SPLIT)
    full = ad.read_h5ad(H5AD)
    full.obs["condition"] = full.obs["condition"].astype(str)
    pair_c, single_c = condition_maps(full.obs["condition"].unique())
    pos = {g: i for i, g in enumerate(full.var["gene_name"].astype(str))}
    gidx = np.array([pos[g] for g in pb["gene_names"].astype(str)])
    ctrl_idx = np.where(full.obs["condition"].values == "ctrl")[0]

    folds = [int(FOLD)] if FOLD is not None else list(range(K))
    for f in folds:
        np.random.seed(SEED + f); torch.manual_seed(SEED + f)
        tr, va, te = split_pairs(fold_of, f, SEED)
        cond = lambda idx: [pair_c[frozenset(pb["pair_genes"][m])] for m in idx
                            if frozenset(pb["pair_genes"][m]) in pair_c]
        split_of = {"ctrl": "train"}
        split_of.update({c: "train" for c in single_c.values()})
        split_of.update({c: "train" for c in cond(tr)})
        split_of.update({c: "valid" for c in cond(va)})
        obs_c = full.obs["condition"].values
        keep = np.where(np.isin(obs_c, list(split_of)))[0]
        real = full[keep].copy()
        real.obs = real.obs[["condition"]].copy()
        real.obs["split"] = [split_of[c] for c in real.obs["condition"]]

        # counterfactual rows: control cells relabelled with the condition to predict
        test_pairs = [tuple(sorted(pb["pair_genes"][m])) for m in te]
        test_genes = sorted({g for p in test_pairs for g in p})
        targets = [f"{a}+{b}" for a, b in test_pairs] + test_genes
        rng = np.random.default_rng([SEED, f, 7])
        blocks = []
        for t in targets:
            ci = rng.choice(ctrl_idx, size=min(N_CTRL_PRED, len(ctrl_idx)), replace=False)
            b = full[ci].copy(); b.obs = b.obs[["condition"]].copy()
            b.obs["condition"] = "pred:" + t      # placeholder, harmonised below
            b.obs["split"] = "ood"
            blocks.append(b)
        a = ad.concat([real] + blocks, join="inner", index_unique="-")
        a.var = full.var.copy()
        if not sp.issparse(a.X):
            a.X = sp.csr_matrix(a.X)
        # CPA labels: drop "ctrl" from single conditions, "A+B" for pairs
        harm = []
        for c in a.obs["condition"].astype(str):
            if c.startswith("pred:"):
                harm.append(c[5:])
            elif c == "ctrl":
                harm.append("ctrl")
            else:
                harm.append("+".join(x for x in c.split("+") if x != "ctrl"))
        a.obs["cond_harm"] = harm
        a.obs["dose_value"] = ["1.0+1.0" if "+" in h else "1.0" for h in harm]

        cpa.CPA.pert_encoder = None      # class-level caches must be reset between folds
        cpa.CPA.covars_encoder = None
        cpa.CPA.setup_anndata(a, perturbation_key="cond_harm", control_group="ctrl",
                              dosage_key="dose_value", categorical_covariate_keys=[],
                              is_count_data=False, max_comb_len=2)
        model = cpa.CPA(adata=a, split_key="split", train_split="train",
                        valid_split="valid", test_split="ood", **MODEL_PARAMS)
        model.train(max_epochs=MAX_EPOCHS, use_gpu=True, batch_size=BATCH,
                    plan_kwargs=TRAINER_PARAMS, early_stopping_patience=5,
                    check_val_every_n_epoch=5, save_path=f"cpa_ckpt/{DS}_{SPLIT}_f{f}")

        model.predict(a, batch_size=BATCH)        # writes a.obsm["CPA_pred"] for ALL rows
        P = np.asarray(a.obsm["CPA_pred"])
        h = np.asarray(harm); is_ood = a.obs["split"].values == "ood"
        mean_of = lambda t: P[is_ood & (h == t)].mean(0)[gidx]
        combo_pred = np.stack([mean_of(f"{x}+{y}") for x, y in test_pairs]) if test_pairs else \
            np.zeros((0, len(gidx)))
        single_pred = np.stack([mean_of(g) for g in test_genes]) if test_genes else np.zeros((0, len(gidx)))
        out = out_name("cpa", DS, SPLIT, f)
        # name pairs like the GEARS condition string when that pair exists in the h5ad
        names = [pair_c.get(frozenset((x, y)), f"{x}+{y}") for x, y in test_pairs]
        np.savez(out, combo_names=np.array(names, dtype=str),
                 combo_pred=combo_pred.astype(np.float32), combo_idx=np.asarray(te, dtype=int),
                 single_names=np.array(test_genes, dtype=str),
                 single_pred=single_pred.astype(np.float32), fold=f, seed=SEED)
        print(f"[fold {f}] wrote {out}: {len(test_pairs)} pairs; "
              f"{sum(1 for m in te if frozenset(pb['pair_genes'][m]) not in pair_c)} test pairs had no cells")
        del model, a; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
