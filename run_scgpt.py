"""
run_scgpt.py -- scGPT perturbation fine-tuning (official Tutorial_Perturbation) on our folds.

MANUAL STEP (if auto-download fails): put the pretrained "whole-human" checkpoint folder
(args.json, best_model.pt, vocab.json) at ./scgpt_ckpt/scGPT_human  (or SCGPT_CKPT=...).
Official source (README of bowang-lab/scGPT): Google Drive folder
  https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y
The script tries `gdown.download_folder` on that URL first (Drive quotas often block it).

Install (SEPARATE env: scGPT pins cell-gears<0.0.3, which conflicts with GEARS 0.1.2;
this script does NOT import GEARS, it reads the GEARS-processed h5ad directly):
    conda create -n scgpt python=3.10 -y && conda activate scgpt
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    pip install "scgpt @ git+https://github.com/bowang-lab/scGPT@cebd6fae655b9c585a4807daa3ac31bb764f06b4"
        # (0.2.5 dev; torchtext-free vocab).  PyPI scgpt==0.2.4 needs torchtext -> torch<=2.3.
    pip install gdown scanpy "numpy<2"
    optional: pip install flash-attn==2.7.4.post1 --no-build-isolation  (then SCGPT_FLASH=1)

Faithful to the tutorial: GEARS-style training pairs (each perturbed cell paired with a
random control cell as input; control cells map to themselves), pert flags on the target
genes, include_zero_gene="all", max_seq_len=1536 random gene subsample per batch during
training, lr 1e-4, batch 64, StepLR(1, 0.9), 15 epochs, early stop 10, AMP, load only
encoder/value_encoder/transformer_encoder weights, best epoch by validation Pearson.
Adaptations: (i) our folds; validation = random 10% of training pairs (seeded);
(ii) perturbed genes not among the measured genes are appended as extra tokens
(value 0, flag 1, excluded from the loss) instead of being dropped.
Env: DATASET, SPLIT, FOLD, SEED=1, EPOCHS=15, POOL=300 (ctrl cells per prediction),
     EVAL_BS=16, FORCE_PERT=0 (1 = always keep perturbed genes in the 1536 subsample)
"""
import os, json, copy, time, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
DS = os.environ.get("DATASET", "norman").lower()
SPLIT = os.environ.get("SPLIT", "rand").lower()
FOLD = os.environ.get("FOLD")
SEED = int(os.environ.get("SEED", "1"))
EPOCHS = int(os.environ.get("EPOCHS", "15"))
POOL = int(os.environ.get("POOL", "300"))
EVAL_BS = int(os.environ.get("EVAL_BS", "16"))
FORCE_PERT = os.environ.get("FORCE_PERT", "0") == "1"
USE_FLASH = os.environ.get("SCGPT_FLASH", "0") == "1"
CKPT = os.environ.get("SCGPT_CKPT", "scgpt_ckpt/scGPT_human")
GDRIVE = "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
H5AD = "data/norman/perturb_processed.h5ad" if DS == "norman" else "data_wessels_gears/wessels/perturb_processed.h5ad"
LR, BS, MAX_SEQ, EARLY_STOP = 1e-4, 64, 1536, 10
assert DS in ("norman", "wessels") and SPLIT in ("rand", "gd")


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


def split_pairs(fold_of, f, seed, frac=0.1):
    test = np.where(fold_of == f)[0]
    train = np.where(fold_of != f)[0]
    rng = np.random.default_rng([seed, f])
    val = np.sort(rng.choice(train, max(1, int(round(frac * len(train)))), replace=False))
    return np.setdiff1d(train, val), val, test


def ensure_ckpt():
    need = ["args.json", "best_model.pt", "vocab.json"]
    if all(os.path.exists(os.path.join(CKPT, n)) for n in need):
        return
    try:
        import gdown
        os.makedirs(CKPT, exist_ok=True)
        gdown.download_folder(url=GDRIVE, output=CKPT, quiet=False)
    except Exception as e:
        print("gdown failed:", repr(e))
    for root, _, files in os.walk(CKPT):           # gdown may nest a sub-folder
        if all(n in files for n in need) and root != CKPT:
            for n in need:
                os.replace(os.path.join(root, n), os.path.join(CKPT, n))
    if not all(os.path.exists(os.path.join(CKPT, n)) for n in need):
        raise SystemExit(f"MANUAL STEP: download {GDRIVE} (args.json, best_model.pt, vocab.json) into {CKPT}")


def main():
    import torch, anndata as ad
    from scgpt.model import TransformerGenerator
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import map_raw_id_to_vocab_id
    dev = torch.device("cuda")
    ensure_ckpt()
    pb, fold_of, K = load_bench(DS, SPLIT)
    adata = ad.read_h5ad(H5AD)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32)
    cond = adata.obs["condition"].astype(str).values
    genes = list(adata.var["gene_name"].astype(str)); R = len(genes)
    pert_genes = sorted({g for c in set(cond) for g in c.split("+") if g != "ctrl"} |
                        {g for p in pb["pair_genes"] for g in p})
    genes_aug = genes + [g for g in pert_genes if g not in set(genes)]   # extra tokens
    col = {g: i for i, g in enumerate(genes_aug)}; N = len(genes_aug)
    gidx = np.array([col[g] for g in pb["gene_names"].astype(str)])
    key = lambda c: frozenset(g for g in c.split("+") if g != "ctrl")
    cells_of = {}
    for i, c in enumerate(cond):
        cells_of.setdefault(key(c), []).append(i)
    ctrl = np.array(cells_of[frozenset()])

    vocab = GeneVocab.from_file(os.path.join(CKPT, "vocab.json"))
    for s in ["<pad>", "<cls>", "<eoc>"]:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes_aug], dtype=int)
    print(f"{(gene_ids != vocab['<pad>']).sum()}/{N} genes in scGPT vocab")
    cfg = json.load(open(os.path.join(CKPT, "args.json")))
    sd_all = torch.load(os.path.join(CKPT, "best_model.pt"), map_location="cpu")

    def make_model():
        m = TransformerGenerator(len(vocab), cfg["embsize"], cfg["nheads"], cfg["d_hid"], cfg["nlayers"],
                                 nlayers_cls=cfg["n_layers_cls"], n_cls=1, vocab=vocab, dropout=0.0,
                                 pad_token="<pad>", pad_value=0, pert_pad_id=0,
                                 use_fast_transformer=USE_FLASH)
        sd = {k: v for k, v in sd_all.items()
              if k.startswith(("encoder", "value_encoder", "transformer_encoder"))}
        if not USE_FLASH:   # flash-attn checkpoint -> torch nn.MultiheadAttention names
            sd = {k.replace("self_attn._impl.", "self_attn.").replace("self_attn.Wqkv.", "self_attn.in_proj_"): v
                  for k, v in sd.items()}
        msd = m.state_dict()
        ok = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        print(f"loaded {len(ok)}/{len(sd)} pretrained tensors")
        msd.update(ok); m.load_state_dict(msd)
        return m.to(dev)

    def forward(m, xin, flags, ids):
        mapped = map_raw_id_to_vocab_id(ids, gene_ids).repeat(xin.shape[0], 1)
        with torch.cuda.amp.autocast(enabled=True):
            o = m(mapped, xin[:, ids], flags[:, ids], src_key_padding_mask=torch.zeros_like(
                xin[:, ids], dtype=torch.bool), CLS=False, CCE=False, MVC=False, ECS=False)
        return o["mlm_output"].float()

    def batch_tensors(inp_idx, perts):
        xin = torch.zeros(len(inp_idx), N, device=dev); xin[:, :R] = torch.from_numpy(X[inp_idx]).to(dev)
        fl = torch.zeros(len(inp_idx), N, dtype=torch.long, device=dev)
        for r, p in enumerate(perts):
            for g in p:
                fl[r, col[g]] = 1
        return xin, fl

    def predict(m, pert, rng):
        m.eval(); idx = rng.choice(ctrl, size=min(POOL, len(ctrl)), replace=False); out = []
        allids = torch.arange(N, device=dev)
        with torch.no_grad():
            for s in range(0, len(idx), EVAL_BS):
                xin, fl = batch_tensors(idx[s:s + EVAL_BS], [pert] * len(idx[s:s + EVAL_BS]))
                out.append(forward(m, xin, fl, allids)[:, :R].cpu().numpy())
        return np.concatenate(out).mean(0)

    folds = [int(FOLD)] if FOLD is not None else list(range(K))
    for f in folds:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        tr, va, te = split_pairs(fold_of, f, SEED)
        train_keys = [k for k in cells_of if len(k) <= 1] + \
                     [frozenset(pb["pair_genes"][m]) for m in tr if frozenset(pb["pair_genes"][m]) in cells_of]
        val_keys = [frozenset(pb["pair_genes"][m]) for m in va if frozenset(pb["pair_genes"][m]) in cells_of]
        tgt = np.concatenate([cells_of[k] for k in train_keys])
        tgt_pert = {i: tuple(sorted(key(cond[i]))) for i in tgt}
        model = make_model()
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        rng = np.random.default_rng([SEED, f])
        best, best_state, patience, hist = -np.inf, None, 0, []
        for ep in range(1, EPOCHS + 1):
            model.train(); t0 = time.time(); order = rng.permutation(tgt); losses = []
            for s in range(0, len(order) - BS + 1, BS):            # drop_last like GEARS loader
                b = order[s:s + BS]; perts = [tgt_pert[i] for i in b]
                inp = np.array([i if len(p) == 0 else rng.choice(ctrl) for i, p in zip(b, perts)])
                xin, fl = batch_tensors(inp, perts)
                y = torch.zeros(len(b), N, device=dev); y[:, :R] = torch.from_numpy(X[b]).to(dev)
                ids = torch.arange(N, device=dev)
                if N > MAX_SEQ:
                    ids = torch.randperm(N, device=dev)[:MAX_SEQ]
                    if FORCE_PERT:
                        pc = torch.tensor(sorted({col[g] for p in perts for g in p}), dtype=torch.long, device=dev)
                        ids = torch.unique(torch.cat([pc, ids]))[:MAX_SEQ] if len(pc) else ids
                pred = forward(model, xin, fl, ids)
                real = (ids < R).float().unsqueeze(0)               # extra pert tokens not in loss
                loss = (((pred - y[:, ids]) ** 2) * real).sum() / (real.sum() * len(b))
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
                scaler.step(opt); scaler.update(); losses.append(loss.item())
            # validation: Pearson of mean prediction vs mean truth per val pair (tutorial "pearson")
            prs = [np.corrcoef(predict(model, tuple(sorted(k)), np.random.default_rng(0))[gidx[gidx < R]],
                               X[cells_of[k]].mean(0)[gidx[gidx < R]])[0, 1] for k in val_keys]
            score = float(np.nanmean(prs)) if prs else -float(np.mean(losses))
            hist.append((ep, float(np.mean(losses)), score))
            print(f"[fold {f}] epoch {ep} loss {np.mean(losses):.4f} val_pearson {score:.4f} ({time.time()-t0:.0f}s)")
            if score > best:
                best, best_state, patience = score, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
                if patience >= EARLY_STOP:
                    break
            sched.step()
        model.load_state_dict(best_state)

        prng = np.random.default_rng([SEED, f, 99])
        names, preds = [], []
        for m in te:
            a, b = pb["pair_genes"][m]
            names.append(f"{a}+{b}"); preds.append(predict(model, (a, b), prng)[gidx])
        s_names = sorted({g for p in (pb["pair_genes"][m] for m in te) for g in p})
        s_pred = [predict(model, (g,), prng)[gidx] for g in s_names]
        G = len(gidx); out = out_name("scgpt", DS, SPLIT, f)
        np.savez(out, combo_names=np.array(names, dtype=str), combo_pred=np.array(preds, np.float32).reshape(-1, G),
                 combo_idx=np.asarray(te, int), single_names=np.array(s_names, dtype=str),
                 single_pred=np.array(s_pred, np.float32).reshape(-1, G),
                 history=np.array(hist, np.float32), fold=f, seed=SEED)
        print(f"[fold {f}] wrote {out}")
        del model, best_state; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
