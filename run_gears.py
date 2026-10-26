# Train GEARS per fold on a custom split and save combo + single predictions.
# Usage: DATASET=norman|wessels SPLIT=rand|gd GEARS_EPOCHS=20 python run_gears.py
import os, numpy as np, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from gears import GEARS, PertData
DS = os.environ.get("DATASET", "norman"); SPLIT = os.environ.get("SPLIT", "rand")
EPOCHS = int(os.environ.get("GEARS_EPOCHS", "20"))
if DS == "norman":
    pb = np.load("pseudobulk.npz", allow_pickle=True)
    split_file = {"rand": "custom_split.npz", "gd": "gd_split.npz"}[SPLIT]
    prefix = {"rand": "gears_fold", "gd": "gears_gd_fold"}[SPLIT]
    to_gears = lambda c: c
    pd_ = PertData("./data"); pd_.load(data_name="norman")
else:
    pb = np.load("pseudobulk_wessels.npz", allow_pickle=True)
    split_file = {"rand": "custom_split_wessels.npz", "gd": "gd_split_wessels.npz"}[SPLIT]
    prefix = f"gears_wessels_{SPLIT}_fold"
    to_gears = lambda c: c.replace("_", "+")
    pd_ = PertData("./data_wessels_gears"); pd_.load(data_path="./data_wessels_gears/wessels")
G = pb["ctrl_mean"].shape[0]; combo_names = list(pb["combo_names"])
sp = np.load(split_file, allow_pickle=True); fold_of = sp["fold_of"]; K = int(sp["K"])
def n2g(c): return [g for g in c.split("+") if g != "ctrl"]
FOLDS = [int(os.environ["FOLD"])] if os.environ.get("FOLD") else range(K)
for f in FOLDS:
    ckpt = f"{prefix}{f}.npz"
    if os.path.exists(ckpt): print(f"fold {f}: exists, skip", flush=True); continue
    test_mask = fold_of == f
    if test_mask.sum() == 0: print(f"fold {f}: empty, skip", flush=True); continue
    test_combos = [to_gears(combo_names[k]) for k in range(len(combo_names)) if test_mask[k]]
    conds = list(pd_.adata.obs["condition"].unique()); tset = set(test_combos)
    train_c = [c for c in conds if c not in tset]; test_c = [c for c in conds if c in tset]
    pd_.set2conditions = {"train": train_c, "val": test_c[:1], "test": test_c}
    pd_.split = "custom"; pd_.seed = f
    pd_.get_dataloader(batch_size=64, test_batch_size=64)
    torch.manual_seed(f); np.random.seed(f)
    g = GEARS(pd_, device="cuda" if torch.cuda.is_available() else "cpu")
    g.model_initialize(hidden_size=64)
    g.train(epochs=EPOCHS)
    predable = [c for c in test_combos if all(x in g.pert_list for x in n2g(c))]
    cp_n, cp_v, sp_n, sp_v = [], [], [], []
    if predable:
        pdict = g.predict([n2g(c) for c in predable])
        for c in predable: cp_n.append(c); cp_v.append(pdict["_".join(n2g(c))])
        need = sorted({gg for c in predable for gg in n2g(c) if gg in g.pert_list})
        sdict = g.predict([[gg] for gg in need])
        for gg in need: sp_n.append(gg); sp_v.append(sdict[gg])
    np.savez_compressed(ckpt, combo_names=np.array(cp_n),
        combo_pred=np.vstack(cp_v).astype(np.float32) if cp_v else np.zeros((0, G), np.float32),
        single_names=np.array(sp_n, dtype=object),
        single_pred=np.vstack(sp_v).astype(np.float32) if sp_v else np.zeros((0, G), np.float32), fold=f)
    print(f"{DS} {SPLIT} fold {f}: saved {len(cp_n)} combos, {len(sp_n)} singles", flush=True)
print("GEARS DONE", DS, SPLIT, flush=True)
