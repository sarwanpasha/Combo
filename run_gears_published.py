"""
run_gears_published.py -- reproduce GEARS' published Norman evaluation (simulation split).

Install: same env as GEARS (torch 2.5.1+cu121, torch_geometric, cell-gears==0.1.2).

For each seed in SEEDS (default 1..5, as in the GEARS paper):
    pd_.prepare_split(split="simulation", seed=seed)        # GEARS' own split + subgroups
    pd_.get_dataloader(batch_size=32, test_batch_size=128)   # (0.1.2 actually uses batch_size for all loaders)
    g = GEARS(pd_); g.model_initialize(hidden_size=64); g.train(epochs=20)
Then evaluate the best model on the test loader with GEARS' own utilities
(gears.inference.evaluate / compute_metrics / deeper_analysis / non_dropout_analysis),
and aggregate per test subgroup (combo_seen0, combo_seen1, combo_seen2, unseen_single) exactly
like GEARS.train() does internally (it only prints them; here they are saved).

Metrics closest to the paper figures:
  mse_de                         MSE on top-20 DE genes (cell-level DE idx, compute_metrics)
  mse_top20_de_non_dropout       MSE on top-20 non-dropout DE genes (non_dropout_analysis)
  pearson_delta                  Pearson of (pred-ctrl) vs (truth-ctrl), all genes (deeper_analysis)
  pearson_delta_top20_de_non_dropout / pearson_delta_de
Output: gears_published_norman.json (per seed, per subgroup, all metrics + seed means).
Env: SEEDS="1,2,3,4,5"  EPOCHS=20  DEVICE=cuda
"""
import os, json, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3,4,5").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "20"))
DEVICE = os.environ.get("DEVICE", "cuda")
OUT = os.environ.get("OUT", "gears_published_norman.json")
KEY = ["mse", "mse_de", "pearson", "pearson_de", "pearson_delta", "pearson_delta_de",
       "mse_top20_de_non_dropout", "pearson_delta_top20_de_non_dropout",
       "frac_opposite_direction_top20_non_dropout", "frac_correct_direction_20_nonzero"]


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    import torch
    from gears import PertData, GEARS
    from gears.inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis

    pd_ = PertData("./data")
    pd_.load(data_name="norman")
    results = {}
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        pd_.prepare_split(split="simulation", seed=seed)
        pd_.get_dataloader(batch_size=32, test_batch_size=128)
        g = GEARS(pd_, device=DEVICE)
        g.model_initialize(hidden_size=64)
        g.train(epochs=EPOCHS, lr=1e-3)   # also prints GEARS' own subgroup analysis

        test_res = evaluate(pd_.dataloader["test_loader"], g.best_model,
                            g.config["uncertainty"], g.device)
        test_metrics, test_pert_res = compute_metrics(test_res)
        out = deeper_analysis(pd_.adata, test_res)
        out_nd = non_dropout_analysis(pd_.adata, test_res)

        # merge all per-perturbation metrics
        per_pert = {}
        for p in test_pert_res:
            d = dict(test_pert_res[p]); d.update(out.get(p, {})); d.update(out_nd.get(p, {}))
            per_pert[p] = {k: to_float(v) for k, v in d.items()}

        sub = {}
        for name, plist in pd_.subgroup["test_subgroup"].items():
            plist = [p for p in plist if p in per_pert]
            agg = {}
            for m in sorted({k for p in plist for k in per_pert[p]}):
                vals = [per_pert[p][m] for p in plist if per_pert[p].get(m) is not None]
                agg[m] = float(np.nanmean(vals)) if vals else None
            agg["n_perts"] = len(plist)
            sub[name] = agg
        results[str(seed)] = {"overall": {k: to_float(v) for k, v in test_metrics.items()},
                              "subgroups": sub, "per_pert": per_pert}

        print(f"\n=== seed {seed} ===")
        for name, agg in sub.items():
            print(name, {k: (round(agg[k], 4) if agg.get(k) is not None else None)
                         for k in KEY + ["n_perts"] if k in agg})
        del g; torch.cuda.empty_cache()
        with open(OUT, "w") as fh:
            json.dump(results, fh, indent=1)

    # mean +- sd across seeds per subgroup for the headline metrics
    summary = {}
    names = sorted({n for r in results.values() for n in r["subgroups"]})
    for n in names:
        summary[n] = {}
        for m in KEY:
            v = [r["subgroups"][n].get(m) for r in results.values() if n in r["subgroups"]]
            v = [x for x in v if x is not None]
            if v:
                summary[n][m] = {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n_seeds": len(v)}
    results["summary"] = summary
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=1)
    print("\n=== summary across seeds ===")
    for n, d in summary.items():
        print(n, {m: f"{x['mean']:.4f}+-{x['sd']:.4f}" for m, x in d.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
