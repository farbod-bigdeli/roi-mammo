"""Within-fold and per-seed contrasts for out-of-fold (CMMD) results (added after peer review).

The pre-specified CMMD analysis pools the out-of-fold predictions of the five fold models and
computes one AUC. That AUC also ranks positive and negative breasts that were scored by different
models, so shifts in the score scale between folds can move it. This script keeps the pooled
analysis and adds, for every comparison of the final analysis:
    within-fold AUC   the AUC computed only over positive-negative pairs from the same fold
                      (the per-fold AUCs weighted by n_pos * n_neg), with a 95 % bootstrap interval
                      that resamples patients within folds;
    per-fold          the AUC difference in each fold (mean over seeds) and its SD across folds;
    per-seed          the pooled and within-fold differences for each seed separately.
The intervals are conditional on the fitted models (as in the main analysis); the per-seed
differences show the variation from retraining.

    python -m roimammo.foldcheck --dataset cmmd --study all --out results/foldcheck
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import comparisons_for, load_runs, pool_oof, pretty_arm, pretty_model, unit_preds
from .metrics import auc


def within_fold_auc(y, s, fold) -> float:
    num = den = 0.0
    for f in np.unique(fold):
        m = fold == f
        n1 = y[m].sum()
        n0 = m.sum() - n1
        if n1 and n0:
            num += auc(y[m], s[m]) * n1 * n0
            den += n1 * n0
    return num / den if den else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--dataset", default="cmmd")
    ap.add_argument("--study", default="all", choices=["main", "all"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("results/foldcheck"))
    a = ap.parse_args(argv)

    sp = pd.read_csv(a.data_root / a.dataset / "split.csv")
    fold_of = sp.set_index("patient_id")["fold"]
    runs = [r for r in pool_oof(load_runs(a.runs, "final", a.study)) if r["dataset"] == a.dataset
            and r.get("oof_fold") is not None]
    if not runs:
        raise SystemExit(f"no out-of-fold final runs for {a.dataset}")
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, seed_rows, fold_rows = [], [], []
    for model, grp in pd.DataFrame(runs).groupby("model"):
        by_arm = {k: g.sort_values("seed").to_dict("records") for k, g in grp.groupby("arm_label")}
        labels = set(by_arm)
        for arm_x, arm_ref in comparisons_for(sorted(by_arm)):
            rx = {r["seed"]: r for r in by_arm[arm_x]}
            rr = {r["seed"]: r for r in by_arm[arm_ref]}
            seeds = sorted(set(rx) & set(rr))
            px = {s: unit_preds(rx[s], "breast").set_index("key") for s in seeds}
            pr = {s: unit_preds(rr[s], "breast").set_index("key") for s in seeds}
            base = px[seeds[0]][["patient_id", "label"]].sort_index()
            keys = base.index
            y = base.label.to_numpy().astype(int)
            fold = base.patient_id.map(fold_of).to_numpy()
            SX = [px[s].loc[keys, "prob"].to_numpy() for s in seeds]
            SR = [pr[s].loc[keys, "prob"].to_numpy() for s in seeds]
            name = f"{pretty_arm(arm_x, labels)} vs {pretty_arm(arm_ref, labels)}"
            pooled = [auc(y, x) - auc(y, r) for x, r in zip(SX, SR)]
            within = [within_fold_auc(y, x, fold) - within_fold_auc(y, r, fold) for x, r in zip(SX, SR)]
            for s, dp, dw in zip(seeds, pooled, within):
                seed_rows.append({"model": pretty_model(model), "comparison": name, "seed": s,
                                  "delta_pooled": dp, "delta_within_fold": dw})
            per_fold = []
            for f in np.unique(fold):
                m = fold == f
                d = np.mean([auc(y[m], x[m]) - auc(y[m], r[m]) for x, r in zip(SX, SR)])
                per_fold.append(d)
                fold_rows.append({"model": pretty_model(model), "comparison": name, "fold": int(f), "delta": d,
                                  "n": int(m.sum()), "n_pos": int(y[m].sum())})
            # bootstrap: resample patients within each fold
            pats = base.patient_id.to_numpy()
            groups = []
            for f in np.unique(fold):
                pf = np.unique(pats[fold == f])
                groups.append([np.nonzero(pats == p)[0] for p in pf])
            boot = np.empty(a.n_boot)
            for b in range(a.n_boot):
                idx = np.concatenate([np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]) for g in groups])
                yy, ff = y[idx], fold[idx]
                boot[b] = np.mean([within_fold_auc(yy, x[idx], ff) - within_fold_auc(yy, r[idx], ff)
                                   for x, r in zip(SX, SR)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            lo90, hi90 = np.percentile(boot, [5, 95])
            rows.append({"model": pretty_model(model), "comparison": name, "seeds": len(seeds),
                         "delta_pooled": np.mean(pooled), "delta_within_fold": np.mean(within),
                         "within_ci95_lo": lo, "within_ci95_hi": hi, "within_ci90_lo": lo90, "within_ci90_hi": hi90,
                         "fold_delta_mean": np.mean(per_fold), "fold_delta_sd": np.std(per_fold, ddof=1),
                         "fold_delta_min": np.min(per_fold), "fold_delta_max": np.max(per_fold),
                         "seed_delta_min": np.min(pooled), "seed_delta_max": np.max(pooled)})
            print(f"[info] {pretty_model(model)}: {name} done", flush=True)
    s = pd.DataFrame(rows)
    s.to_csv(a.out / f"foldcheck_{a.dataset}.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(a.out / f"foldcheck_seeds_{a.dataset}.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(a.out / f"foldcheck_folds_{a.dataset}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(s.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        print()
        print(pd.DataFrame(seed_rows).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
