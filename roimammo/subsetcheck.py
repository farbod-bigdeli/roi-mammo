"""CMMD subset check (added after the third review; no retraining, uses the saved predictions).

CMMD patients come in two subsets, D1 and D2 (patient-ID prefix). D2 is the subset with molecular
subtypes and is almost entirely malignant, so subset membership alone predicts the label. This
script reports
    * breasts and the share malignant per subset, and the AUC of the subset indicator alone;
    * for each arm, how well the model score separates D2 from D1 among malignant breasts only
      (about 0.5 if the model does not recognise the subset);
    * the paired arm comparisons of the main analysis, addendum A3 and addendum B, restricted to D1
      (where both classes are well represented), with the statistics of reanalysis.py.

    python -m roimammo.subsetcheck --runs runs --results results --out results/subsetcheck
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import load_runs, pool_oof, unit_preds
from .controlarm import MODELS, labels
from .metrics import auc
from .reanalysis import one

COMPS = (("roi", "std"), ("rand", "std"), ("roi", "rand"), ("roi", "crb"), ("crb", "rand"), ("crb", "std"))
NAMES = {"std": "Standard", "rand": "Random crop", "roi": "ROI crop", "crb": "central random box"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/subsetcheck"))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-hier", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(t=""):
        print(t, flush=True)
        lines.append(t)

    lab = labels(json.load(open(a.results / "chosen_miniddsm.json")))
    inv = {v: k for k, v in lab.items()}
    runs = pd.DataFrame(pool_oof(load_runs(a.runs, "final", "control")))
    runs = runs[(runs.dataset == "cmmd") & runs.model.isin(MODELS) & runs.arm_label.isin(lab.values())]

    desc, jobs = [], []
    for model, g in runs.groupby("model"):
        by = {inv[k]: {r["seed"]: r for r in gg.to_dict("records")} for k, gg in g.groupby("arm_label")}
        preds = {arm: {s: unit_preds(r, "breast") for s, r in d.items()} for arm, d in by.items()}
        if not desc:
            u = preds["std"][min(preds["std"])]
            sub = u.patient_id.str[:2]
            say("CMMD breasts by subset (breast level):")
            for s in sorted(sub.unique()):
                m = sub == s
                say(f"  {s}: {int(m.sum())} breasts, {100 * u.label[m].mean():.1f}% malignant")
            say(f"  AUC of the subset indicator (D2) alone: {auc(u.label.to_numpy(), (sub == 'D2').astype(float).to_numpy()):.3f}")
            say("\nAmong malignant breasts, AUC of the model score for D2 vs D1 (0.5 = subset not recognised):")
        for arm, d in preds.items():
            v = []
            for s, u in d.items():
                m = u.label == 1
                v.append(auc((u.patient_id[m].str[:2] == "D2").astype(int).to_numpy(), u.prob[m].to_numpy()))
            desc.append(dict(model=model, arm=NAMES[arm], seeds=len(v), auc_d2_among_malignant=np.mean(v)))
            say(f"  {model:22s} {NAMES[arm]:20s} seeds {len(v)}: {np.mean(v):.3f}")
        for x, r in COMPS:
            if x not in preds or r not in preds:
                continue
            seeds = sorted(set(preds[x]) & set(preds[r]))
            base = preds[x][seeds[0]][["key", "patient_id", "label"]]
            base = base[base.patient_id.str.startswith("D1")].sort_values(["patient_id", "key"]).reset_index(drop=True)
            y = base.label.to_numpy().astype(int)
            X = [preds[x][s].set_index("key").loc[base.key, "prob"].to_numpy() for s in seeds]
            R = [preds[r][s].set_index("key").loc[base.key, "prob"].to_numpy() for s in seeds]
            key = dict(dataset="cmmd-D1", model=model, comparison=f"{NAMES[x]} - {NAMES[r]}",
                       n_units=len(base), n_pos=int(y.sum()))
            jobs.append((key, y, X, R, base.patient_id.to_numpy(), None, a.n_boot, a.n_hier))
    pd.DataFrame(desc).to_csv(a.out / "subset_recognition.csv", index=False)
    with Pool(a.workers) as pool:
        res = pd.DataFrame(pool.map(one, jobs))
    res.to_csv(a.out / "subset_d1_breast.csv", index=False)
    say("\nArm comparisons restricted to D1 (breast-level AUC difference, seed-averaged):")
    cols = ["model", "comparison", "n_units", "n_pos", "seeds", "delta", "ci95_lo", "ci95_hi", "per_seed"]
    cols += [c for c in ("hier_ci95_lo", "hier_ci95_hi") if c in res]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        say(res[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    (a.out / "subsetcheck.txt").write_text("\n".join(lines) + "\n")
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
