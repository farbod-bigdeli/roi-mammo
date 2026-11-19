"""Addendum B (ANALYSIS_PLAN.md): placement-matched control for the ROI arm.

Compares, for each dataset and backbone (EfficientNet-B0, ResNet-50),
    ROI crop              roi[...,<chosen bank>]   (saliency-ranked boxes)
    central random box    roi[...,random]          (same windows, sizes and placement rules, random ranking)
    Random crop           randcrop[...,<chosen bank>]
    Standard              none
with three comparisons each (12 in total):
    ROI crop - central random box      value of the saliency with placement held fixed (B-primary)
    central random box - Random crop   value of the placement rules
    central random box - Standard
Statistics as in reanalysis.py: seed-averaged paired AUC difference, patient bootstrap with 20,000
replicates, two-level seed+patient bootstrap (5,000), Holm within the 12, bootstrap TOST for
equivalence within +-0.01 (Holm within the 12), and the interpretation rule fixed in the plan.

    python -m roimammo.controlarm --runs runs --results results --out results/control
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import load_runs, pool_oof, unit_preds
from .metrics import holm
from .reanalysis import one

MODELS = ("effb0/gap/image", "resnet50/gap/image")


def labels(chosen: dict):
    tail = f"p={chosen['p_crop']:g},j={chosen['jitter']:g},K={chosen['roi_k']}"
    return {"std": "none", "rand": f"randcrop[{tail},{chosen['bank']}]", "roi": f"roi[{tail},{chosen['bank']}]",
            "crb": f"roi[{tail},random]"}


COMPS = (("roi", "crb", "ROI crop - central random box"),
         ("crb", "rand", "central random box - Random crop"),
         ("crb", "std", "central random box - Standard"))


def interpret(rows: pd.DataFrame) -> str:
    sal = rows[rows.comparison == COMPS[0][2]].iloc[0]
    plc = rows[rows.comparison == COMPS[1][2]].iloc[0]
    if sal.ci95_lo > 0:
        return "saliency"
    if plc.ci95_lo > 0 and sal.ci95_lo <= 0 <= sal.ci95_hi:
        return "placement"
    return "inconclusive"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/control"))
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--n-hier", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    lab = labels(json.load(open(a.results / "chosen_miniddsm.json")))
    runs = pd.DataFrame(pool_oof(load_runs(a.runs, "final", "control")))
    runs = runs[runs.model.isin(MODELS) & runs.arm_label.isin(lab.values())]

    jobs, missing = [], []
    for (ds, model), g in runs.groupby(["dataset", "model"]):
        by = {k: {r["seed"]: r for r in gg.to_dict("records")} for k, gg in g.groupby("arm_label")}
        for x, r, name in COMPS:
            lx, lr = lab[x], lab[r]
            if lx not in by or lr not in by:
                missing.append((ds, model, name))
                continue
            seeds = sorted(set(by[lx]) & set(by[lr]))
            px = {s: unit_preds(by[lx][s], "breast") for s in seeds}
            pr = {s: unit_preds(by[lr][s], "breast") for s in seeds}
            base = px[seeds[0]][["key", "patient_id", "label"]].sort_values(["patient_id", "key"]).reset_index(drop=True)
            y = base.label.to_numpy().astype(int)
            X = [px[s].set_index("key").loc[base.key, "prob"].to_numpy() for s in seeds]
            R = [pr[s].set_index("key").loc[base.key, "prob"].to_numpy() for s in seeds]
            key = dict(dataset=ds, model=model, comparison=name, arm=lx, reference=lr, role=name,
                       n_units=len(base), n_pos=int(y.sum()))
            jobs.append((key, y, X, R, base.patient_id.to_numpy(), None, a.n_boot, a.n_hier))
    if missing:
        print("[warn] comparisons without runs:", missing)
    with Pool(a.workers) as pool:
        res = pd.DataFrame(pool.map(one, jobs))
    res["p_holm"] = holm(res.p_boot)
    res["p_tost_holm"] = holm(res.p_tost)
    res["equivalent_holm"] = res.p_tost_holm < 0.05
    res.to_csv(a.out / "control_breast.csv", index=False)
    verdict = [{"dataset": ds, "model": m, "interpretation": interpret(g)}
               for (ds, m), g in res.groupby(["dataset", "model"]) if len(g) == 3]
    pd.DataFrame(verdict).to_csv(a.out / "control_interpretation.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res[["dataset", "model", "comparison", "seeds", "delta", "ci95_lo", "ci95_hi", "hier_ci95_lo",
                   "hier_ci95_hi", "per_seed", "p_holm", "p_tost_holm"]].to_string(
            index=False, float_format=lambda v: f"{v:+.4f}"))
        print()
        print(pd.DataFrame(verdict).to_string(index=False))
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
