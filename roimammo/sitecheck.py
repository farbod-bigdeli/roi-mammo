"""Within-site AUC on the test set (pre-specified in ANALYSIS_PLAN.md; script added after the final runs).

For every final model and arm on a dataset with more than one site, the breast-level AUC is computed
separately within each site (mean over seeds), so that the site/label association cannot contribute.
Also reports a site-only baseline: each test breast is scored with the positive-breast rate of its
site among development breasts, which is what a network that only recognised the digitiser could do.

    python -m roimammo.sitecheck --dataset miniddsm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import load_runs, pool_oof, unit_preds
from .metrics import auc




def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--dataset", default="miniddsm")
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)

    ddir = a.data_root / a.dataset
    index = pd.read_csv(ddir / "index.csv")
    split = pd.read_csv(ddir / "split.csv")
    index["site"] = index["site"].astype(object).where(index["site"].notna(), "unknown").astype(str)
    site_of = index.groupby("patient_id")["site"].first()
    if site_of.nunique() < 2:
        raise SystemExit("dataset has a single site")

    # site-only baseline: development breast-level positive rate per site, applied to test breasts
    br = index.groupby("breast_id").agg(label=("label", "max"), patient_id=("patient_id", "first"),
                                        site=("site", "first")).reset_index()
    br["fold"] = br.patient_id.map(split.set_index("patient_id")["fold"])
    dev, test = br[br.fold >= 0], br[br.fold < 0]
    rate = dev.groupby("site").label.mean()
    base_auc = auc(test.label.to_numpy(), test.site.map(rate).to_numpy())
    counts = test.groupby("site").label.agg(n="size", n_pos="sum")
    print("Test breasts by site:\n" + counts.to_string())
    print(f"Development positive rate by site: {rate.round(3).to_dict()}")
    print(f"Site-only baseline, breast-level test AUC: {base_auc:.4f}\n")

    runs = [r for r in pool_oof(load_runs(a.runs, "final")) if r["dataset"] == a.dataset]
    rows = []
    for r in runs:
        u = unit_preds(r, "breast")
        u["site"] = u.patient_id.map(site_of)
        row = {"model": r["model"], "arm": r["arm_label"], "seed": r["seed"], "all": auc(u.label, u.prob)}
        for s, g in u.groupby("site"):
            row[s] = auc(g.label, g.prob) if g.label.nunique() == 2 else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    sites = sorted(c for c in df.columns if c not in ("model", "arm", "seed", "all"))
    summ = df.groupby(["model", "arm"])[["all"] + sites].mean()
    summ["mean_within_site"] = summ[sites].mean(axis=1)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / f"site_auc_runs_{a.dataset}.csv", index=False)
    summ.to_csv(a.out / f"site_auc_{a.dataset}.csv")
    with pd.option_context("display.width", 200):
        print(summ.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"[done] {a.out / f'site_auc_{a.dataset}.csv'}")


if __name__ == "__main__":
    main()
