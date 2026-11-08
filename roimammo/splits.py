"""Patient-level splits: a locked test set plus k development folds, and a shortcut check.

With --test-frac 0 there is no test set: all patients go into k folds and final models are
evaluated out-of-fold (each patient predicted once by a model that never saw them). Use this
for datasets on which nothing is tuned (CMMD in this study).

Every image of a patient lands in exactly one partition. Splits are stratified by the
patient label (any positive breast) and, where possible, by site (scanner/institution).

Usage:
    python -m roimammo.splits --data-dir data/miniddsm --test-frac 0.2 --folds 4 --seed 2026
Writes <data-dir>/split.csv (patient_id, split, fold) and <data-dir>/split_report.txt.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict


def make_splits(index: pd.DataFrame, test_frac: float, folds: int, seed: int) -> pd.DataFrame:
    pat = index.groupby("patient_id").agg(label=("label", "max"), site=("site", "first")).reset_index()
    strat = pat["label"].astype(str) + "|" + pat["site"].astype(str)
    counts = strat.value_counts()
    need = max(folds, round(1 / test_frac)) if test_frac > 0 else folds
    strat = strat.where(strat.map(counts) >= need, pat["label"].astype(str))
    pat["split"] = "dev"
    if test_frac > 0:  # locked test set; with test_frac = 0 every patient is in a fold (out-of-fold evaluation)
        n_test_splits = max(2, round(1 / test_frac))
        sgk = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=seed)
        _, test_idx = next(sgk.split(pat, strat, groups=pat["patient_id"]))
        pat.loc[test_idx, "split"] = "test"
    pat["fold"] = -1
    dev = pat[pat.split == "dev"]
    sgk = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + 1)
    for k, (_, vi) in enumerate(sgk.split(dev, strat[dev.index], groups=dev["patient_id"])):
        pat.loc[dev.index[vi], "fold"] = k
    return pat[["patient_id", "split", "fold"]]


def shortcut_check(index: pd.DataFrame, seed: int) -> str:
    """How well does the site alone predict the breast label? (5-fold, grouped by patient)"""
    br = index.groupby("breast_id").agg(label=("label", "max"), site=("site", "first"),
                                        patient_id=("patient_id", "first")).reset_index()
    lines = ["Site x breast label (counts):", pd.crosstab(br["site"], br["label"]).to_string(), ""]
    if br["site"].nunique() > 1 and br["label"].nunique() > 1:
        X = pd.get_dummies(br["site"]).to_numpy(float)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        p = cross_val_predict(LogisticRegression(max_iter=1000), X, br["label"], cv=cv,
                              groups=br["patient_id"], method="predict_proba")[:, 1]
        auc = roc_auc_score(br["label"], p)
        lines.append(f"Breast-level AUC of predicting the label from site alone: {auc:.3f}")
        lines.append("(0.5 = site carries no label information; report this value in the paper.)")
    else:
        lines.append("Only one site or one class: no site shortcut can be estimated.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true", help="overwrite an existing split.csv")
    a = ap.parse_args(argv)

    out = a.data_dir / "split.csv"
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists; splits are locked. Use --force only if you have not trained yet.")
    index = pd.read_csv(a.data_dir / "index.csv")
    index["site"] = index["site"].astype(object).where(index["site"].notna(), "unknown").astype(str)
    sp = make_splits(index, a.test_frac, a.folds, a.seed)
    sp.to_csv(out, index=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()

    m = index.merge(sp, on="patient_id")
    m["part"] = np.where(m.split == "test", "test", "fold" + m.fold.astype(str))
    rep = [f"split.csv sha256: {digest}", f"seed={a.seed} test_frac={a.test_frac} folds={a.folds}", ""]
    g = m.groupby("part").agg(patients=("patient_id", "nunique"), breasts=("breast_id", "nunique"),
                              images=("image_id", "count"), positive_image_rate=("label", "mean"))
    rep += ["Partitions:", g.to_string(), "", shortcut_check(index, a.seed)]
    text = "\n".join(rep)
    (a.data_dir / "split_report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
