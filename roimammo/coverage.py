"""Mechanism check: do label-free ROI boxes land on lesions more often than random boxes?

Uses lesion outlines (Mini-DDSM) for analysis only; they are never used in training.
For each image with a lesion mask and each bank variant, at K in {1, 3, 5, 10}:
    hit      = some box among the top K contains the lesion centroid
    coverage = fraction of lesion pixels inside the union of the top-K boxes
Baseline: boxes of the same sizes centred on random breast pixels (--repeats draws per image).
Confidence intervals come from a bootstrap over patients.

Usage:
    python -m roimammo.coverage --data-dir data/miniddsm --banks data/miniddsm/roi_bank_*.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .roibank import random_box_like

KS = (1, 3, 5, 10)


def _scores(boxes, les: np.ndarray, cy: float, cx: float):
    union = np.zeros_like(les)
    hit = False
    for b in boxes:
        union[b[1]:b[3], b[0]:b[2]] = True
        hit |= (b[0] <= cx < b[2]) and (b[1] <= cy < b[3])
    return float(hit), float((union & les).sum() / max(les.sum(), 1))


def evaluate(data_dir: Path, bank: dict, index: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for r in index.itertuples():
        les = cv2.imread(str(data_dir / "lesions" / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        tis = cv2.imread(str(data_dir / "tissue" / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        boxes = bank.get(r.image_id, [])
        if les is None or tis is None or not boxes:
            continue
        les, tis = les > 127, tis > 127
        ys, xs = np.nonzero(les)
        cy, cx = ys.mean(), xs.mean()
        coords = np.nonzero(tis)  # computed once per image (drawing boxes is otherwise the bottleneck)
        for k in KS:
            bk = boxes[:k]
            h, c = _scores(bk, les, cy, cx)
            rh, rc = [], []
            for _ in range(repeats):
                rb = [random_box_like(b[2] - b[0], b[3] - b[1], tis, rng, coords) for b in bk]
                a, b_ = _scores(rb, les, cy, cx)
                rh.append(a)
                rc.append(b_)
            rows.append({"image_id": r.image_id, "patient_id": r.patient_id, "label": r.label, "K": k,
                         "view": getattr(r, "view", "OTHER"),
                         "hit": h, "coverage": c, "hit_random": np.mean(rh), "coverage_random": np.mean(rc),
                         "lesion_frac_of_breast": les.sum() / max(tis.sum(), 1)})
    return pd.DataFrame(rows)


def boot_ci(df: pd.DataFrame, col: str, n: int = 2000, seed: int = 0):
    pats = df.patient_id.unique()
    g = df.groupby("patient_id")[col].agg(["sum", "count"]).loc[pats]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pats), (n, len(pats)))
    s, c = g["sum"].to_numpy()[idx].sum(1), g["count"].to_numpy()[idx].sum(1)
    return np.percentile(s / c, [2.5, 97.5])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--banks", required=True, type=Path, nargs="+")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--subset", choices=["all", "dev"], default="all",
                    help="dev = development patients only; use this for any design decision")
    a = ap.parse_args(argv)

    index = pd.read_csv(a.data_dir / "index.csv")
    index = index[index.has_lesion_mask.astype(bool)]
    if a.subset == "dev":
        sp = pd.read_csv(a.data_dir / "split.csv")
        index = index[index.patient_id.isin(sp.patient_id[sp.split == "dev"])]
    if index.empty:
        raise SystemExit("no lesion masks in this dataset; coverage needs Mini-DDSM outlines")
    summ = []
    print(f"[info] {len(index)} images with lesion masks, {len(a.banks)} banks", flush=True)
    for bp in a.banks:
        bank = json.load(open(bp))["boxes"]
        variant = json.load(open(bp))["meta"].get("variant", bp.stem)
        print(f"[info] bank {variant} ...", flush=True)
        df = evaluate(a.data_dir, bank, index, a.repeats, a.seed)
        df.insert(0, "variant", variant)
        df.to_csv(a.data_dir / f"coverage_{variant}_{a.subset}.csv", index=False)
        for k in KS:
            d = df[df.K == k]
            subsets = [("all lesions", d), ("cancer", d[d.label == 1]), ("benign", d[d.label == 0]),
                       ("CC views", d[d.view == "CC"]), ("MLO views", d[d.view == "MLO"])]  # views: exploratory
            for lab, dd in subsets:
                if dd.empty:
                    continue
                lo, hi = boot_ci(dd, "hit")
                summ.append({"variant": variant, "K": k, "subset": lab, "n_images": len(dd),
                             "hit_rate": dd.hit.mean(), "hit_ci_lo": lo, "hit_ci_hi": hi,
                             "hit_rate_random": dd.hit_random.mean(),
                             "coverage": dd.coverage.mean(), "coverage_random": dd.coverage_random.mean()})
    s = pd.DataFrame(summ)
    out = a.out or a.data_dir / f"coverage_summary_{a.subset}.csv"
    s.to_csv(out, index=False)
    with pd.option_context("display.width", 160, "display.precision", 3):
        print(s.to_string(index=False))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()