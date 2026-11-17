"""Replay the training crop sampler and measure what the crops kept of the lesion (added after review).

The coverage analysis (coverage.py) asks whether any of the top-K bank boxes contains the lesion
centroid. Training does something more specific: with probability p it samples ONE of the top-K
boxes (roi) or a box of the same size centred on a random breast pixel (randcrop), jitters it,
redraws boxes with too little tissue and, with --crop-fit fill, widens the box to the input
aspect. This script runs exactly that code (MammoDataset.crop_box) on development images with a
lesion outline, many times per image, and records for every crop:
    centroid   the lesion centroid lies inside the crop
    any        at least one lesion pixel lies inside the crop
    area       fraction of the lesion area inside the crop
The geometric augmentation applied after cropping (rotation up to 10 deg, 5 % shift) is not
replayed; it can move the lesion border slightly but not the centroid out of a typical crop.

Rates are conditional on a crop being taken; the fraction of all training samples of a positive
image that are a lesion-bearing crop is p times the rate. Differences roi - randcrop are paired by
image and come with a 95 % patient-bootstrap interval.

    python -m roimammo.cropreplay --data-dir data/miniddsm --results results --draws 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .data import MammoDataset

METRICS = ("centroid", "any", "area")


def replay(ddir: Path, index: pd.DataFrame, bank: dict, arm: str, k: int, jitter: float, crop_fit: str,
           img_hw: tuple[int, int], draws: int, seeds: list[int]) -> pd.DataFrame:
    ds = MammoDataset(index, ddir, img_hw=img_hw, train=True, arm=arm, p_crop=1.0, jitter=jitter,
                      roi_bank=bank, roi_k=k, crop_fit=crop_fit)
    rows = []
    for i, r in enumerate(ds.df.itertuples()):
        les = cv2.imread(str(ddir / "lesions" / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE) > 127
        tis = cv2.imread(str(ddir / "tissue" / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE) > 127
        H, W = tis.shape
        ys, xs = np.nonzero(les)
        if ys.size == 0:
            continue
        cy, cx, area = ys.mean(), xs.mean(), ys.size
        coords = np.nonzero(tis)
        acc = np.zeros(3)
        n = 0
        for seed in seeds:
            for epoch in range(draws):
                rng = np.random.default_rng([seed, epoch, i])
                rng.random()  # the draw that decides whether to crop (p), as in MammoDataset._crop
                x0, y0, x1, y1 = ds.crop_box(r.image_id, H, W, rng, tissue=tis, coords=coords)
                inside = les[y0:y1, x0:x1].sum()
                acc += [(x0 <= cx < x1) and (y0 <= cy < y1), inside > 0, inside / area]
                n += 1
        rows.append({"image_id": r.image_id, "patient_id": r.patient_id, "label": r.label, "view": r.view,
                     **dict(zip(METRICS, acc / n))})
    return pd.DataFrame(rows)


def boot_mean(df: pd.DataFrame, col: str, n: int = 2000, seed: int = 0):
    g = df.groupby("patient_id")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    return np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/miniddsm"))
    ap.add_argument("--results", type=Path, default=Path("results"), help="folder with chosen_<dataset>.json")
    ap.add_argument("--draws", type=int, default=10, help="epochs replayed per seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--img-h", type=int, default=896)
    ap.add_argument("--img-w", type=int, default=448)
    ap.add_argument("--subset", choices=["dev", "all"], default="dev")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    ch = json.load(open(a.results / f"chosen_{a.data_dir.name}.json"))
    bank = json.load(open(a.data_dir / f"roi_bank_{ch['bank']}.json"))["boxes"]
    index = pd.read_csv(a.data_dir / "index.csv")
    index = index[index.has_lesion_mask.astype(bool)]
    if a.subset == "dev":
        sp = pd.read_csv(a.data_dir / "split.csv")
        index = index[index.patient_id.isin(sp.patient_id[sp.fold >= 0])]
    index = index[index.image_id.map(lambda i: bool(bank.get(i)))]
    print(f"[info] setting {ch['bank']}, K={ch['roi_k']}, jitter={ch['jitter']}, p={ch['p_crop']}; "
          f"{len(index)} images with a lesion outline; {a.draws} x {len(a.seeds)} crops per image and arm", flush=True)

    per = []
    for fit in ("letterbox", "fill"):
        for arm in ("roi", "randcrop"):
            d = replay(a.data_dir, index, bank, arm, ch["roi_k"], ch["jitter"], fit, (a.img_h, a.img_w),
                       a.draws, a.seeds)
            per.append(d.assign(arm=arm, crop_fit=fit))
            print(f"[info] {fit} {arm} done", flush=True)
    per = pd.concat(per, ignore_index=True)
    out = a.out or a.results / "cropreplay"
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "cropreplay_per_image.csv", index=False)

    rows = []
    groups = [("malignant (positive images)", lambda d: d.label == 1), ("benign", lambda d: d.label == 0),
              ("malignant, CC", lambda d: (d.label == 1) & (d.view == "CC")),
              ("malignant, MLO", lambda d: (d.label == 1) & (d.view == "MLO"))]
    for fit in ("letterbox", "fill"):
        roi = per[(per.arm == "roi") & (per.crop_fit == fit)].set_index("image_id")
        rnd = per[(per.arm == "randcrop") & (per.crop_fit == fit)].set_index("image_id").loc[roi.index]
        for gname, sel in groups:
            m = sel(roi).to_numpy()
            if not m.any():
                continue
            gr, gx = roi[m], rnd[m]
            for met in METRICS:
                diff = pd.DataFrame({"patient_id": gr.patient_id, "d": gr[met].to_numpy() - gx[met].to_numpy()})
                lo, hi = boot_mean(diff, "d")
                rows.append({"crop_fit": fit, "group": gname, "metric": met, "n_images": int(m.sum()),
                             "roi": gr[met].mean(), "randcrop": gx[met].mean(), "diff": diff.d.mean(),
                             "diff_lo": lo, "diff_hi": hi,
                             "roi_per_sample": ch["p_crop"] * gr[met].mean(),
                             "randcrop_per_sample": ch["p_crop"] * gx[met].mean()})
    s = pd.DataFrame(rows)
    s.to_csv(out / "cropreplay_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"[done] {out / 'cropreplay_summary.csv'}")


if __name__ == "__main__":
    main()
