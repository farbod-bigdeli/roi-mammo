"""Did the white-patch removal erase lesion pixels? (added after the third review)

For Mini-DDSM, prepare.py blacks out pixels at or above 0.98 (after percentile scaling), grown by
0.6 % of the image side, to remove painted-out labels and film edges. Very bright calcifications or
clips inside the breast could fall under this rule. For every image with a lesion outline this
script repeats the preparation in the source frame (same functions and parameters) and counts the
outline pixels that the removal set to zero.

    python -m roimammo.whitecheck --data-dir data/miniddsm --out results/whitecheck [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .prepare import load_source, read_any_gray
from .tissue import bbox_of_mask, tissue_mask, white_patches


def _job(row):
    img = load_source(row["src_path"])
    acc = np.zeros(img.shape, bool)
    for f in json.loads(row["lesion_src"]):
        lm = read_any_gray(Path(f))
        if lm.ndim == 3:
            lm = lm[..., 0]
        lm = lm > (lm.max() / 2 if lm.max() > 0 else 1)
        if lm.shape != img.shape:
            lm = cv2.resize(lm.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        acc |= lm
    # as in tissue.crop_breast(remove_white=True, mask_background=True, dilate_frac=0.01)
    m = tissue_mask(img, remove_white=True)
    x0, y0, x1, y1 = bbox_of_mask(m)
    crop, mcrop, les = img[y0:y1, x0:x1], m[y0:y1, x0:x1], acc[y0:y1, x0:x1]
    r = max(1, int(round(0.01 * max(crop.shape))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    md = cv2.dilate(mcrop.astype(np.uint8), k).astype(bool)
    crop = np.where(md, crop, 0.0).astype(np.float32)
    wp = white_patches(crop)
    raw_white = crop >= 0.98
    n = int(les.sum())
    return {"image_id": row["image_id"], "label": row["label"], "status": row.get("status", ""),
            "lesion_px": n, "lesion_px_removed": int((les & wp).sum()),
            "lesion_px_at_or_above_0.98": int((les & raw_white).sum()),
            "lesion_px_outside_tissue_mask": int((les & ~mcrop).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/miniddsm"))
    ap.add_argument("--out", type=Path, default=Path("results/whitecheck"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(a.data_dir / "index.csv")
    idx = idx[idx.has_lesion_mask.astype(bool)]
    rows = idx.to_dict("records")
    print(f"[info] {len(rows)} images with a lesion outline", flush=True)
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(_job, rows, chunksize=8))
    else:
        res = [_job(r) for r in rows]
    d = pd.DataFrame(res)
    d["frac_removed"] = d.lesion_px_removed / d.lesion_px.clip(lower=1)
    d.to_csv(a.out / "whitecheck_images.csv", index=False)
    lines = [f"Images with a lesion outline: {len(d)}"]
    for name, g in [("all", d)] + [(f"status {s}", gg) for s, gg in d.groupby("status")]:
        hit = g.lesion_px_removed > 0
        lines.append(f"{name}: {len(g)} images; any outline pixel removed in {int(hit.sum())} "
                     f"({100 * hit.mean():.1f}%); outline pixels removed overall "
                     f"{100 * g.lesion_px_removed.sum() / max(g.lesion_px.sum(), 1):.3f}%; "
                     f">=10% of the outline removed in {int((g.frac_removed >= 0.1).sum())}; "
                     f"outline pixels themselves >= 0.98 in {int((g['lesion_px_at_or_above_0.98'] > 0).sum())} images")
    lines.append("Largest removed fractions:")
    lines.append(d.sort_values("frac_removed", ascending=False).head(10)[
        ["image_id", "status", "lesion_px", "lesion_px_removed", "frac_removed"]].to_string(index=False))
    (a.out / "whitecheck.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
