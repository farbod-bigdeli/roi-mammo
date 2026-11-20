"""Size of the bank boxes, per bank (added after the third review).

The random-crop arm takes its box size from the ROI bank (brightness), so ROI crop and random crop
are size-matched by construction. The central-random-box arm (addendum B) uses the random-score
bank's own boxes, and the coverage table compares banks that each use their own boxes. A bank
scored by the mean of uniform noise may prefer the smallest window size, so this script reports,
for each bank, how often the top box (K=1) has each of the three window sizes, and, image by image,
how often the random-score top box has the same size as the brightness top box.

Window sides are recomputed exactly as in roibank.propose (side = round(f * sqrt(breast area)) at
the 512-pixel working size, f in side_fracs), so every box is assigned to its size class without
approximation; boxes from the fallback rules are counted as "fallback".

    python -m roimammo.boxsize --data-root data --out results/boxsize
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .roibank import WORK_SIDE

BANKS = ("intensity", "random", "blend")


def work_sides(tissue: np.ndarray, side_fracs) -> tuple[float, list[int]]:
    h, w = tissue.shape
    s = min(1.0, WORK_SIDE / max(h, w))
    if s < 1:
        size = (max(1, round(w * s)), max(1, round(h * s)))
        mm = cv2.resize(tissue.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mm = tissue
    base = np.sqrt(max(mm.sum(), 1))
    return s, [max(8, int(round(f * base))) for f in side_fracs]


def size_class(box, s: float, sides: list[int], names: list[str]) -> str:
    side = (box[2] - box[0]) * s  # back to working pixels (boxes are square there)
    d = [abs(side - sd) / sd for sd in sides]
    i = int(np.argmin(d))
    return names[i] if d[i] <= 0.06 else "fallback"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/boxsize"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    lines, rows = [], []

    def say(t=""):
        print(t, flush=True)
        lines.append(t)

    for ds in ("miniddsm", "cmmd"):
        dd = a.data_root / ds
        banks = {}
        for b in BANKS:
            f = dd / f"roi_bank_{b}.json"
            if f.exists():
                j = json.load(open(f))
                banks[b] = (j["meta"], j["boxes"])
        if "intensity" not in banks:
            say(f"[skip] {ds}: no roi_bank_intensity.json")
            continue
        fracs = banks["intensity"][0]["side_fracs"]
        names = [f"{f:.2f}" for f in fracs]
        ids = sorted(banks["intensity"][1])
        for iid in ids:
            t = cv2.imread(str(dd / "tissue" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
            if t is None:
                continue
            s, sides = work_sides(t > 127, fracs)
            r = {"dataset": ds, "image_id": iid}
            for b, (meta, boxes) in banks.items():
                bx = boxes.get(iid) or []
                if not bx:
                    continue
                top = bx[0]
                r[f"{b}_class"] = size_class(top, s, sides, names)
                r[f"{b}_area"] = (top[2] - top[0]) * (top[3] - top[1])
                r[f"{b}_k5_classes"] = ",".join(size_class(x, s, sides, names) for x in bx[:5])
            rows.append(r)
        d = pd.DataFrame([r for r in rows if r["dataset"] == ds])
        say(f"\n== {ds}: {len(d)} images; window sides = {', '.join(names)} x sqrt(breast area)")
        say("Top box (K=1): share of images by window size")
        tab = {}
        for b in banks:
            if f"{b}_class" in d:
                tab[b] = d[f"{b}_class"].value_counts(normalize=True)
        say((pd.DataFrame(tab).fillna(0) * 100).round(1).to_string())
        say("Top-5 boxes: share of boxes by window size")
        tab5 = {}
        for b in banks:
            if f"{b}_k5_classes" in d:
                tab5[b] = d[f"{b}_k5_classes"].str.split(",").explode().value_counts(normalize=True)
        say((pd.DataFrame(tab5).fillna(0) * 100).round(1).to_string())
        for b in ("random", "blend"):
            if f"{b}_class" in d:
                q = d.dropna(subset=[f"{b}_class", "intensity_class"])
                same = (q[f"{b}_class"] == q["intensity_class"]).mean()
                ratio = (q[f"{b}_area"] / q["intensity_area"])
                say(f"{b} vs brightness top box: same size in {100 * same:.1f}% of images; "
                    f"area ratio median {ratio.median():.2f}, mean {ratio.mean():.2f} "
                    f"(IQR {ratio.quantile(.25):.2f}-{ratio.quantile(.75):.2f})")
    pd.DataFrame(rows).to_csv(a.out / "boxsize_images.csv", index=False)
    (a.out / "boxsize.txt").write_text("\n".join(lines) + "\n")
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
