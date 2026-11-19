"""How many prepared images still contain the whole film (and so possibly film labels/markers)?

The background (and with it film labels) is removed by masking everything outside the breast.
When the breast mask covers almost the whole prepared image, the segmentation did not separate
the breast from the film, and labels or markers inside the frame survive. This script counts
such images by site and label, and draws a montage of a random sample for visual inspection.

    python -m roimammo.labelcheck --data-dir data/miniddsm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--fill", type=float, default=0.85, help="mask fraction above which the frame counts as 'whole film'")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)

    idx = pd.read_csv(a.data_dir / "index.csv")
    rows = []
    for r in idx.itertuples():
        m = cv2.imread(str(a.data_dir / "tissue" / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = m > 127
        rows.append({"image_id": r.image_id, "patient_id": r.patient_id, "site": getattr(r, "site", "all"),
                     "label": r.label, "view": r.view, "mask_frac": float(m.mean()),
                     "touches": int(m[0].any()) + int(m[-1].any()) + int(m[:, 0].any()) + int(m[:, -1].any())})
    df = pd.DataFrame(rows)
    df["whole_film"] = df.mask_frac >= a.fill
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / f"labelcheck_{a.data_dir.name}.csv", index=False)
    print(f"images: {len(df)}  mask fraction >= {a.fill}: {df.whole_film.sum()} ({100 * df.whole_film.mean():.1f}%)")
    print("\nby site (count, % of site's images):")
    print(df.groupby("site").whole_film.agg(["sum", "mean"]).assign(mean=lambda d: (100 * d["mean"]).round(1)).to_string())
    print("\nby label (count, % of label's images):")
    print(df.groupby("label").whole_film.agg(["sum", "mean"]).assign(mean=lambda d: (100 * d["mean"]).round(1)).to_string())
    print("\nmask fraction quantiles:", df.mask_frac.quantile([0.5, 0.9, 0.95, 0.99]).round(3).to_dict())

    sel = df[df.whole_film]
    if len(sel):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = sel.sample(min(a.n, len(sel)), random_state=0)
        cols = 8
        rows_ = int(np.ceil(len(sel) / cols))
        fig, axes = plt.subplots(rows_, cols, figsize=(cols * 1.5, rows_ * 2.3), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, r in zip(axes.ravel(), sel.itertuples()):
            img = cv2.imread(str(a.data_dir / "images" / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{r.image_id}\nsite {r.site} y={r.label}", fontsize=6)
        fig.tight_layout()
        fig.savefig(a.out / f"labelcheck_{a.data_dir.name}.png", dpi=120)
        print(f"[montage] {a.out / f'labelcheck_{a.data_dir.name}.png'}")


if __name__ == "__main__":
    main()