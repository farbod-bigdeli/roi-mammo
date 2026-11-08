"""Quality check of breast segmentation on prepared images.

Flags per image:
    lesion_outside   a lesion outline (Mini-DDSM) lies mostly (> 50 %) outside the breast mask:
                     the breast was cut away
    tiny_tissue      breast mask covers < 15 % of the prepared image
    border_band      breast mask touches > 50 % of the image border (film-edge strips merged
                     into the mask; a breast normally touches only the chest-wall side)
    ragged           mask perimeter^2 / (4 pi area) > 4 (jagged outline, e.g. fat cut away)
    not_breast       the masked region is nearly uniform (std < 0.03) or > 20 % pure white:
                     a blank/corrupt image or a label instead of a breast

    python -m roimammo.qc --data-dir data/miniddsm
Writes qc.csv, prints flag counts overall and by site, and saves qc_montage.png with up to 30
flagged images (breast mask outline in green, lesion outline in red).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def metrics(data_dir: Path, image_id: str, has_lesion: bool) -> dict:
    tis = cv2.imread(str(data_dir / "tissue" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE) > 127
    h, w = tis.shape
    border = np.concatenate([tis[0], tis[-1], tis[:, 0], tis[:, -1]])
    cnts, _ = cv2.findContours(tis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    area = max(tis.sum(), 1)
    img = cv2.imread(str(data_dir / "images" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    v = img[tis] if tis.any() else img.ravel()
    out = {"image_id": image_id, "tissue_std": float(v.std()), "tissue_white": float((v >= 0.98).mean()),
           "tissue_frac": tis.mean(), "border_contact": border.mean(),
           "roughness": per ** 2 / (4 * np.pi * area), "lesion_inside": np.nan}
    lp = data_dir / "lesions" / f"{image_id}.png"
    if has_lesion and lp.exists():
        les = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE) > 127
        if les.shape == tis.shape:
            out["lesion_inside"] = (les & tis).sum() / max(les.sum(), 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--n-show", type=int, default=30)
    ap.add_argument("--drop-not-breast", action="store_true",
                    help="remove images flagged not_breast from index.csv (listed in excluded_images.csv)")
    a = ap.parse_args(argv)
    idx = pd.read_csv(a.data_dir / "index.csv")
    q = pd.DataFrame([metrics(a.data_dir, i, bool(h)) for i, h in zip(idx.image_id, idx.has_lesion_mask)])
    q = idx[["image_id", "patient_id", "site", "status", "label", "lesion_src", "has_lesion_mask"]].merge(q, on="image_id")
    # lesion outside the mask, or outside the breast crop altogether (outline lost in prepare)
    lost = (q.lesion_src.fillna("[]") != "[]") & ~q.has_lesion_mask.astype(bool)
    q["lesion_outside"] = (q.lesion_inside < 0.5) | lost
    q["tiny_tissue"] = q.tissue_frac < 0.15
    q["border_band"] = q.border_contact > 0.5
    q["ragged"] = q.roughness > 4
    q["not_breast"] = (q.tissue_std < 0.03) | (q.tissue_white > 0.2)  # blank strip or label only
    flags = ["lesion_outside", "tiny_tissue", "border_band", "ragged", "not_breast"]
    q["any_flag"] = q[flags].any(axis=1)
    q.to_csv(a.data_dir / "qc.csv", index=False)
    print(f"[qc] {len(q)} images; flagged {int(q.any_flag.sum())} ({100 * q.any_flag.mean():.1f} %)")
    print(q[flags + ["any_flag"]].sum().to_string())
    print("\n[qc] flagged share by site:")
    print(q.groupby(q.site.astype(object).where(q.site.notna(), "unknown"))[flags + ["any_flag"]].mean().round(3).to_string())
    print(f"\n[qc] roughness median {q.roughness.median():.2f}, 95th pct {q.roughness.quantile(.95):.2f}; "
          f"border contact median {q.border_contact.median():.2f}")

    bad = q[q.any_flag].sample(min(a.n_show, int(q.any_flag.sum())), random_state=0) if q.any_flag.any() else q.head(0)
    if len(bad):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = 6
        rows = int(np.ceil(len(bad) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 3.4 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, r in zip(axes.ravel(), bad.itertuples()):
            img = cv2.imread(str(a.data_dir / "images" / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE)
            tis = cv2.imread(str(a.data_dir / "tissue" / f"{r.image_id}.png"), cv2.IMREAD_GRAYSCALE) > 127
            ax.imshow(img, cmap="gray")
            ax.contour(tis, levels=[0.5], colors="lime", linewidths=0.8)
            lp = a.data_dir / "lesions" / f"{r.image_id}.png"
            if r.has_lesion_mask and lp.exists():
                ax.contour(cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE) > 127, levels=[0.5], colors="red", linewidths=0.8)
            f = ",".join(x.split("_")[0] for x in flags if getattr(r, x))
            ax.set_title(f"{r.image_id}\n{f}", fontsize=6)
        fig.tight_layout()
        fig.savefig(a.data_dir / "qc_montage.png", dpi=110)
        plt.close(fig)
        print(f"[done] {a.data_dir / 'qc_montage.png'}")

    if a.drop_not_breast:
        bad = q[q.not_breast]
        if bad.empty:
            print("[drop] no image flagged not_breast")
        else:
            idx2 = pd.read_csv(a.data_dir / "index.csv")
            idx2[~idx2.image_id.isin(bad.image_id)].to_csv(a.data_dir / "index.csv", index=False)
            ex = a.data_dir / "excluded_images.csv"
            old = pd.read_csv(ex) if ex.exists() else pd.DataFrame(columns=["image_id", "reason"])
            new = pd.DataFrame({"image_id": bad.image_id, "reason": "not a breast (QC: blank image or label only)"})
            pd.concat([old, new]).drop_duplicates("image_id").to_csv(ex, index=False)
            print(f"[drop] removed {len(bad)} image(s): {', '.join(bad.image_id)}; listed in excluded_images.csv. "
                  "Splits and ROI banks stay valid (splits are by patient; extra bank entries are ignored).")


if __name__ == "__main__":
    main()