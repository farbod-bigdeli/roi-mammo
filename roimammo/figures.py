"""Publication figures that do not depend on trained models.

    python -m roimammo.figures examples --data-dir data/miniddsm --bank data/miniddsm/roi_bank_blend.json
        Annotated examples: one row per class (Normal / Benign / Cancer for Mini-DDSM;
        Benign / Malignant for CMMD), CC and MLO of the same breast, lesion outline (red; analysis
        only), top-5 ROI boxes (best in yellow), class/view/laterality labels.

    python -m roimammo.figures arms --data-dir data/miniddsm --bank data/miniddsm/roi_bank_blend.json
        What each augmentation arm feeds the network, for the same images (crop probability 1
        for illustration).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .data import MEAN, STD, MammoDataset
from .roibank import load_prepared


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    return plt


def examples(a):
    plt = _plt()
    idx = pd.read_csv(a.data_dir / "index.csv")
    bank = json.load(open(a.bank))["boxes"] if a.bank else {}
    rng = np.random.default_rng(a.seed)
    groups = []
    for status in ["Normal", "Benign", "Cancer", "Malignant"]:
        sub = idx[idx.status == status]
        if status in ("Cancer", "Malignant"):
            sub = sub[sub.label == 1]
        if status in ("Benign", "Cancer") and sub.has_lesion_mask.any():
            sub = sub[sub.has_lesion_mask.astype(bool)]
        two = sub.groupby("breast_id").filter(lambda g: {"CC", "MLO"} <= set(g.view))
        if two.empty:
            continue
        bid = rng.choice(two.breast_id.unique())
        groups.append((status, two[two.breast_id == bid].sort_values("view")))
    fig, axes = plt.subplots(len(groups), 2, figsize=(4.2, 2.9 * len(groups)), squeeze=False)
    for (status, g), row in zip(groups, axes):
        for ax, (_, r) in zip(row, g.head(2).iterrows()):
            img, _ = load_prepared(a.data_dir, r.image_id)
            ax.imshow(img, cmap="gray")
            les = a.data_dir / "lesions" / f"{r.image_id}.png"
            if les.exists():
                ax.contour(cv2.imread(str(les), cv2.IMREAD_GRAYSCALE) > 127, levels=[0.5], colors="red", linewidths=1)
            for k, b in enumerate(bank.get(r.image_id, [])[:5]):
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                           ec="yellow" if k == 0 else "cyan", lw=1))
            ax.set_title(f"{status} | {'left' if r.laterality == 'L' else 'right'} {r.view}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Red: lesion outline (not used in training). Yellow/cyan: top-5 label-free ROI boxes.", fontsize=7)
    fig.tight_layout()
    out = a.out / f"fig_examples_{a.data_dir.name}"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=250)
    print(f"[done] {out}.pdf")


def arms(a):
    plt = _plt()
    idx = pd.read_csv(a.data_dir / "index.csv")
    bank = json.load(open(a.bank))["boxes"]
    pool = idx[idx.has_lesion_mask.astype(bool) & (idx.label == 1)]
    pool = pool if len(pool) >= a.n else idx
    df = pool.sample(a.n, random_state=a.seed)
    names = [("none", "Standard", "letterbox"), ("rrc", "RandomResizedCrop", "letterbox"),
             ("randcrop", "Random crop", "letterbox"), ("roi", "ROI crop", "letterbox")]
    if a.fill:
        names += [("randcrop", "Random crop (fill)", "fill"), ("roi", "ROI crop (fill)", "fill")]
    fig, axes = plt.subplots(len(names), a.n, figsize=(1.35 * a.n + 0.6, 1.95 * len(names)), squeeze=False)
    for (arm, title, fit), row in zip(names, axes):
        ds = MammoDataset(df, a.data_dir, img_hw=(a.img_h, a.img_w), train=True, arm=arm, p_crop=1.0,
                          jitter=0.1, roi_bank=bank, roi_k=a.roi_k, seed=a.seed, crop_fit=fit)
        for i, ax in enumerate(row):
            x = ds[i]["x"].numpy()[0] * STD[0] + MEAN[0]
            ax.imshow(np.clip(x, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(title, fontsize=8)
    fig.suptitle("Network inputs per augmentation arm (crop probability set to 1 for illustration)", fontsize=8)
    fig.tight_layout()
    out = a.out / f"fig_arms_{a.data_dir.name}"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=250)
    print(f"[done] {out}.pdf")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("examples", "arms"):
        s = sub.add_parser(name)
        s.add_argument("--data-dir", required=True, type=Path)
        s.add_argument("--bank", type=Path, required=(name == "arms"))
        s.add_argument("--out", type=Path, default=Path("results"))
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--n", type=int, default=5)
        s.add_argument("--img-h", type=int, default=448)  # same 2:1 aspect as the training input
        s.add_argument("--img-w", type=int, default=224)
        s.add_argument("--roi-k", type=int, default=1, help="arms: boxes to draw from (the chosen K)")
        s.add_argument("--fill", type=int, default=1, help="arms: also show the crop-fill variant (addendum A1)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    examples(a) if a.cmd == "examples" else arms(a)


if __name__ == "__main__":
    main()
