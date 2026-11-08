"""Label-free ROI bank: saliency-scored sliding windows inside the breast, NMS, top-K.

Algorithm (one image, prepared frame):
    1. Breast mask M (from prepare.py). Work at long side 512 for speed. Scores are computed on
       the interior of M (M eroded by 12 px, image border counted as outside, and no scoring in a
       4 % band along the image edges), so neither the skin line nor film edges count as texture.
    2. Saliency S, normalised to [0, 1] inside the interior (1st-99th percentile):
         var   : local variance, window w_v
         log   : local energy of the Laplacian of Gaussian (scale sigma), window w_v
         blend : lambda * var + (1 - lambda) * log            (default, lambda = 0.6)
         intensity : local mean brightness (dense-tissue proxy; ablation)
         random    : random scores (ablation: bank with no saliency)
    3. Square windows with side = f * sqrt(area(M)) for f in side_fracs, stride = min side / 4.
       Keep windows that are at least 50 % breast tissue. Score = mean S in the window.
    4. Greedy non-maximum suppression (IoU 0.5); keep the top K_max boxes by score.
Boxes are stored as [x0, y0, x1, y1, score] in prepared-image pixels (x1, y1 exclusive),
keyed by image_id. Training uses the first K boxes, so K can be ablated without rebuilding.

Usage:
    python -m roimammo.roibank --data-dir data/miniddsm --variant blend
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

WORK_SIDE = 512


def _norm_in_mask(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    v = x[m]
    if v.size == 0:
        return np.zeros_like(x)
    lo, hi = np.percentile(v, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1) * m


def saliency(img: np.ndarray, m: np.ndarray, variant: str, lam: float = 0.6, wv: int = 15,
             sigma: float = 1.5, seed: int = 0) -> np.ndarray:
    k = (wv, wv)
    if variant == "random":
        return np.random.default_rng(seed).random(img.shape).astype(np.float32) * m
    if variant == "intensity":
        return _norm_in_mask(cv2.blur(img, k), m)
    mean = cv2.blur(img, k)
    var = np.maximum(cv2.blur(img * img, k) - mean * mean, 0)
    log = cv2.blur(ndimage.gaussian_laplace(img, sigma) ** 2, k)
    var_n, log_n = _norm_in_mask(var, m), _norm_in_mask(log, m)
    if variant == "var":
        return var_n
    if variant == "log":
        return log_n
    if variant == "blend":
        return lam * var_n + (1 - lam) * log_n
    raise ValueError(variant)


def _box_sum(ii: np.ndarray, x0, y0, x1, y1):
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(boxes: list, thr: float, kmax: int) -> list:
    out = []
    for b in sorted(boxes, key=lambda z: -z[4]):
        if all(iou(b, o) <= thr for o in out):
            out.append(b)
            if len(out) == kmax:
                break
    return out


def propose(img: np.ndarray, m: np.ndarray, variant: str = "blend", lam: float = 0.6,
            side_fracs=(0.25, 0.33, 0.42), min_tissue: float = 0.5, nms_iou: float = 0.5,
            kmax: int = 10, seed: int = 0) -> list:
    h, w = img.shape
    s = min(1.0, WORK_SIDE / max(h, w))
    if s < 1:
        size = (max(1, round(w * s)), max(1, round(h * s)))
        im = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        mm = cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        im, mm = img, m
    # Score only the breast interior: the tissue/background edge has high variance and LoG
    # energy and would otherwise attract the boxes to the skin line.
    r = 15 // 2 + 5  # half the saliency window + 3 sigma of the LoG
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    # erode with the image border counted as background, and drop a 4 % band along the image edges:
    # film edges and rounded film corners are strongly textured and would attract the boxes
    inner = cv2.erode(mm.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    eb = max(2, int(round(0.04 * max(mm.shape))))
    inner[:eb] = inner[-eb:] = False
    inner[:, :eb] = inner[:, -eb:] = False
    if inner.sum() < 0.1 * mm.sum():
        inner = mm
    S = saliency(im, inner, variant, lam=lam, seed=seed).astype(np.float64)
    iS = cv2.integral(S)
    iM = cv2.integral(mm.astype(np.uint8))
    hh, ww = im.shape
    side_base = np.sqrt(max(mm.sum(), 1))
    sides = sorted({max(8, int(round(f * side_base))) for f in side_fracs})
    stride = max(2, sides[0] // 4)

    def candidates(min_t, sides):
        cand = []
        for sd in sides:
            sd = min(sd, hh, ww)
            for y0 in range(0, hh - sd + 1, stride):
                for x0 in range(0, ww - sd + 1, stride):
                    x1, y1 = x0 + sd, y0 + sd
                    a = sd * sd
                    if _box_sum(iM, x0, y0, x1, y1) < min_t * a:
                        continue
                    cand.append([x0, y0, x1, y1, float(_box_sum(iS, x0, y0, x1, y1) / a)])
        return cand

    # Fallbacks for unusual images (very narrow or small breast): relax the tissue rule, then
    # use smaller windows. Every image therefore gets boxes; such images are logged by main().
    keep = nms(candidates(min_tissue, sides), nms_iou, kmax)
    if not keep:
        keep = nms(candidates(0.2, sides), nms_iou, kmax)
    if not keep:
        keep = nms(candidates(0.2, [max(8, sd // 2) for sd in sides]), nms_iou, kmax)
    if not keep:
        ys, xs = np.nonzero(mm) if mm.any() else (np.array([0, hh - 1]), np.array([0, ww - 1]))
        keep = [[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, 0.0]]
    inv = 1.0 / s
    out = []
    for x0, y0, x1, y1, sc in keep:
        out.append([int(round(x0 * inv)), int(round(y0 * inv)), min(w, int(round(x1 * inv))),
                    min(h, int(round(y1 * inv))), round(sc, 5)])
    return out


def random_box_like(bw: int, bh: int, tissue: np.ndarray, rng: np.random.Generator, coords=None):
    """A box of size (bw, bh) whose centre is a uniformly drawn breast pixel, clipped to the image.
    Pass ``coords = np.nonzero(tissue)`` when drawing many boxes for the same image."""
    h, w = tissue.shape
    ys, xs = coords if coords is not None else np.nonzero(tissue)
    if ys.size == 0:
        cy, cx = h / 2, w / 2
    else:
        i = rng.integers(ys.size)
        cy, cx = ys[i], xs[i]
    x0 = int(np.clip(cx - bw / 2, 0, max(0, w - bw)))
    y0 = int(np.clip(cy - bh / 2, 0, max(0, h - bh)))
    return [x0, y0, min(w, x0 + bw), min(h, y0 + bh)]


def load_prepared(data_dir: Path, image_id: str):
    img = cv2.imread(str(data_dir / "images" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(str(data_dir / "tissue" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or m is None:
        raise IOError(f"prepared files missing for {image_id} in {data_dir}")
    return img.astype(np.float32) / 255.0, m > 127


def _job(args):
    data_dir, image_id, kw = args
    img, m = load_prepared(data_dir, image_id)
    return image_id, propose(img, m, seed=zlib.crc32(image_id.encode()), **kw)


def overlay_grid(data_dir: Path, bank: dict, ids: list, path: Path, k: int = 5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(ids)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))
    for ax, iid in zip(np.atleast_1d(axes).ravel(), ids):
        img, _ = load_prepared(data_dir, iid)
        ax.imshow(img, cmap="gray")
        les = data_dir / "lesions" / f"{iid}.png"
        if les.exists():
            ax.contour(cv2.imread(str(les), cv2.IMREAD_GRAYSCALE) > 127, levels=[0.5], colors="red", linewidths=1)
        for r, b in enumerate(bank.get(iid, [])[:k]):
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                       ec="yellow" if r == 0 else "cyan", lw=1.2))
        ax.set_title(iid, fontsize=7)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    fig.suptitle("Top-%d ROI boxes (yellow = best); red = lesion outline (analysis only)" % k, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--variant", default="blend", choices=["blend", "var", "log", "intensity", "random"])
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--side-fracs", type=float, nargs="+", default=[0.25, 0.33, 0.42])
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None, help="default <data-dir>/roi_bank_<variant>.json")
    ap.add_argument("--overlays", type=int, default=20, help="number of example images to draw")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)

    index = pd.read_csv(a.data_dir / "index.csv")
    kw = dict(variant=a.variant, lam=a.lam, side_fracs=tuple(a.side_fracs), nms_iou=a.nms_iou, kmax=a.kmax)
    jobs = [(a.data_dir, i, kw) for i in index.image_id]
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(_job, jobs, chunksize=32))
    else:
        res = [_job(j) for j in jobs]
    bank = dict(res)
    n_empty = sum(1 for v in bank.values() if not v)
    # images where the normal rule found no window (fewer boxes than kmax): list them for inspection
    odd = sorted(i for i, v in bank.items() if len(v) < a.kmax)
    if odd:
        (a.data_dir / f"roi_fewboxes_{a.variant}.txt").write_text("\n".join(odd) + "\n")
        print(f"[info] {len(odd)} images got fewer than {a.kmax} boxes (fallback rules); "
              f"listed in roi_fewboxes_{a.variant}.txt - look at a few of them")
    out = a.out or a.data_dir / f"roi_bank_{a.variant}.json"
    json.dump({"meta": {**kw, "side_fracs": list(a.side_fracs), "work_side": WORK_SIDE,
                        "frame": "prepared images (images/<image_id>.png)", "n_images": len(bank)},
               "boxes": bank}, open(out, "w"))
    nb = [len(v) for v in bank.values()]
    print(f"[done] {out}: {len(bank)} images, boxes per image median {np.median(nb):.0f} (min {min(nb)}), "
          f"{n_empty} images without boxes")
    if a.overlays:
        rng = np.random.default_rng(0)
        les = index[index.has_lesion_mask.astype(bool)].image_id.tolist()
        other = index[~index.has_lesion_mask.astype(bool)].image_id.tolist()
        pick = list(rng.choice(les, min(len(les), a.overlays // 2), replace=False)) if les else []
        pick += list(rng.choice(other, min(len(other), a.overlays - len(pick)), replace=False))
        overlay_grid(a.data_dir, bank, pick, a.data_dir / f"roi_overlays_{a.variant}.png")
        print(f"[done] overlays: {a.data_dir / f'roi_overlays_{a.variant}.png'}")


if __name__ == "__main__":
    main()
