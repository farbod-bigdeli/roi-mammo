"""Breast tissue segmentation and cropping, shared by data preparation, the ROI bank and training.

All images are handled as float32 arrays in [0, 1], shape (H, W).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_triangle


def to_unit_float(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Convert an 8- or 16-bit (or float) grayscale image to float32 in [0, 1].

    8-bit images are divided by 255. Deeper images are rescaled between robust
    percentiles of the non-zero pixels, so that 16-bit data are not clipped
    (PIL's convert('L') clips 16-bit values above 255 to white).
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    x = img.astype(np.float32)
    nz = x[x > 0]
    if nz.size < 100:
        nz = x.ravel()
    lo, hi = np.percentile(nz, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _largest(m: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    return lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)


def white_patches(img: np.ndarray, level: float = 0.98, grow_frac: float = 0.006) -> np.ndarray:
    """Pure-white areas (painted-out film labels, unexposed film edges), slightly grown to cover
    the JPEG halo. On the 8-bit Mini-DDSM films breast tissue, including the pectoral muscle,
    never reaches this level (checked on 25 films)."""
    w = img >= level
    if not w.any():
        return w
    return cv2.dilate(w.astype(np.uint8), _disk(grow_frac * max(img.shape))).astype(bool)


def tissue_mask(img: np.ndarray, work_side: int = 512, margin_frac: float = 0.04,
                remove_white: bool = False) -> np.ndarray:
    """Binary breast mask. Designed and checked on scanned film (Mini-DDSM, incl. hazy scanner-A
    films with bright film edges and labels) and on digital images (CMMD).

    1. Work at long side 512; Gaussian blur (sigma 2 px).
    2. Triangle threshold (the foot of the background peak) computed on the image interior,
       i.e. excluding an outer border band of 4 %: keeps fatty tissue near the skin line and
       ignores film edges and scanner borders. (A plain Otsu threshold cut fatty tissue away and,
       on hazy films, selected the whole film instead of the breast.)
    2b. Optionally (scanned film) pure-white patches are removed first: painted-out labels and
       unexposed film edges.
    3. Light opening; the largest connected component of the interior is the breast.
    4. The breast is extended into the border band only straight outwards from where it
       reaches the band (chest wall, top, bottom), so bright strips running ALONG the border are
       never added; corners are included where the breast fills both adjacent bands.
    5. Closing and hole filling.
    Returns a bool array with the same shape as ``img``.
    """
    h, w = img.shape
    s = min(1.0, work_side / max(h, w))
    sm = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    b = cv2.GaussianBlur(sm.astype(np.float32), (0, 0), 2.0)
    H, W = b.shape
    mg = max(2, int(round(margin_frac * max(H, W))))
    if H <= 2 * mg + 2 or W <= 2 * mg + 2:
        return np.ones_like(img, dtype=bool)
    inner = np.zeros((H, W), bool)
    inner[mg:H - mg, mg:W - mg] = True
    if float(b[inner].max() - b[inner].min()) < 1e-3:
        return np.ones_like(img, dtype=bool)
    t = threshold_triangle(b[inner], nbins=256)
    fg = b > t
    if remove_white:  # labels and film edges must not join the breast
        fg &= ~white_patches(sm)
    m = cv2.morphologyEx((fg & inner).astype(np.uint8), cv2.MORPH_OPEN, _disk(0.006 * max(H, W))).astype(bool)
    comp = _largest(m)
    if not comp.any():
        return np.ones_like(img, dtype=bool)
    ext = comp.copy()
    ext[:mg] = fg[:mg] & comp[mg][None, :]
    ext[H - mg:] = fg[H - mg:] & comp[H - mg - 1][None, :]
    ext[:, :mg] |= fg[:, :mg] & comp[:, mg][:, None]
    ext[:, W - mg:] |= fg[:, W - mg:] & comp[:, W - mg - 1][:, None]
    for ys, xs, ry, cx in [(slice(0, mg), slice(0, mg), mg, mg), (slice(0, mg), slice(W - mg, W), mg, W - mg - 1),
                           (slice(H - mg, H), slice(0, mg), H - mg - 1, mg),
                           (slice(H - mg, H), slice(W - mg, W), H - mg - 1, W - mg - 1)]:
        ext[ys, xs] = fg[ys, xs] & ext[ys, cx][:, None] & ext[ry, xs][None, :]
    ext = cv2.morphologyEx(ext.astype(np.uint8), cv2.MORPH_CLOSE, _disk(0.01 * max(H, W))).astype(bool)
    ext = ndimage.binary_fill_holes(_largest(ext))
    if s < 1:
        ext = cv2.resize(ext.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return ext


def bbox_of_mask(mask: np.ndarray, pad_frac: float = 0.02) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) with exclusive x1, y1, padded by a fraction of the box size."""
    ys, xs = np.where(mask)
    h, w = mask.shape
    if ys.size == 0:
        return 0, 0, w, h
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = int(pad_frac * (x1 - x0)), int(pad_frac * (y1 - y0))
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)


def crop_breast(img: np.ndarray, max_side: int = 1024, mask_background: bool = True,
                dilate_frac: float = 0.01, remove_white: bool = False):
    """Crop to the breast and downscale (never upscale) so the long side <= max_side.

    Returns (image, mask, meta). ``meta`` records the crop box in source pixels and the
    scale, so that lesion masks can be mapped into the same frame.
    If ``mask_background`` is set, pixels outside a slightly dilated breast mask are set
    to 0, which removes film labels and markers that could act as shortcuts.
    """
    m = tissue_mask(img, remove_white=remove_white)
    x0, y0, x1, y1 = bbox_of_mask(m)
    crop, mcrop = img[y0:y1, x0:x1], m[y0:y1, x0:x1]
    if mask_background:
        r = max(1, int(round(dilate_frac * max(crop.shape))))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        md = cv2.dilate(mcrop.astype(np.uint8), k).astype(bool)
        crop = np.where(md, crop, 0.0).astype(np.float32)
    if remove_white:  # painted-out labels inside the crop are blacked out too
        crop = np.where(white_patches(crop), 0.0, crop).astype(np.float32)
    h, w = crop.shape
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        size = (max(1, round(w * s)), max(1, round(h * s)))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        mcrop = cv2.resize(mcrop.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    meta = {"crop_x0": int(x0), "crop_y0": int(y0), "crop_x1": int(x1), "crop_y1": int(y1),
            "scale": float(s), "h": int(crop.shape[0]), "w": int(crop.shape[1])}
    return crop.astype(np.float32), mcrop, meta


def map_mask_to_frame(src_mask: np.ndarray, meta: dict) -> np.ndarray:
    """Apply the same crop and scale as ``crop_breast`` to a source-frame mask (e.g. a lesion)."""
    m = src_mask[meta["crop_y0"]:meta["crop_y1"], meta["crop_x0"]:meta["crop_x1"]]
    if m.shape != (meta["h"], meta["w"]):
        m = cv2.resize(m.astype(np.uint8), (meta["w"], meta["h"]), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)
