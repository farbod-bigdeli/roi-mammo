"""Dataset with the augmentation arms compared in the study.

Arms (train only; validation/test always see the full prepared image):
    none      standard augmentation only (flip, small affine, brightness/contrast/gamma)
    rrc       + RandomResizedCrop (area 30-100 %, aspect 3/4-4/3), every sample: common practice
    randcrop  + with prob p, a crop of the same size as a bank box but centred on a random breast pixel
    roi       + with prob p, a crop from the image's ROI bank (top K), with jitter
    cutmix    standard augmentation here; CutMix is applied on the batch in train.py
'randcrop' differs from 'roi' only in where the box is placed, so roi - randcrop isolates the
value of the saliency. Jitter (independent scale for w and h, shift of the centre, each
Uniform(-a, a)) is applied identically in both arms. Crops with less than 30 % breast tissue
are redrawn (up to 10 times).

All images are letterboxed (aspect preserved, zero padding) to (img_h, img_w), so full images
and square crops are not distorted differently.

Randomness: every sample draws from np.random.default_rng([seed, epoch, index]), so runs are
reproducible regardless of the number of data-loader workers.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .roibank import random_box_like

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ARMS = ("none", "rrc", "randcrop", "roi", "cutmix")


def letterbox(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ih, iw = img.shape
    s = min(h / ih, w / iw)
    nh, nw = max(1, round(ih * s)), max(1, round(iw * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((h, w), np.float32)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def jitter_box(b, W, H, a: float, rng):
    x0, y0, x1, y1 = b[:4]
    w, h = x1 - x0, y1 - y0
    cx, cy = x0 + w / 2, y0 + h / 2
    e = rng.uniform(-a, a, 4) if a > 0 else np.zeros(4)
    nw, nh = max(16, w * (1 + e[0])), max(16, h * (1 + e[1]))
    cx, cy = cx + e[2] * w, cy + e[3] * h
    nx0, ny0 = int(np.clip(cx - nw / 2, 0, W - 1)), int(np.clip(cy - nh / 2, 0, H - 1))
    nx1, ny1 = int(np.clip(cx + nw / 2, nx0 + 1, W)), int(np.clip(cy + nh / 2, ny0 + 1, H))
    return [nx0, ny0, nx1, ny1]


def photometric_geometric(img: np.ndarray, rng) -> np.ndarray:
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
        M[:, 2] += rng.uniform(-0.05, 0.05, 2) * np.array([w, h])
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.5:
        img = np.clip((img - 0.5) * rng.uniform(0.9, 1.1) + 0.5 + rng.uniform(-0.1, 0.1), 0, 1)
    if rng.random() < 0.5:
        img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)
    return np.ascontiguousarray(img, dtype=np.float32)


class MammoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir, img_hw=(896, 448), train=False, arm="none",
                 p_crop=0.1, jitter=0.1, roi_bank: dict | None = None, roi_k=5, unit="image",
                 seed=0, rrc_scale=(0.3, 1.0), min_tissue=0.3):
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        self.df = df.reset_index(drop=True)
        self.dir = data_dir
        self.h, self.w = img_hw
        self.train, self.arm, self.p, self.jit = train, arm, p_crop, jitter
        self.bank, self.k, self.unit = roi_bank or {}, roi_k, unit
        self.seed, self.epoch, self.rrc_scale, self.min_tissue = seed, 0, rrc_scale, min_tissue
        if unit == "breast":
            self.groups = []
            for bid, g in self.df.groupby("breast_id", sort=True):
                cc = g[g.view == "CC"].index.tolist()
                mlo = g[g.view == "MLO"].index.tolist()
                rest = g.index.tolist()
                a = cc[0] if cc else rest[0]
                b = mlo[0] if mlo else (rest[1] if len(rest) > 1 else a)
                self.groups.append((bid, a, b))
        if train and arm in ("roi", "randcrop"):
            miss = [i for i in self.df.image_id if not self.bank.get(i)]
            if miss:
                raise RuntimeError(f"{len(miss)} training images have no ROI boxes (e.g. {miss[:3]}). "
                                   "Rebuild the bank for this data directory.")

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return len(self.groups) if self.unit == "breast" else len(self.df)

    def _load(self, image_id):
        img = cv2.imread(str(self.dir / "images" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"missing prepared image {image_id}")
        return img.astype(np.float32) / 255.0

    def _tissue(self, image_id):
        return cv2.imread(str(self.dir / "tissue" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE) > 127

    def _crop(self, img, image_id, rng):
        H, W = img.shape
        if self.arm == "rrc":
            area = H * W
            for _ in range(10):
                ta = area * rng.uniform(*self.rrc_scale)
                ar = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
                cw, ch = int(round(np.sqrt(ta * ar))), int(round(np.sqrt(ta / ar)))
                if 0 < cw <= W and 0 < ch <= H:
                    x0, y0 = rng.integers(0, W - cw + 1), rng.integers(0, H - ch + 1)
                    return img[y0:y0 + ch, x0:x0 + cw]
            return img
        if self.arm not in ("roi", "randcrop") or rng.random() >= self.p:
            return img
        boxes = self.bank[image_id][: self.k]
        tissue = self._tissue(image_id)
        for _ in range(10):
            b = boxes[rng.integers(len(boxes))]
            if self.arm == "randcrop":
                b = random_box_like(b[2] - b[0], b[3] - b[1], tissue, rng)
            x0, y0, x1, y1 = jitter_box(b, W, H, self.jit, rng)
            if tissue[y0:y1, x0:x1].mean() >= self.min_tissue:
                return img[y0:y1, x0:x1]
        x0, y0, x1, y1 = boxes[0][:4]
        return img[y0:y1, x0:x1]

    def _one(self, i, rng):
        r = self.df.iloc[i]
        img = self._load(r.image_id)
        if self.train:
            img = self._crop(img, r.image_id, rng)
        img = letterbox(img, self.h, self.w)
        if self.train:
            img = photometric_geometric(img, rng)
        x = (np.repeat(img[..., None], 3, 2) - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy())

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        if self.unit == "breast":
            bid, a, b = self.groups[idx]
            x = torch.stack([self._one(a, rng), self._one(b, rng)])
            r = self.df.iloc[a]
            return {"x": x, "y": torch.tensor(float(r.label)), "id": bid, "patient_id": r.patient_id,
                    "breast_id": bid}
        r = self.df.iloc[idx]
        return {"x": self._one(idx, rng), "y": torch.tensor(float(r.label)), "id": r.image_id,
                "patient_id": r.patient_id, "breast_id": r.breast_id}
