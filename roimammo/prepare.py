"""Prepare a dataset once: index every image, crop to the breast, save 8-bit PNGs.

Output directory layout (same for every dataset):
    <out>/images/<image_id>.png   8-bit, breast-cropped, long side <= --max-side
    <out>/tissue/<image_id>.png   breast mask in the same frame (0/255)
    <out>/lesions/<image_id>.png  lesion mask in the same frame (only where available; analysis only)
    <out>/index.csv               one row per image

Every later step (splits, ROI bank, training, analysis) works in this single frame and
refers to images by ``image_id``, never by file path.

Usage:
    python -m roimammo.prepare --dataset miniddsm --root /data/miniddsm --out data/miniddsm
    python -m roimammo.prepare --dataset cmmd --root /data/cmmd --out data/cmmd
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .tissue import crop_breast, map_mask_to_frame, to_unit_float

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ----------------------------------------------------------------------------- helpers

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:  # some TCIA xlsx files trip pandas; fall back to a raw openpyxl read
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows = list(wb.worksheets[0].iter_rows(values_only=True))
        return pd.DataFrame(rows[1:], columns=[str(c).strip() for c in rows[0]])


def read_any_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def file_map(root: Path) -> dict[str, list[Path]]:
    """Map lower-case file stem -> paths, for every image file under root."""
    out: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            out.setdefault(p.stem.lower(), []).append(p)
    return out


def _norm_view(v: str) -> str:
    v = str(v).upper()
    if "CC" in v or "CRANIO" in v:
        return "CC"
    if "MLO" in v or "OBLIQUE" in v:
        return "MLO"
    return "OTHER"


# ----------------------------------------------------------------------------- Mini-DDSM

def index_miniddsm(root: Path, variant: str) -> pd.DataFrame:
    """Index Mini-DDSM using Data-MoreThanTwoMasks.xlsx.

    Breast-level label: a breast is positive if the case status is Cancer AND that breast has a
    lesion outline. A cancer patient's other breast is negative (DDSM labels cases, not breasts).
    """
    xlsx = next(iter(sorted(root.rglob("Data-MoreThanTwoMasks.xlsx"))), None)
    if xlsx is None:
        sys.exit(f"Data-MoreThanTwoMasks.xlsx not found under {root}")
    df = read_table(xlsx)
    need = {"fileName", "View", "Side", "Status"}
    if not need.issubset(df.columns):
        sys.exit(f"{xlsx} lacks columns {need - set(df.columns)}; found {list(df.columns)}")
    img_dir = next((p for p in root.rglob(f"MINI-DDSM-Complete-{variant}") if p.is_dir()), None)
    if img_dir is None:
        sys.exit(f"MINI-DDSM-Complete-{variant} folder not found under {root}")
    fmap_img = file_map(img_dir)
    fmap_all = file_map(root)

    contour_cols = [c for c in df.columns if str(c).startswith("Tumour_Contour")]
    df = df.copy()
    df["patient_id"] = df["fileName"].astype(str).str.split(".").str[0]
    df["laterality"] = df["Side"].astype(str).str.upper().str[0]
    df["view"] = df["View"].map(_norm_view)

    def contours(row):
        out = []
        for c in contour_cols:
            v = row[c]
            if isinstance(v, str) and len(v.strip()) > 5:
                out.append(v.strip())
        return out

    df["contours"] = df.apply(contours, axis=1)
    df["has_outline"] = df["contours"].map(len) > 0
    df["breast_id"] = df["patient_id"] + "_" + df["laterality"]
    breast_outline = df.groupby("breast_id")["has_outline"].any()
    status = df["Status"].astype(str).str.capitalize()
    bad = df.loc[(status == "Normal") & df["has_outline"], "fileName"]
    if len(bad):
        print(f"[warn] {len(bad)} Normal images carry an outline; kept as negative")
    df["status"] = status
    df["label"] = ((status == "Cancer") & df["breast_id"].map(breast_outline)).astype(int)
    df["site"] = df["patient_id"].str[0]  # DDSM volume letter; marks scanner/institution
    df["age"] = pd.to_numeric(df.get("Age"), errors="coerce")
    df["density"] = pd.to_numeric(df.get("Density"), errors="coerce")

    paths, masks, missing = [], [], 0
    for _, r in df.iterrows():
        idx = r["patient_id"].split("_")[1] if "_" in r["patient_id"] else ""
        cand = [img_dir / r["Status"] / idx / r["fileName"]]
        stem = Path(r["fileName"]).stem.lower()
        cand += fmap_img.get(stem, [])
        p = next((c for c in cand if Path(c).is_file()), None)
        if p is None:
            missing += 1
        paths.append(str(p) if p else "")
        ms = []
        for c in r["contours"]:
            st = Path(c.replace("\\", "/")).stem.lower()
            hits = fmap_all.get(st, [])
            if hits:
                ms.append(str(hits[0]))
        masks.append(json.dumps(ms))
    df["src_path"], df["lesion_src"] = paths, masks
    if missing:
        print(f"[warn] {missing} images listed in the xlsx were not found; they are dropped")
        df = df[df["src_path"] != ""]
    df["image_id"] = df["patient_id"] + "_" + df["laterality"] + "_" + df["view"]
    dup = df["image_id"].duplicated(keep=False)
    if dup.any():
        df.loc[dup, "image_id"] += "_" + df.loc[dup].groupby("image_id").cumcount().astype(str)
    keep = ["image_id", "patient_id", "breast_id", "laterality", "view", "label", "status",
            "site", "age", "density", "src_path", "lesion_src"]
    return df[keep].reset_index(drop=True)


# ----------------------------------------------------------------------------- CMMD

def index_cmmd(root: Path, keep_unlabeled: bool, clinical: Path | None = None) -> pd.DataFrame:
    """Index CMMD from DICOM headers plus CMMD_clinicaldata_revision.xlsx (or a CSV export).

    Breast-level label: Malignant = 1, Benign = 0 (biopsy-confirmed). Breasts without a row in
    the clinical table were not biopsied; they are dropped unless --cmmd-keep-unlabeled.
    """
    import pydicom

    if clinical is not None:
        tab = clinical
    else:
        cands = [p for p in root.rglob("*") if p.is_file() and "clinical" in p.name.lower()
                 and p.suffix.lower() in (".xlsx", ".xls", ".csv")]
        tab = sorted(cands, key=lambda p: (p.suffix.lower() != ".xlsx", len(p.name)))[0] if cands else None
    if tab is None or not Path(tab).is_file():
        sys.exit(f"CMMD clinical table not found under {root}. Download 'Clinical data (XLSX)' from "
                 "https://doi.org/10.7937/tcia.eqde-4b16 and put it in that folder, or pass --clinical <file>.")
    print(f"[info] clinical table: {tab}")
    lab = read_table(tab)
    lab.columns = [str(c).strip() for c in lab.columns]
    # TCIA: "for rows where there exists BOTH a value for ID1 and ID2, TCIA stores ONLY the ID2
    # value as PatientID", so register every row under both IDs.
    side = lab["LeftRight"].astype(str).str.strip().str.upper().str[0]
    c = lab["classification"].astype(str).str.strip().str.capitalize()
    cls = {}
    for col in [x for x in ("ID1", "ID2") if x in lab.columns]:
        ids = lab[col].astype(str).str.strip()
        ok = lab[col].notna() & (ids != "") & (ids.str.lower() != "nan")
        for pid, s, v in zip(ids[ok], side[ok], c[ok]):
            cls[f"{pid}_{s}"] = v
    cls = pd.Series(cls)
    print(f"[info] clinical table: {len(lab)} rows -> {len(cls)} labelled breast keys (ID1 and ID2)")

    rows = []
    for p in sorted(root.rglob("*.dcm")):
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        pid = str(getattr(ds, "PatientID", "") or p.parts[-4]).strip()
        lat = str(getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", "")).upper()[:1]
        view = "OTHER"
        try:
            view = _norm_view(ds.ViewCodeSequence[0].CodeMeaning)
        except Exception:
            view = _norm_view(getattr(ds, "ViewPosition", ""))
        rows.append({"patient_id": pid, "laterality": lat or "U", "view": view,
                     "site": str(getattr(ds, "Manufacturer", "unknown")), "src_path": str(p),
                     "age": pd.to_numeric(re.sub(r"\D", "", str(getattr(ds, "PatientAge", ""))) or np.nan)})
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit(f"no DICOM files under {root}")
    df["breast_id"] = df["patient_id"] + "_" + df["laterality"]
    df["status"] = df["breast_id"].map(cls).fillna("Unlabeled")
    if not keep_unlabeled:
        n0 = len(df)
        df = df[df["status"] != "Unlabeled"]
        print(f"[info] dropped {n0 - len(df)} images of breasts without a biopsy result")
    df["label"] = (df["status"] == "Malignant").astype(int)
    df["density"] = np.nan
    df["lesion_src"] = "[]"
    df = df.sort_values(["patient_id", "laterality", "view", "src_path"])
    df["image_id"] = df["breast_id"] + "_" + df["view"]
    df["image_id"] += "_" + df.groupby("image_id").cumcount().astype(str)
    keep = ["image_id", "patient_id", "breast_id", "laterality", "view", "label", "status",
            "site", "age", "density", "src_path", "lesion_src"]
    return df[keep].reset_index(drop=True)


def load_source(path: str) -> np.ndarray:
    if path.lower().endswith(".dcm"):
        import pydicom
        ds = pydicom.dcmread(path)
        x = ds.pixel_array.astype(np.float32)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            x = x.max() - x
        x -= x.min()
        return to_unit_float(x.astype(np.uint16) if x.max() > 255 else x.astype(np.uint8))
    return to_unit_float(read_any_gray(Path(path)))


# ----------------------------------------------------------------------------- processing

def _process(args):
    row, out, max_side, mask_bg, rm_white = args
    img = load_source(row["src_path"])
    crop, m, meta = crop_breast(img, max_side=max_side, mask_background=mask_bg, remove_white=rm_white)
    cv2.imwrite(str(out / "images" / f"{row['image_id']}.png"), np.round(crop * 255).astype(np.uint8))
    cv2.imwrite(str(out / "tissue" / f"{row['image_id']}.png"), m.astype(np.uint8) * 255)
    lesion_files = json.loads(row["lesion_src"])
    has_lesion = False
    if lesion_files:
        acc = np.zeros(img.shape, bool)
        for f in lesion_files:
            lm = read_any_gray(Path(f))
            if lm.ndim == 3:
                lm = lm[..., 0]
            lm = lm > (lm.max() / 2 if lm.max() > 0 else 1)
            if lm.shape != img.shape:
                lm = cv2.resize(lm.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            acc |= lm
        lf = map_mask_to_frame(acc, meta)
        if lf.any():
            cv2.imwrite(str(out / "lesions" / f"{row['image_id']}.png"), lf.astype(np.uint8) * 255)
            has_lesion = True
    meta.update({"image_id": row["image_id"], "src_h": int(img.shape[0]), "src_w": int(img.shape[1]),
                 "has_lesion_mask": has_lesion, "tissue_frac": float(m.mean())})
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["miniddsm", "cmmd"])
    ap.add_argument("--root", required=True, type=Path, help="folder holding the downloaded dataset")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variant", default="JPEG-8", choices=["JPEG-8", "PNG-16"], help="Mini-DDSM image folder")
    ap.add_argument("--max-side", type=int, default=1024, help="long side after cropping (no upscaling)")
    ap.add_argument("--no-mask-background", action="store_true", help="keep pixels outside the breast")
    ap.add_argument("--remove-white", choices=["auto", "yes", "no"], default="auto",
                    help="remove pure-white label patches/film edges (auto: Mini-DDSM yes, CMMD no)")
    ap.add_argument("--cmmd-keep-unlabeled", action="store_true")
    ap.add_argument("--clinical", type=Path, default=None, help="CMMD clinical table (xlsx/csv), if not under --root")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="process only the first N images (testing)")
    a = ap.parse_args(argv)

    import shutil
    for sub in ["images", "tissue", "lesions"]:  # start empty: no files from earlier runs survive
        if (a.out / sub).exists():
            shutil.rmtree(a.out / sub)
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    stale = [f for f in ("index_full.csv", "excluded_patients.csv", "excluded_images.csv", "duplicates.csv", "qc.csv")
             if (a.out / f).exists()]
    for f in stale:  # a fresh index invalidates earlier duplicate/QC results
        (a.out / f).unlink()
    if stale:
        print(f"[info] removed stale {', '.join(stale)}: rerun dupcheck, splits (--force), roibank, coverage, qc")
    df = index_miniddsm(a.root, a.variant) if a.dataset == "miniddsm" else index_cmmd(a.root, a.cmmd_keep_unlabeled, a.clinical)
    if a.limit:
        df = df.head(a.limit)
    print(f"[info] {len(df)} images, {df.patient_id.nunique()} patients, {df.breast_id.nunique()} breasts")
    rm_white = a.dataset == "miniddsm" if a.remove_white == "auto" else a.remove_white == "yes"
    print(f"[info] remove pure-white label patches: {rm_white}")
    jobs = [(r, a.out, a.max_side, not a.no_mask_background, rm_white) for r in df.to_dict("records")]
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            metas = list(ex.map(_process, jobs, chunksize=16))
    else:
        metas = [_process(j) for j in jobs]
    out = df.merge(pd.DataFrame(metas), on="image_id")
    out.insert(0, "dataset", a.dataset)
    out.to_csv(a.out / "index.csv", index=False)

    # summary for the Methods section
    print("\n[summary] images by status and label")
    print(pd.crosstab(out["status"], out["label"]))
    print("\n[summary] source image size (px): height median %d (range %d-%d), width median %d (range %d-%d)" % (
        out.src_h.median(), out.src_h.min(), out.src_h.max(), out.src_w.median(), out.src_w.min(), out.src_w.max()))
    print("[summary] prepared size: height median %d, width median %d; tissue fraction median %.2f" % (
        out.h.median(), out.w.median(), out.tissue_frac.median()))
    print(f"[summary] images with a lesion mask: {int(out.has_lesion_mask.sum())}")
    lost = (out.lesion_src != "[]") & ~out.has_lesion_mask.astype(bool)
    if lost.any():
        print(f"[summary] images whose lesion outline fell outside the breast crop: {int(lost.sum())} "
              "(segmentation misses; listed by roimammo.qc)")
    print(f"[done] wrote {a.out / 'index.csv'}")


if __name__ == "__main__":
    main()