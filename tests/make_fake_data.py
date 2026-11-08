"""Create small synthetic Mini-DDSM and CMMD trees with the real folder/file conventions.

Only for testing the pipeline; images are crude drawings, not mammograms.
    python tests/make_fake_data.py --out /tmp/fake
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def breast(h, w, side, rng, lesion=False, label_box=True, bright=1.0):
    y, x = np.mgrid[0:h, 0:w]
    cx = 0 if side == "R" else w - 1  # chest wall on the image edge
    r = np.sqrt(((x - cx) / (0.75 * w)) ** 2 + ((y - h / 2) / (0.45 * h)) ** 2)
    img = np.clip(1.1 - r, 0, 1) ** 0.6 * 0.7 * bright
    tex = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 3)
    img = np.where(r < 1, img + 0.25 * (tex - 0.5), 0) + 0.03 * rng.random((h, w))
    les = np.zeros((h, w), np.uint8)
    if lesion:
        ly, lx = int(h * rng.uniform(0.35, 0.65)), int(abs(cx - w * rng.uniform(0.25, 0.45)))
        cv2.ellipse(les, (lx, ly), (int(w * 0.07), int(h * 0.05)), rng.uniform(0, 180), 0, 360, 255, -1)
        img = np.where(les > 0, img + 0.25 + 0.15 * rng.random((h, w)), img)
    if label_box:  # bright film label away from the breast, should be masked out
        x0 = w - 45 if side == "R" else 5
        img[8:30, x0:x0 + 40] = 0.95
    return np.clip(img, 0, 1), les


def fake_miniddsm(root: Path, n_per=(10, 10, 10), seed=0):
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for status, n in zip(["Normal", "Benign", "Cancer"], n_per):
        for _ in range(n):
            k += 1
            letter = "ABCD"[k % 4]
            pid = f"{letter}_{k:04d}_1"
            idx = f"{k:04d}"
            d = root / "MINI-DDSM-Complete-JPEG-8" / status / idx
            d.mkdir(parents=True, exist_ok=True)
            les_side = rng.choice(["L", "R"]) if status != "Normal" else None
            for side in ["LEFT", "RIGHT"]:
                for view in ["CC", "MLO"]:
                    name = f"{pid}.{side}_{view}.jpg"
                    has = les_side == side[0]
                    img, les = breast(320, 200, side[0], rng, lesion=has)
                    cv2.imwrite(str(d / name), (img * 255).astype(np.uint8))
                    tc = ""
                    if has:
                        mname = f"{pid}.{side}_{view}_Mask.jpg"
                        cv2.imwrite(str(d / mname), les)
                        tc = f"{status}\\{idx}\\{mname}"
                    rows.append({"fullPath": f"{status}\\{idx}\\{name}", "fileName": name, "View": view,
                                 "Side": side, "Status": status, "Tumour_Contour": tc, "Tumour_Contour2": "",
                                 "Age": int(rng.integers(40, 75)), "Density": int(rng.integers(1, 5))})
    x = root / "Data-MoreThanTwoMasks"
    x.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(x / "Data-MoreThanTwoMasks.xlsx", index=False)


def fake_cmmd(root: Path, n=30, seed=1):
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    rng = np.random.default_rng(seed)
    clin = []
    for i in range(1, n + 1):
        pid = f"D1-{i:04d}"
        lab_side = rng.choice(["L", "R"])
        cls = "Malignant" if i % 2 else "Benign"
        clin.append({"ID1": pid, "LeftRight": lab_side, "Age": int(rng.integers(30, 80)), "number": 2,
                     "abnormality": "mass", "classification": cls, "subtype": ""})
        d = root / "CMMD" / pid / "07-18-2010-NA-NA-12345" / "1.000000-NA-67890"
        d.mkdir(parents=True, exist_ok=True)
        j = 0
        for side in ["L", "R"]:
            for view, meaning in [("CC", "cranio-caudal"), ("MLO", "medio-lateral oblique")]:
                j += 1
                img, _ = breast(300, 240, side, rng, lesion=(side == lab_side and cls == "Malignant"), label_box=False)
                mono1 = i % 5 == 0
                px = (img * 4000).astype(np.uint16)
                if mono1:
                    px = (4000 - px).astype(np.uint16)
                fm = FileMetaDataset()
                fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"
                fm.MediaStorageSOPInstanceUID = generate_uid()
                fm.TransferSyntaxUID = ExplicitVRLittleEndian
                ds = Dataset()
                ds.file_meta = fm
                ds.PatientID, ds.ImageLaterality, ds.Manufacturer, ds.PatientAge = pid, side, "GE", "045Y"
                vc = Dataset()
                vc.CodeMeaning = meaning
                ds.ViewCodeSequence = [vc]
                ds.SOPClassUID, ds.SOPInstanceUID = fm.MediaStorageSOPClassUID, fm.MediaStorageSOPInstanceUID
                ds.Modality = "MG"
                ds.Rows, ds.Columns = px.shape
                ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit = 1, 16, 12, 11
                ds.PixelRepresentation = 0
                ds.PhotometricInterpretation = "MONOCHROME1" if mono1 else "MONOCHROME2"
                ds.PixelData = px.tobytes()
                ds.save_as(d / f"1-{j}.dcm", enforce_file_format=True)
    pd.DataFrame(clin).to_excel(root / "CMMD_clinicaldata_revision.xlsx", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fake_miniddsm(a.out / "miniddsm_raw")
    fake_cmmd(a.out / "cmmd_raw")
    print("fake data in", a.out)
