"""Copy the ORIGINAL source files of chosen images into one folder, for debugging segmentation.

    python tests/collect_raw.py --data-dir data/miniddsm --out ~/qc_raw/miniddsm --ids A_1833_1_L_MLO B_3053_1_R_MLO
    python tests/collect_raw.py --data-dir data/miniddsm --out ~/qc_raw/miniddsm --from-qc 12
--from-qc N takes N flagged images from qc.csv (spread over the flag types) plus 4 unflagged ones.
DICOMs are converted to 16-bit PNG to keep uploads small. Then:  cd ~ && zip -r qc_raw.zip qc_raw
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--from-qc", type=int, default=0)
    a = ap.parse_args()
    full = a.data_dir / "index_full.csv"
    idx = pd.read_csv(full if full.exists() else a.data_dir / "index.csv").set_index("image_id")
    ids = list(a.ids)
    if a.from_qc:
        q = pd.read_csv(a.data_dir / "qc.csv")
        for f in ["lesion_outside", "tiny_tissue", "border_band", "ragged"]:
            ids += q[q[f]].sample(min(int(q[f].sum()), max(1, a.from_qc // 4)), random_state=1).image_id.tolist()
        ids += q[~q.any_flag].sample(min(4, int((~q.any_flag).sum())), random_state=1).image_id.tolist()
    a.out.mkdir(parents=True, exist_ok=True)
    for i in dict.fromkeys(ids):
        src = Path(idx.loc[i, "src_path"])
        if src.suffix.lower() == ".dcm":
            import pydicom
            ds = pydicom.dcmread(str(src))
            x = ds.pixel_array
            if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                x = x.max() - x
            cv2.imwrite(str(a.out / f"{i}.png"), x.astype(np.uint16))
        else:
            shutil.copy(src, a.out / f"{i}{src.suffix.lower()}")
        print("copied", i)


if __name__ == "__main__":
    main()
