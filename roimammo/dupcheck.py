"""Find exact and near-duplicate images across patients (possible leakage between folds).

TCIA notes that some CMMD files have identical pixel data. A duplicate shared by two patients in
different partitions would put the same image in training and evaluation.

    python -m roimammo.dupcheck --data-dir data/cmmd                 # report + duplicates_montage.png
    python -m roimammo.dupcheck --data-dir data/cmmd --drop exact    # remove patients in exact groups
    python -m roimammo.dupcheck --data-dir data/cmmd --drop all      # ... and in (confirmed) near groups
Writes <data-dir>/duplicates.csv and prints how many duplicate groups span several patients and
several partitions. Exact = identical prepared pixels; near = identical 16x16 difference hash
(look at the montage: near matches can be different images that merely look alike).

--drop removes ALL images of every patient involved in a cross-patient duplicate group (the same
exam registered under two IDs would otherwise leak through the patient's other images too). The
original index is kept as index_full.csv, removed patients are listed in excluded_patients.csv,
and split.csv must then be rebuilt with  python -m roimammo.splits ... --force
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def dhash(img: np.ndarray, n: int = 16) -> str:
    small = cv2.resize(img, (n + 1, n), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return np.packbits(bits).tobytes().hex()


def montage(data_dir: Path, d: pd.DataFrame, idx: pd.DataFrame, path: Path, max_groups=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = d.drop_duplicates("image_ids").head(max_groups)
    if d.empty:
        return
    cols = int(min(4, d.n_images.max()))
    fig, axes = plt.subplots(len(d), cols, figsize=(2.2 * cols, 3.0 * len(d)), squeeze=False)
    info = idx.set_index("image_id")
    for row, g in zip(axes, d.itertuples()):
        for ax in row:
            ax.axis("off")
        for ax, iid in zip(row, g.image_ids.split()[:cols]):
            img = cv2.imread(str(data_dir / "images" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap="gray")
            r = info.loc[iid]
            ax.set_title(f"[{g.kind}] {iid}\nlabel {r.label} | {r.part}", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--drop", choices=["exact", "all"], default=None)
    a = ap.parse_args(argv)
    full = a.data_dir / "index_full.csv"
    idx = pd.read_csv(full if full.exists() else a.data_dir / "index.csv")  # always check the full index
    sp_path = a.data_dir / "split.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path)
        sp["part"] = np.where(sp.split == "test", "test", "fold" + sp.fold.astype(str))
        idx = idx.merge(sp[["patient_id", "part"]], on="patient_id", how="left")
    else:
        idx["part"] = "?"
    exact, near = [], []
    for iid in idx.image_id:
        img = cv2.imread(str(a.data_dir / "images" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE)
        exact.append(hashlib.md5(img.tobytes()).hexdigest())
        near.append(dhash(img))
    idx["md5"], idx["dhash"] = exact, near
    rows = []
    for kind in ("md5", "dhash"):
        for h, g in idx.groupby(kind):
            if len(g) > 1:
                rows.append({"kind": "exact" if kind == "md5" else "near", "hash": h, "n_images": len(g),
                             "n_patients": g.patient_id.nunique(), "n_parts": g.part.nunique(),
                             "labels": "/".join(map(str, sorted(g.label.unique()))),
                             "image_ids": " ".join(g.image_id)})
    d = pd.DataFrame(rows)
    out = a.data_dir / "duplicates.csv"
    d.to_csv(out, index=False)
    if d.empty:
        print("[done] no duplicate images found")
        return
    for kind in ("exact", "near"):
        k = d[d.kind == kind]
        cross_pat = k[k.n_patients > 1]
        cross_part = k[k.n_parts > 1]
        print(f"[{kind}] groups: {len(k)} | spanning >1 patient: {len(cross_pat)} | "
              f"spanning >1 partition (leakage risk): {len(cross_part)} | label conflicts: {(k.labels.str.contains('/')).sum()}")
    print(f"[done] details in {out}")
    # near groups that are also exact are shown once (as exact)
    show = pd.concat([d[d.kind == "exact"], d[(d.kind == "near") & ~d.image_ids.isin(d[d.kind == "exact"].image_ids)]])
    montage(a.data_dir, show[show.n_patients > 1], idx, a.data_dir / "duplicates_montage.png")
    print(f"[done] look at {a.data_dir / 'duplicates_montage.png'}")

    if a.drop:
        kinds = ["exact"] if a.drop == "exact" else ["exact", "near"]
        grp = d[d.kind.isin(kinds) & (d.n_patients > 1)]
        ids = set(" ".join(grp.image_ids).split())
        pats = sorted(idx[idx.image_id.isin(ids)].patient_id.unique())
        if not full.exists():
            idx.drop(columns=["md5", "dhash", "part"]).to_csv(full, index=False)
        keep = idx[~idx.patient_id.isin(pats)].drop(columns=["md5", "dhash", "part"])
        keep.to_csv(a.data_dir / "index.csv", index=False)
        pd.DataFrame({"patient_id": pats, "reason": f"cross-patient duplicate image ({a.drop})"}).to_csv(
            a.data_dir / "excluded_patients.csv", index=False)
        print(f"[drop] removed {len(pats)} patients ({len(idx) - len(keep)} images); index.csv now has {len(keep)} images. "
              f"Original kept as index_full.csv.\n[drop] NOW REBUILD THE SPLITS with --force (same arguments as before).")


if __name__ == "__main__":
    main()
