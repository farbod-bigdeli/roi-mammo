"""Audit of the breast-level reference standard and of the cohort counts (added after peer review).

Mini-DDSM labels cases, not breasts. The study labels a breast positive if the case is a cancer
case and that breast carries a lesion outline (prepare.index_miniddsm). Two situations can
mislabel a breast:
    cancer_both   a cancer case with outlines on both breasts: both are labelled positive, but
                  one of them may carry only a benign finding;
    cancer_none   a cancer case with no outline on either breast: both are labelled negative.
The script counts both, by split, and writes the affected breasts to exclude_breasts.csv, so that
the main analysis can be repeated without them:
    python -m roimammo.analyze final --exclude results/labelaudit/exclude_breasts.csv \
        --out results/labelaudit/sens_breast
If original DDSM .OVERLAY files are available (--ddsm-overlays), their per-lesion PATHOLOGY is
used to check every breast label directly.

It also prints breasts that do not have exactly one CC and one MLO view (the two-view model uses
the single available view twice for such a breast) and, with --cmmd-root, reconciles the CMMD
counts: DICOM files, unique SOP instances, images per breast and view, and clinical-table rows.

    python -m roimammo.labelaudit --data-root data --out results/labelaudit \
        [--ddsm-overlays /path/to/DDSM] [--cmmd-root /path/to/CMMD/raw]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT_LINES: list[str] = []


def say(s: str = ""):
    print(s, flush=True)
    OUT_LINES.append(s)


def miniddsm_audit(ddir: Path, out: Path, overlays: Path | None):
    idx = pd.read_csv(ddir / "index.csv")
    sp = pd.read_csv(ddir / "split.csv")
    fold = sp.set_index("patient_id")["fold"]
    idx["has_outline"] = idx["lesion_src"].fillna("[]").map(lambda s: len(json.loads(s)) > 0)
    br = idx.groupby("breast_id").agg(patient_id=("patient_id", "first"), laterality=("laterality", "first"),
                                      status=("status", "first"), label=("label", "max"),
                                      outline=("has_outline", "any"), n_img=("image_id", "size"),
                                      views=("view", lambda v: ",".join(sorted(v)))).reset_index()
    br["split"] = np.where(br.patient_id.map(fold).fillna(-1) >= 0, "dev", "test")
    case = br.groupby("patient_id").agg(status=("status", "first"), n_outlined=("outline", "sum"),
                                        n_breasts=("breast_id", "size"))
    cancer = case[case.status == "Cancer"]
    both = cancer.index[cancer.n_outlined >= 2]
    none = cancer.index[cancer.n_outlined == 0]
    say("== Mini-DDSM: breast labels ==")
    say(f"cases: {len(case)} ({case.status.value_counts().to_dict()})")
    say(f"breasts: {len(br)}, positive: {int(br.label.sum())}")
    say(f"cancer cases: {len(cancer)}")
    say(f"  outline on exactly one breast (label unambiguous): {int((cancer.n_outlined == 1).sum())}")
    say(f"  outlines on both breasts (one may be benign):        {len(both)}")
    say(f"  no outline on either breast (both labelled negative): {len(none)}")
    nb = br[(br.status != "Cancer") & br.outline]
    say(f"benign/normal cases with an outlined breast (negative, as intended): {len(nb)} breasts "
        f"({nb.status.value_counts().to_dict()})")
    amb = br[br.patient_id.isin(both) | br.patient_id.isin(none)].copy()
    amb["reason"] = np.where(amb.patient_id.isin(both), "cancer_both", "cancer_none")
    say("affected breasts by split (all breasts of the affected cases):")
    tab = amb.groupby(["split", "reason"]).agg(breasts=("breast_id", "size"), labelled_pos=("label", "sum"))
    say(tab.to_string())
    for s in ("dev", "test"):
        n_s, p_s = (br.split == s).sum(), int(br[br.split == s].label.sum())
        a_s = amb[amb.split == s]
        say(f"  {s}: {len(a_s)} of {n_s} breasts ({100 * len(a_s) / max(n_s, 1):.1f}%), "
            f"{int(a_s.label.sum())} of {p_s} positive breasts ({100 * a_s.label.sum() / max(p_s, 1):.1f}%)")
    out.mkdir(parents=True, exist_ok=True)
    amb[["breast_id", "patient_id", "split", "reason", "label", "outline"]].to_csv(out / "exclude_breasts.csv",
                                                                                  index=False)
    say(f"-> {out / 'exclude_breasts.csv'}")

    if overlays is not None:
        say("\n== Mini-DDSM labels vs DDSM overlay pathology ==")
        path_of: dict[str, set] = {}
        files = list(overlays.rglob("*.OVERLAY")) + list(overlays.rglob("*.overlay"))
        for f in files:
            m = re.match(r"([A-D]_\d{4}_1)\.(LEFT|RIGHT)_(CC|MLO)", f.name, re.I)
            if not m:
                continue
            key = f"{m.group(1).upper()}_{m.group(2)[0].upper()}"
            txt = f.read_text(errors="ignore")
            path_of.setdefault(key, set()).update(p.upper() for p in re.findall(r"PATHOLOGY\s+(\S+)", txt))
        say(f"overlay files: {len(files)}, breasts with pathology: {len(path_of)}")
        if path_of:
            br["truth"] = br.breast_id.map(lambda b: int("MALIGNANT" in path_of.get(b, set())))
            br["has_overlay"] = br.breast_id.isin(path_of)
            cmp_ = br[br.has_overlay | (br.label == 1)]
            ct = pd.crosstab(cmp_.label.rename("study label"), cmp_.truth.rename("overlay malignant"))
            say(ct.to_string())
            wrong = br[(br.label != br.truth) & (br.has_overlay | (br.label == 1))]
            wrong[["breast_id", "split", "status", "label", "truth"]].to_csv(out / "label_vs_overlay.csv", index=False)
            say(f"disagreements: {len(wrong)} (dev {int((wrong.split == 'dev').sum())}, "
                f"test {int((wrong.split == 'test').sum())}) -> {out / 'label_vs_overlay.csv'}")
            if len(wrong):
                wrong[["breast_id", "patient_id", "split"]].assign(reason="overlay_disagrees", label=wrong.label,
                                                                   outline=True).to_csv(
                    out / "exclude_breasts_overlay.csv", index=False)
    views_audit(br, "Mini-DDSM")


def views_audit(br: pd.DataFrame, name: str):
    odd = br[br.views != "CC,MLO"]
    say(f"\n== {name}: breasts without exactly one CC and one MLO view: {len(odd)} ==")
    if len(odd):
        say(odd[["breast_id", "views", "n_img", "label"]].head(20).to_string(index=False))
        say("(two-view model: a breast with one view uses that view for both inputs; with extra images,"
            " the first CC and the first MLO image are used)")


def cmmd_audit(ddir: Path, raw: Path | None):
    say("\n== CMMD: cohort counts ==")
    idx = pd.read_csv(ddir / "index.csv")
    br = idx.groupby("breast_id").agg(label=("label", "max"), n_img=("image_id", "size"),
                                      views=("view", lambda v: ",".join(sorted(v)))).reset_index()
    say(f"index.csv (after exclusions): {len(idx)} images, {idx.patient_id.nunique()} patients, {len(br)} breasts, "
        f"{int(br.label.sum())} malignant")
    dup = idx.groupby(["breast_id", "view"]).size()
    say(f"breast-view pairs with more than one image: {int((dup > 1).sum())} (extra images: {int((dup - 1).sum())})")
    views_audit(br, "CMMD")
    if raw is None or not raw.is_dir():
        src = [str(Path(x).parent) for x in idx.src_path.dropna()]
        guess = Path(os.path.commonpath(src)) if src else None  # the folder that holds every CMMD DICOM
        if raw is not None:
            say(f"[warn] --cmmd-root {raw} is not a folder")
        raw = guess
        say(f"raw CMMD folder inferred from index.csv: {raw}")
    if raw is None or not raw.is_dir():
        say("(raw CMMD folder not found; skipping the DICOM count)")
        return
    import pydicom
    rows = []
    for p in sorted(raw.rglob("*.dcm")):
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        rows.append({"path": str(p), "sop": str(getattr(ds, "SOPInstanceUID", "")),
                     "series": str(getattr(ds, "SeriesInstanceUID", "")),
                     "pid": str(getattr(ds, "PatientID", "")).strip(),
                     "lat": str(getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", "")).upper()[:1]})
    d = pd.DataFrame(rows, columns=["path", "sop", "series", "pid", "lat"])
    if d.empty:
        say(f"(no .dcm files under {raw}; skipping the DICOM count)")
        return
    say(f"raw DICOM files: {len(d)}; unique SOPInstanceUID: {d.sop.nunique()}; patients: {d.pid.nunique()}; "
        f"series: {d.series.nunique()}")
    if d.sop.nunique() < len(d):
        dd = d[d.sop.duplicated(keep=False)].sort_values("sop")
        say(f"files sharing a SOPInstanceUID (the same image stored twice): {len(dd)}")
        say(dd[["pid", "lat", "path"]].head(20).to_string(index=False))
    per_breast = d.groupby(["pid", "lat"]).size()
    say(f"images per breast in the raw data: {per_breast.value_counts().sort_index().to_dict()}")
    tabs = [p for q in (raw, raw.parent, raw.parent.parent) for p in q.glob("*")
            if p.is_file() and "clinical" in p.name.lower() and p.suffix.lower() in (".xlsx", ".xls", ".csv")]
    if tabs:
        t = tabs[0]
        lab = pd.read_excel(t) if t.suffix.lower() != ".csv" else pd.read_csv(t)
        say(f"clinical table {t.name}: {len(lab)} rows "
            f"({lab['classification'].value_counts().to_dict() if 'classification' in lab else 'no classification column'})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/labelaudit"))
    ap.add_argument("--ddsm-overlays", type=Path, default=None, help="folder with original DDSM .OVERLAY files")
    ap.add_argument("--cmmd-root", type=Path, default=None, help="raw CMMD download (DICOMs + clinical table)")
    a = ap.parse_args(argv)
    if (a.data_root / "miniddsm" / "index.csv").exists():
        miniddsm_audit(a.data_root / "miniddsm", a.out, a.ddsm_overlays)
    if (a.data_root / "cmmd" / "index.csv").exists():
        cmmd_audit(a.data_root / "cmmd", a.cmmd_root)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "labelaudit.txt").write_text("\n".join(OUT_LINES) + "\n")
    print(f"[done] {a.out / 'labelaudit.txt'}")


if __name__ == "__main__":
    main()