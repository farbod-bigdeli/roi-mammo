"""Package the per-image predictions and result tables for a data release (no images, no weights).

Collects, with their paths relative to the repository root:
    runs/<run>/preds.csv, preds_epNN.csv, log.csv, config.json, env.json, DONE   (every finished run)
    results/**.csv|.txt|.json                                                     (all result tables)
and writes MANIFEST.tsv (path, bytes, sha256) and README.txt into the zip. Model weights (*.pt),
images and figures are left out. Unzip in a clone of the code (which already holds the splits,
exclusions, ROI banks and QC files under data/) to rerun every analysis without training:

    python -m roimammo.analyze final --level breast
    python -m roimammo.analyze final --study all --out results/addendum

    python -m roimammo.package_release --out roi-mammo-results-v1.0.zip
"""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

RUN_FILES = ("preds.csv", "log.csv", "config.json", "env.json", "DONE")
RESULT_SUFFIXES = (".csv", ".txt", ".json")

README = """roi-mammo results v{version}

Per-image predictions and result tables for "Region-of-Interest Augmentation for Mammography
Classification under Patient-Level Cross-Validation" (arXiv:2509.20585, v3).
Code: https://github.com/farbod-bigdeli/roi-mammo (archived: {code_doi}).

Contents (paths relative to the repository root):
  runs/<run>/preds.csv        final predictions of each run (one row per image: image_id,
                              patient_id, breast_id, label, prob, ...)
  runs/<run>/preds_epNN.csv   development runs: predictions after each epoch
  runs/<run>/log.csv          training log; config.json: run settings; env.json: software versions
  results/                    all result tables (main analysis, addendum in results/addendum,
                              review checks in results/labelaudit, cropreplay, foldcheck,
                              foldcheck_main)
  MANIFEST.tsv                path, size and SHA-256 of every file

The splits, excluded patients and images, ROI banks and quality-control files are in the code
repository under data/. No images and no model weights are included. Mini-DDSM and CMMD must be
obtained from their sources (Kaggle and The Cancer Imaging Archive) to retrain.

To reproduce the tables without training: clone the code at tag v{version}, unzip this file in the
repository root, and run for example
    python -m roimammo.analyze final --level breast
    python -m roimammo.analyze final --study all --out results/addendum
Runs: {n_runs}; files: {n_files}.
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root: Path) -> tuple[list[Path], int]:
    files, n_runs = [], 0
    for run in sorted((root / "runs").iterdir()) if (root / "runs").is_dir() else []:
        if not (run.is_dir() and (run / "DONE").exists()):
            continue
        n_runs += 1
        files += [run / f for f in RUN_FILES if (run / f).exists()]
        files += sorted(run.glob("preds_ep*.csv"))
    if (root / "results").is_dir():
        files += sorted(p for p in (root / "results").rglob("*") if p.is_file() and p.suffix in RESULT_SUFFIXES)
    return files, n_runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("roi-mammo-results-v1.0.zip"))
    ap.add_argument("--version", default="1.0")
    ap.add_argument("--code-doi", default="https://doi.org/10.5281/zenodo.24326049")
    a = ap.parse_args(argv)
    files, n_runs = collect(a.root)
    if not files:
        raise SystemExit("nothing to package: run from the repository root (runs/ and results/ expected)")
    rows = []
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            rel = p.relative_to(a.root).as_posix()
            z.write(p, rel)
            rows.append(f"{rel}\t{p.stat().st_size}\t{sha256(p)}")
        z.writestr("MANIFEST.tsv", "path\tbytes\tsha256\n" + "\n".join(rows) + "\n")
        z.writestr("README.txt", README.format(version=a.version, code_doi=a.code_doi, n_runs=n_runs,
                                               n_files=len(files)))
    n_preds = sum(1 for p in files if p.name == "preds.csv")
    print(f"[done] {a.out}: {n_runs} runs ({n_preds} with preds.csv), {len(files)} files, "
          f"{a.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
