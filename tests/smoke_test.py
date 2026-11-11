"""End-to-end check on synthetic data (CPU, ~3-5 minutes). Run from the repository root:
    python tests/smoke_test.py
It builds fake Mini-DDSM and CMMD trees, then runs prepare -> splits -> ROI bank -> coverage ->
cv training -> selection -> final runs -> analysis -> figures, and checks the outputs exist.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from make_fake_data import fake_cmmd, fake_miniddsm  # noqa: E402


def sh(*args):
    cmd = [sys.executable, "-m", *map(str, args)]
    print(">>", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    t = Path(tempfile.mkdtemp(prefix="roimammo_smoke_"))
    fake_miniddsm(t / "raw_mini")
    fake_cmmd(t / "raw_cmmd")
    data, runs, res = t / "data", t / "runs", t / "results"
    tiny = ["--img-h", 128, "--img-w", 96, "--epochs", 2, "--batch", 4, "--workers", 0, "--pretrained", 0, "--cpu"]
    sh("roimammo.prepare", "--dataset", "miniddsm", "--root", t / "raw_mini", "--out", data / "miniddsm", "--workers", 2)
    sh("roimammo.prepare", "--dataset", "cmmd", "--root", t / "raw_cmmd", "--out", data / "cmmd", "--workers", 2)
    sh("roimammo.splits", "--data-dir", data / "miniddsm", "--folds", 2, "--test-frac", 0.25)
    sh("roimammo.splits", "--data-dir", data / "cmmd", "--folds", 2, "--test-frac", 0)  # out-of-fold design
    for ds in ("miniddsm", "cmmd"):
        sh("roimammo.roibank", "--data-dir", data / ds, "--workers", 2, "--overlays", 4)
        sh("roimammo.roibank", "--data-dir", data / ds, "--variant", "intensity", "--workers", 2, "--overlays", 0)
    sh("roimammo.roibank", "--data-dir", data / "miniddsm", "--variant", "random", "--workers", 2, "--overlays", 0)
    sh("roimammo.qc", "--data-dir", data / "miniddsm")
    sh("roimammo.coverage", "--data-dir", data / "miniddsm", "--subset", "dev", "--banks", data / "miniddsm" / "roi_bank_blend.json",
       data / "miniddsm" / "roi_bank_intensity.json", data / "miniddsm" / "roi_bank_random.json", "--repeats", 3)
    sh("roimammo.sweep", "--plan", "dev", "--data-root", data, "--runs", runs, "--folds", 2,
       "--filter", "--jitter 0.1", "--", *tiny)
    sh("roimammo.sweep", "--plan", "dev", "--data-root", data, "--runs", runs, "--folds", 2,
       "--filter", "--arm none", "--", *tiny)
    sh("roimammo.analyze", "select", "--runs", runs, "--dataset", "miniddsm", "--out", res)
    sh("roimammo.sweep", "--plan", "final", "--data-root", data, "--runs", runs, "--results", res, "--seeds", 1, 2,
       "--filter", "--backbone effb0", "--", *tiny)
    for p in ("ablation", "extra"):  # plan generation only
        sh("roimammo.sweep", "--plan", p, "--data-root", data, "--runs", runs, "--results", res, "--folds", 2, "--dry-run")
    sh("roimammo.analyze", "final", "--runs", runs, "--out", res, "--n-boot", 200)
    sh("roimammo.sitecheck", "--runs", runs, "--data-root", data, "--dataset", "miniddsm", "--out", res)
    # addendum: ConvNeXt-T learning-rate choice, crop-fill sensitivity, extra out-of-fold seeds
    sh("roimammo.sweep", "--plan", "addendum-tune", "--data-root", data, "--runs", runs, "--folds", 2, "--", *tiny)
    sh("roimammo.analyze", "select-lr", "--runs", runs, "--out", res)
    sh("roimammo.sweep", "--plan", "addendum", "--data-root", data, "--runs", runs, "--results", res, "--seeds", 1, 2,
       "--dry-run")
    sh("roimammo.sweep", "--plan", "addendum", "--data-root", data, "--runs", runs, "--results", res, "--seeds", 1, 2,
       "--datasets", "miniddsm", "--filter", "--backbone effb0", "--", *tiny)
    sh("roimammo.sweep", "--plan", "addendum", "--data-root", data, "--runs", runs, "--results", res, "--seeds", 1, 2,
       "--datasets", "cmmd", "--filter", "--seed 2", "--", *tiny)
    before = (res / "comparisons_breast.csv").read_bytes()
    sh("roimammo.analyze", "final", "--runs", runs, "--out", res, "--n-boot", 200)  # main analysis must ignore addendum runs
    if (res / "comparisons_breast.csv").read_bytes() != before:
        sys.exit("SMOKE TEST FAILED: addendum runs changed the main analysis")
    sh("roimammo.analyze", "final", "--study", "all", "--runs", runs, "--out", res / "addendum", "--n-boot", 200)
    sh("roimammo.figures", "examples", "--data-dir", data / "miniddsm", "--bank", data / "miniddsm" / "roi_bank_blend.json", "--out", res)
    sh("roimammo.figures", "arms", "--data-dir", data / "miniddsm", "--bank", data / "miniddsm" / "roi_bank_blend.json", "--out", res)
    need = ["chosen_miniddsm.json", "comparisons_breast.csv", "forest_breast.pdf", "final_summary_breast.csv",
            "efficiency.csv", "fig_examples_miniddsm.pdf", "fig_arms_miniddsm.pdf", "cv_grid_miniddsm_effb0.pdf",
            "chosen_lr_convnext_t.json", "addendum/comparisons_breast.csv"]
    missing = [n for n in need if not (res / n).exists()]
    if missing:
        sys.exit(f"SMOKE TEST FAILED, missing: {missing}")
    print(f"\nSMOKE TEST PASSED. Outputs in {t}")


if __name__ == "__main__":
    main()
