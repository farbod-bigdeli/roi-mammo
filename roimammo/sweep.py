"""Run the study's experiment plans (sequentially, resumable, shardable across machines).

Plans
    dev       Mini-DDSM, EfficientNet-B0, 4 CV folds, seed 1:
              none, rrc, cutmix, and the ROI grid: box scoring in {blend (texture), intensity}
              x p in {0.1, 0.25} x K in {1, 5} (jitter 0.1)
              -> then: python -m roimammo.analyze select --dataset miniddsm   (writes chosen_miniddsm.json)
    ablation  Mini-DDSM, EfficientNet-B0, 4 CV folds, at the chosen bank, p and K:
              random-crop control; jitter 0 and 0.2; the other K values of {1, 3, 5, 10};
              the other box-scoring variants of {blend, intensity, var, log, random}
    final     every dataset x backbone (effb0, resnet50, convnext_t):
              effb0: none, rrc, cutmix, randcrop, roi; other backbones: none, randcrop, roi
    extra     effb0: attention-MIL head and two-view (CC+MLO) model, each with none and roi
Addendum (ANALYSIS_PLAN.md, added after the main results; all runs carry --study addendum and
never enter the pre-registered main analysis):
    addendum-tune  ConvNeXt-T learning rate in {2e-4, 1e-4, 5e-5}, standard augmentation,
                   Mini-DDSM development folds, seed 1
                   -> then: python -m roimammo.analyze select-lr   (writes chosen_lr_convnext_t.json)
    addendum       A2 ConvNeXt-T none/randcrop/roi at the chosen learning rate (all datasets)
                   A1 crop-fill: effb0 and resnet50, randcrop and roi with --crop-fit fill
                   A3 out-of-fold datasets: seeds 2 and 3 for effb0 (5 arms) and resnet50 (3 arms)
Addendum B (ANALYSIS_PLAN.md; runs carry --study control):
    control        effb0 and resnet50, ROI arm at the chosen p, K and jitter with the random-score
                   bank ("central random box"); Mini-DDSM seeds 1-3, CMMD seeds 1-3 x 5 folds
                   -> then: python -m roimammo.controlarm
Final/extra evaluation: datasets with a locked test set (Mini-DDSM) use seeds 1-3, one run per
seed. Datasets without a test set (CMMD, splits made with --test-frac 0) are evaluated
out-of-fold: one run per fold with seed 1 (--oof-seeds), pooled by the analysis.
The ROI setting chosen on Mini-DDSM development folds is applied unchanged everywhere.

Usage
    python -m roimammo.sweep --plan dev --dry-run
    python -m roimammo.sweep --plan final --datasets miniddsm cmmd --shard 0/2
Extra arguments after "--" are passed to every training run, e.g.  -- --batch 4 --accum 4 --grad-ckpt 1
"""
from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

GRID_BANKS = ("blend", "intensity")
GRID_P = (0.1, 0.25)
GRID_K = (1, 5)
GRID_JITTER = 0.1
ABL_JITTER = (0.0, 0.2)
ABL_K = (1, 3, 5, 10)
ABL_BANKS = ("blend", "intensity", "var", "log", "random")
ADD_LRS = (2e-4, 1e-4, 5e-5)      # addendum A2: ConvNeXt-T learning-rate grid
ADD_OOF_SEEDS = (2, 3)            # addendum A3: extra seeds for out-of-fold datasets
ADD_CONTROL_SEEDS = (1, 2, 3)     # addendum B: seeds for out-of-fold datasets


def chosen(results: Path, dataset="miniddsm") -> dict:
    f = results / f"chosen_{dataset}.json"
    if not f.exists():
        sys.exit(f"{f} missing: run the dev plan and 'python -m roimammo.analyze select --dataset {dataset}' first")
    return json.load(open(f))


def eval_units(data_dir: Path, seeds, oof_seeds):
    """(seed, fold) pairs for final runs: seeds x [None] with a test set, oof_seeds x folds without."""
    sp = pd.read_csv(data_dir / "split.csv")
    if (sp.split == "test").any():
        return [(s, None) for s in seeds]
    folds = sorted(sp.fold[sp.fold >= 0].unique())
    return [(s, int(f)) for s in oof_seeds for f in folds]


def plan(name: str, a) -> list[list[str]]:
    cmds = []

    def run(ds, *extra):
        return [sys.executable, "-m", "roimammo.train", "--data-dir", str(a.data_root / ds),
                "--out-root", str(a.runs), *map(str, extra)]

    def bank(ds, v="blend"):
        return a.data_root / ds / f"roi_bank_{v}.json"

    if name == "dev":
        ds = "miniddsm"
        for fold in range(a.folds):
            base = ["--mode", "cv", "--fold", fold, "--backbone", "effb0", "--seed", 1]
            for arm in ("none", "rrc", "cutmix"):
                cmds.append(run(ds, *base, "--arm", arm))
            for v, p, k in itertools.product(GRID_BANKS, GRID_P, GRID_K):
                cmds.append(run(ds, *base, "--arm", "roi", "--bank", bank(ds, v), "--p-crop", p,
                                "--jitter", GRID_JITTER, "--roi-k", k))
    elif name == "ablation":
        ds, c = "miniddsm", chosen(a.results)
        for fold in range(a.folds):
            base = ["--mode", "cv", "--fold", fold, "--backbone", "effb0", "--seed", 1]
            at = ["--p-crop", c["p_crop"]]
            cb = c.get("bank", "blend")
            cmds.append(run(ds, *base, "--arm", "randcrop", "--bank", bank(ds, cb), *at,
                            "--jitter", c["jitter"], "--roi-k", c["roi_k"]))
            for j in ABL_JITTER:
                if j != c["jitter"]:
                    cmds.append(run(ds, *base, "--arm", "roi", "--bank", bank(ds, cb), *at, "--jitter", j, "--roi-k", c["roi_k"]))
            for k in ABL_K:
                if k not in GRID_K and k != c["roi_k"]:
                    cmds.append(run(ds, *base, "--arm", "roi", "--bank", bank(ds, cb), *at, "--jitter", c["jitter"], "--roi-k", k))
            for v in ABL_BANKS:
                if v in GRID_BANKS:
                    continue  # already in the selection grid
                cmds.append(run(ds, *base, "--arm", "roi", "--bank", bank(ds, v), *at, "--jitter", c["jitter"], "--roi-k", c["roi_k"]))
    elif name in ("final", "extra"):
        c = chosen(a.results)
        crop = ["--p-crop", c["p_crop"], "--jitter", c["jitter"], "--roi-k", c["roi_k"]]
        cb = c.get("bank", "blend")
        for ds in a.datasets:
            for seed, fold in eval_units(a.data_root / ds, a.seeds, a.oof_seeds):
                base = ["--mode", "final", "--seed", seed] + (["--fold", fold] if fold is not None else [])
                if name == "final":
                    for bb in ("effb0", "resnet50", "convnext_t"):
                        arms = ("none", "rrc", "cutmix", "randcrop", "roi") if bb == "effb0" else ("none", "randcrop", "roi")
                        for arm in arms:
                            x = ["--arm", arm] + (["--bank", bank(ds, cb), *crop] if arm in ("roi", "randcrop") else [])
                            cmds.append(run(ds, *base, "--backbone", bb, *x))
                else:
                    for variant in (["--head", "attnmil"], ["--unit", "breast"]):
                        for arm in ("none", "roi"):
                            x = ["--arm", arm] + (["--bank", bank(ds, cb), *crop] if arm == "roi" else [])
                            cmds.append(run(ds, *base, "--backbone", "effb0", *variant, *x))
    elif name == "addendum-tune":
        ds = "miniddsm"
        for fold in range(a.folds):
            for lr in ADD_LRS:
                cmds.append(run(ds, "--mode", "cv", "--fold", fold, "--backbone", "convnext_t", "--seed", 1,
                                "--arm", "none", "--lr", lr, "--study", "addendum"))
    elif name == "addendum":
        c = chosen(a.results)
        crop = ["--p-crop", c["p_crop"], "--jitter", c["jitter"], "--roi-k", c["roi_k"]]
        cb = c.get("bank", "blend")
        f = a.results / "chosen_lr_convnext_t.json"
        if not f.exists():
            sys.exit(f"{f} missing: run --plan addendum-tune and 'python -m roimammo.analyze select-lr' first")
        lr = json.load(open(f))["lr"]
        add = ["--study", "addendum"]
        for ds in a.datasets:  # A2: ConvNeXt-T at the chosen learning rate
            for seed, fold in eval_units(a.data_root / ds, a.seeds, a.oof_seeds):
                base = ["--mode", "final", "--seed", seed] + (["--fold", fold] if fold is not None else [])
                for arm in ("none", "randcrop", "roi"):
                    x = ["--arm", arm] + (["--bank", bank(ds, cb), *crop] if arm != "none" else [])
                    cmds.append(run(ds, *base, "--backbone", "convnext_t", "--lr", lr, *x, *add))
        for ds in a.datasets:  # A1: crops widened to the input aspect ratio
            for seed, fold in eval_units(a.data_root / ds, a.seeds, a.oof_seeds):
                base = ["--mode", "final", "--seed", seed] + (["--fold", fold] if fold is not None else [])
                for bb in ("effb0", "resnet50"):
                    for arm in ("randcrop", "roi"):
                        cmds.append(run(ds, *base, "--backbone", bb, "--arm", arm, "--bank", bank(ds, cb), *crop,
                                        "--crop-fit", "fill", *add))
        for ds in a.datasets:  # A3: more seeds where there is no test set
            sp = pd.read_csv(a.data_root / ds / "split.csv")
            if (sp.split == "test").any():
                continue
            for seed, fold in eval_units(a.data_root / ds, a.seeds, ADD_OOF_SEEDS):
                base = ["--mode", "final", "--seed", seed, "--fold", fold]
                for bb, arms in (("effb0", ("none", "rrc", "cutmix", "randcrop", "roi")),
                                 ("resnet50", ("none", "randcrop", "roi"))):
                    for arm in arms:
                        x = ["--arm", arm] + (["--bank", bank(ds, cb), *crop] if arm in ("roi", "randcrop") else [])
                        cmds.append(run(ds, *base, "--backbone", bb, *x, *add))
    elif name == "control":  # addendum B: ROI arm with the random-score bank (placement without saliency)
        c = chosen(a.results)
        crop = ["--p-crop", c["p_crop"], "--jitter", c["jitter"], "--roi-k", c["roi_k"]]
        for ds in a.datasets:
            if not bank(ds, "random").exists():
                sys.exit(f"{bank(ds, 'random')} missing: python -m roimammo.roibank --data-dir {a.data_root / ds} "
                         "--variant random --overlays 0")
            sp = pd.read_csv(a.data_root / ds / "split.csv")
            oof = ADD_CONTROL_SEEDS if not (sp.split == "test").any() else a.oof_seeds
            for seed, fold in eval_units(a.data_root / ds, a.seeds, oof):
                base = ["--mode", "final", "--seed", seed] + (["--fold", fold] if fold is not None else [])
                for bb in ("effb0", "resnet50"):
                    cmds.append(run(ds, *base, "--backbone", bb, "--arm", "roi", "--bank", bank(ds, "random"), *crop,
                                    "--study", "control"))
    else:
        sys.exit(f"unknown plan {name}")
    return [[str(t) for t in c] + a.passthrough for c in cmds]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    passthrough = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, choices=["dev", "ablation", "final", "extra", "addendum-tune", "addendum",
                                                       "control"])
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--datasets", nargs="+", default=["miniddsm", "cmmd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="for datasets with a test set")
    ap.add_argument("--oof-seeds", type=int, nargs="+", default=[1], help="for out-of-fold datasets")
    ap.add_argument("--folds", type=int, default=4, help="development folds (dev and ablation plans)")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th command starting at i")
    ap.add_argument("--filter", default="", help="only commands containing this text")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    a.passthrough = passthrough
    i, n = map(int, a.shard.split("/"))
    cmds = [c for c in plan(a.plan, a) if a.filter in " ".join(c)][i::n]
    print(f"[sweep] plan={a.plan} shard={a.shard}: {len(cmds)} runs", flush=True)
    for k, c in enumerate(cmds, 1):
        line = " ".join(shlex.quote(t) for t in c)
        print(f"[{k}/{len(cmds)}] {line}", flush=True)
        if not a.dry_run:
            r = subprocess.run(c)
            if r.returncode != 0:
                print(f"[warn] run failed with code {r.returncode}; continuing", flush=True)


if __name__ == "__main__":
    main()
