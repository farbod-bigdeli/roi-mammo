"""Training loss per arm (added after the third review; reads runs/*/log.csv only).

For every final run: the class-weighted training loss per epoch, summarised per dataset, backbone,
learning rate and arm (epochs 1, 5 and 10; mean, min and max over seeds and folds), and a figure of
the mean loss curves per arm for EfficientNet-B0 and ResNet-50. The loss of a constant prediction
at the class-weighted optimum is printed for reference.

    python -m roimammo.losstable --runs runs --out results/losstable
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import load_runs


def trivial_loss(pi: float, w: float) -> float:
    """Weighted BCE of the best constant prediction (pos_weight w, prevalence pi), per sample."""
    p = w * pi / (w * pi + 1 - pi)
    return float(-(w * pi * np.log(p) + (1 - pi) * np.log(1 - p)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results/losstable"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    for r in load_runs(a.runs, "final", "control"):
        for d in r.get("dirs", [r["dir"]]):
            d = Path(d)
            c = json.load(open(d / "config.json"))
            log = pd.read_csv(d / "log.csv")
            if log.empty:
                continue
            key = dict(dataset=r["dataset"], model=r["model"], lr=c["lr"], arm=r["arm_label"], seed=c["seed"],
                       fold=c.get("fold"), pos_weight=c.get("pos_weight_value"))
            rows.append({**key, "loss_ep1": log.loss.iloc[0], "loss_ep5": log.loss.iloc[min(4, len(log) - 1)],
                         "loss_final": log.loss.iloc[-1], "epochs": len(log)})
            for e, l in zip(log.epoch, log.loss):
                curves.append({**key, "epoch": e, "loss": l})
    d = pd.DataFrame(rows)
    d.to_csv(a.out / "loss_runs.csv", index=False)
    g = d.groupby(["dataset", "model", "lr", "arm"]).agg(
        runs=("loss_final", "size"), ep1=("loss_ep1", "mean"), ep5=("loss_ep5", "mean"),
        final=("loss_final", "mean"), final_min=("loss_final", "min"), final_max=("loss_final", "max")).round(3)
    g.to_csv(a.out / "loss_summary.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(g.to_string())
    cv = pd.DataFrame(curves)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        models = [m for m in ("effb0/gap/image", "resnet50/gap/image") if m in set(cv.model)]
        dss = sorted(set(cv.dataset))
        fig, axes = plt.subplots(len(dss), len(models), figsize=(5 * len(models), 3.6 * len(dss)), squeeze=False)
        for i, ds in enumerate(dss):
            for j, m in enumerate(models):
                ax = axes[i][j]
                q = cv[(cv.dataset == ds) & (cv.model == m) & (cv.lr == 2e-4)]
                for arm, qq in q.groupby("arm"):
                    mc = qq.groupby("epoch").loss.mean()
                    ax.plot(mc.index, mc.values, marker="o", ms=3, label=arm)
                ax.set_title(f"{ds}, {m.split('/')[0]}", fontsize=9)
                ax.set_xlabel("epoch")
                ax.set_ylabel("weighted training loss")
                ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(a.out / "loss_curves.png", dpi=150)
    except Exception as e:  # the table is the result; the figure is a convenience
        print("[warn] figure not drawn:", e)
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
