"""Summarise runs.

    python -m roimammo.analyze select --runs runs --dataset miniddsm --backbone effb0
        Development (cv) runs: mean +- SD over folds of the held-out AUC at the last epoch, per
        configuration; heatmap of the ROI grid; learning curves; writes chosen.json with the
        best ROI setting (by mean CV AUC), used for the final runs.

    python -m roimammo.analyze final --runs runs --out results --level breast --margin 0.01
        Final runs (locked test set): per-run metrics; for every comparison the seed-averaged
        AUC difference with a patient-level cluster bootstrap (95 % CI, two-sided p, Holm
        correction), an equivalence test (90 % CI inside +-margin), a paired DeLong test on
        seed-averaged probabilities (secondary), a forest plot and an efficiency table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import aggregate, all_metrics, delong_paired, holm, paired_cluster_bootstrap


def load_runs(root: Path, mode: str, study: str = "main") -> list[dict]:
    """Finished runs of one mode. study="main": the pre-registered runs only (default);
    study="all": main and addendum runs together (sensitivity analyses)."""
    runs = []
    for c in sorted(root.glob("*/config.json")):
        d = c.parent
        if not (d / "DONE").exists():
            continue
        cfg = json.load(open(c))
        if cfg["mode"] != mode:
            continue
        if cfg.get("tag") or cfg.get("max_steps"):  # timing/test runs (e.g. --tag _timing) never enter results
            continue
        if study == "main" and cfg.get("study", "main") != "main":
            continue
        cfg["dir"] = d
        cfg["dataset"] = Path(cfg["data_dir"]).name
        a = cfg["arm"]
        if a in ("roi", "randcrop"):
            bank = Path(cfg["bank"]).stem.replace("roi_bank_", "")
            fill = ",fill" if cfg.get("crop_fit", "letterbox") == "fill" else ""
            a = f"{a}[p={cfg['p_crop']:g},j={cfg['jitter']:g},K={cfg['roi_k']},{bank}{fill}]"
        cfg["arm_label"] = a
        lr = float(cfg.get("lr", 2e-4))
        cfg["model"] = f"{cfg['backbone']}/{cfg['head']}/{cfg['unit']}" + (f"@lr{lr:g}" if lr != 2e-4 else "")
        cfg["dirs"] = [d]
        runs.append(cfg)
    return runs


def pool_oof(runs: list[dict]) -> list[dict]:
    """Merge out-of-fold final runs (one per fold) into one logical run per configuration and seed.
    A configuration is used only if every fold of the dataset's split is present."""
    out, groups = [], {}
    for r in runs:
        if r.get("oof_fold") is None:
            out.append(r)
        else:
            groups.setdefault((r["dataset"], r["model"], r["arm_label"], r["seed"]), []).append(r)
    for key, rs in groups.items():
        sp = pd.read_csv(Path(rs[0]["data_dir"]) / "split.csv")
        need, have = set(sp.fold[sp.fold >= 0].astype(int)), {int(r["oof_fold"]) for r in rs}
        if have != need:
            print(f"[warn] {key}: out-of-fold runs incomplete (have folds {sorted(have)}, need {sorted(need)}); skipped")
            continue
        rs = sorted(rs, key=lambda r: r["oof_fold"])
        m = dict(rs[0])
        m["dirs"] = [r["dir"] for r in rs]
        m["eval_set"] = f"out-of-fold ({len(rs)} folds)"
        out.append(m)
    return out


def unit_preds(run: dict, level: str, fname="preds.csv") -> pd.DataFrame:
    p = pd.concat([pd.read_csv(d / fname) for d in run.get("dirs", [run["dir"]])], ignore_index=True)
    if level == "unit":
        return p.rename(columns={"id": "key"})[["key", "patient_id", "label", "prob"]]
    return aggregate(p, level)[["key", "patient_id", "label", "prob"]]


# ----------------------------------------------------------------------------- select (cv)

def cmd_select(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in load_runs(a.runs, "cv") if r["dataset"] == a.dataset and r["backbone"] == a.backbone
            and r["head"] == a.head and r["unit"] == a.unit]
    if not runs:
        raise SystemExit("no finished cv runs match")
    rows = []
    for r in runs:
        m = all_metrics(*unit_preds(r, a.level)[["label", "prob"]].T.to_numpy())
        rows.append({"arm_label": r["arm_label"], "arm": r["arm"], "p_crop": r["p_crop"], "jitter": r["jitter"],
                     "roi_k": r["roi_k"], "bank": r.get("bank"), "fold": r["fold"], "seed": r["seed"],
                     "bank_variant": Path(str(r.get("bank"))).stem.replace("roi_bank_", "") if r.get("bank") else None, **m})
    df = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / f"cv_runs_{a.dataset}_{a.backbone}.csv", index=False)
    s = df.groupby("arm_label").agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"), pr_auc_mean=("pr_auc", "mean"),
                                     n_runs=("auc", "size"), folds=("fold", "nunique")).sort_values("auc_mean", ascending=False)
    s.to_csv(a.out / f"cv_summary_{a.dataset}_{a.backbone}.csv")
    print(s.to_string(float_format=lambda v: f"{v:.4f}"))

    # selection grid (exactly the dev-plan grid, so ablation runs never enter the choice):
    # box scoring x p x K at the grid jitter
    from .sweep import GRID_BANKS, GRID_K, GRID_P, GRID_JITTER
    roi = df[(df.arm == "roi") & np.isclose(df.jitter.astype(float), GRID_JITTER) & df.bank_variant.isin(GRID_BANKS)
             & df.p_crop.astype(float).round(4).isin([round(x, 4) for x in GRID_P]) & df.roi_k.isin(GRID_K)]
    if not roi.empty:
        cnt = roi.groupby(["bank_variant", "p_crop", "roi_k"]).fold.nunique()
        if (cnt < cnt.max()).any():
            print(f"[warn] some grid cells have fewer folds than others:\n{cnt.to_string()}")
        means = roi.groupby(["bank_variant", "p_crop", "roi_k"]).auc.mean()
        best = means.idxmax()
        chosen = {"bank": best[0], "p_crop": float(best[1]), "roi_k": int(best[2]), "jitter": float(GRID_JITTER),
                  "selected_on": f"mean {a.level}-level AUC over cv folds", "dataset": a.dataset, "backbone": a.backbone,
                  "grid_mean_auc": {f"{v},p={p:g},K={k}": round(x, 5) for (v, p, k), x in means.items()}}
        json.dump(chosen, open(a.out / f"chosen_{a.dataset}.json", "w"), indent=1)
        print(f"[chosen] bank={chosen['bank']} p_crop={chosen['p_crop']:g} K={chosen['roi_k']} jitter={chosen['jitter']:g}")
        banks = list(dict.fromkeys(means.index.get_level_values(0)))
        base = df[df.arm == "none"].auc.mean()
        vmin, vmax = means.min(), means.max()
        fig, axes = plt.subplots(1, len(banks), figsize=(3.4 * len(banks) + 0.8, 3.3), squeeze=False)
        for ax, v in zip(axes[0], banks):
            grid = means.loc[v].unstack()
            im = ax.imshow(grid.to_numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(grid.shape[1]), [f"{x:g}" for x in grid.columns])
            ax.set_yticks(range(grid.shape[0]), [f"{x:g}" for x in grid.index])
            ax.set_xlabel("number of ROI boxes $K$")
            if v == banks[0]:
                ax.set_ylabel("$p_{ROI}$")
            ax.set_title({"blend": "texture saliency (blend)", "intensity": "brightness"}.get(v, v), fontsize=9)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    x = grid.iat[i, j]
                    ax.text(j, i, f"{x:.3f}\n({x - base:+.3f})" if np.isfinite(base) else f"{x:.3f}",
                            ha="center", va="center", fontsize=8, color="w")
        fig.colorbar(im, ax=list(axes[0]), label=f"mean CV {a.level} AUC (diff. vs standard)")
        fig.savefig(a.out / f"cv_grid_{a.dataset}_{a.backbone}.pdf", bbox_inches="tight")
        fig.savefig(a.out / f"cv_grid_{a.dataset}_{a.backbone}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # learning curves (fold-mean held-out AUC per epoch), standard vs every arm
    curves = []
    for r in runs:
        for f in sorted(r["dir"].glob("preds_ep*.csv")):
            ep = int(f.stem[-2:])
            m = all_metrics(*unit_preds(r, a.level, f.name)[["label", "prob"]].T.to_numpy())
            curves.append({"arm_label": r["arm_label"], "fold": r["fold"], "epoch": ep, "auc": m["auc"]})
    if curves:
        c = pd.DataFrame(curves)
        keep = [x for x in ("none", "rrc", "cutmix") if x in set(c.arm_label)]
        cj = a.out / f"chosen_{a.dataset}.json"
        if cj.exists():  # the selected ROI setting
            ch = json.load(open(cj))
            lab = f"roi[p={ch['p_crop']:g},j={ch['jitter']:g},K={ch['roi_k']},{ch['bank']}]"
            if lab in set(c.arm_label):
                keep.append(lab)
        names = {"none": "standard", "rrc": "RandomResizedCrop", "cutmix": "CutMix"}
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for lab in keep:
            g = c[c.arm_label == lab].groupby("epoch").auc
            mu, sd = g.mean(), g.std()
            ax.plot(mu.index, mu, label=names.get(lab, "ROI crop (selected) " + lab[3:] if lab.startswith("roi[") else lab), lw=1.4)
            ax.fill_between(mu.index, mu - sd, mu + sd, alpha=0.15)
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"held-out {a.level} AUC (mean $\\pm$ SD over folds)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(a.out / f"cv_curves_{a.dataset}_{a.backbone}.pdf")
        fig.savefig(a.out / f"cv_curves_{a.dataset}_{a.backbone}.png", dpi=200)
        plt.close(fig)


# ----------------------------------------------------------------------------- final (test)

def comparisons_for(arms: list[str]) -> list[tuple[str, str]]:
    comps = [(x, "none") for x in arms if x != "none" and "none" in arms]
    for x in arms:
        if x.startswith("roi["):
            twin = "randcrop[" + x[4:]
            if twin in arms:
                comps.append((x, twin))
            if "rrc" in arms:
                comps.append((x, "rrc"))
    return comps


def cmd_select_lr(a):
    """Addendum rule (ANALYSIS_PLAN.md, addendum A2): among the learning rates run with standard
    augmentation on all development folds, take the one with the highest mean breast-level AUC.
    The final training loss is reported so that non-converged settings are visible."""
    runs = [r for r in load_runs(a.runs, "cv", study="all") if r["dataset"] == a.dataset
            and r["backbone"] == a.backbone and r["arm"] == "none" and r["head"] == "gap" and r["unit"] == "image"]
    if not runs:
        raise SystemExit("no finished cv runs for this backbone")
    rows = []
    for r in runs:
        m = all_metrics(*unit_preds(r, a.level)[["label", "prob"]].T.to_numpy())
        lg = pd.read_csv(r["dir"] / "log.csv")
        rows.append({"lr": float(r.get("lr", 2e-4)), "fold": r["fold"], "auc": m["auc"],
                     "final_loss": float(lg.loss.iloc[-1]), "first_loss": float(lg.loss.iloc[0])})
    df = pd.DataFrame(rows)
    nf = df.groupby("lr").fold.nunique()
    complete = nf[nf == nf.max()].index
    s = df[df.lr.isin(complete)].groupby("lr").agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                                                   final_loss_mean=("final_loss", "mean"),
                                                   first_loss_mean=("first_loss", "mean"), folds=("fold", "nunique"))
    print(s.to_string(float_format=lambda v: f"{v:.4f}"))
    best = float(s.auc_mean.idxmax())
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / f"lr_runs_{a.dataset}_{a.backbone}.csv", index=False)
    json.dump({"backbone": a.backbone, "lr": best, "selected_on": f"mean {a.level}-level AUC, standard augmentation, "
               f"{int(s.folds.max())} development folds", "grid": {f"{k:g}": round(v, 5) for k, v in s.auc_mean.items()}},
              open(a.out / f"chosen_lr_{a.backbone}.json", "w"), indent=1)
    print(f"[chosen] {a.backbone} lr={best:g}")


def _unit_preds_excluding(exclude: Path | None, level: str):
    """unit_preds, optionally without the breasts listed in a CSV (column breast_id); used for the
    label-audit sensitivity analysis. At patient level, patients with an excluded breast are dropped."""
    if exclude is None:
        return unit_preds
    ex = pd.read_csv(exclude)
    bids, pids = set(ex.breast_id.astype(str)), set(ex.patient_id.astype(str))
    print(f"[info] excluding {len(bids)} breasts ({len(pids)} patients) listed in {exclude}")

    def up(run, lvl, fname="preds.csv"):
        u = globals()["unit_preds"](run, lvl, fname)
        if lvl == "breast":
            return u[~u.key.astype(str).isin(bids)].reset_index(drop=True)
        return u[~u.patient_id.astype(str).isin(pids)].reset_index(drop=True)
    return up


def cmd_final(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = pool_oof(load_runs(a.runs, "final", a.study))
    if not runs:
        raise SystemExit("no finished final runs")
    unit_preds = _unit_preds_excluding(a.exclude, a.level)
    a.out.mkdir(parents=True, exist_ok=True)
    per_run, comp_rows, eff = [], [], []
    for r in runs:
        u = unit_preds(r, a.level)
        per_run.append({"dataset": r["dataset"], "model": r["model"], "arm": r["arm_label"], "seed": r["seed"],
                        "eval_set": r["eval_set"], **all_metrics(u.label, u.prob)})
        lg = pd.concat([pd.read_csv(d / "log.csv") for d in r["dirs"]], ignore_index=True)
        steady = lg[lg.epoch >= 2] if len(lg) > 1 else lg
        eff.append({"dataset": r["dataset"], "model": r["model"], "arm": r["arm_label"], "seed": r["seed"],
                    "train_img_per_s": steady.train_img_per_s.median(),
                    "data_wait_frac": (steady.data_wait_s / steady.train_s).median(),
                    "peak_train_mib": lg.peak_alloc_mib.max(), "eval_img_per_s": lg.eval_img_per_s.dropna().median()})
    pr = pd.DataFrame(per_run)
    pr.to_csv(a.out / f"final_runs_{a.level}.csv", index=False)
    summ = pr.groupby(["dataset", "model", "arm"]).agg(
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"), pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
        brier_mean=("brier", "mean"), ece_mean=("ece", "mean"), seeds=("seed", "nunique"), n=("n", "first"),
        n_pos=("n_pos", "first")).reset_index()
    summ.to_csv(a.out / f"final_summary_{a.level}.csv", index=False)
    pd.DataFrame(eff).groupby(["dataset", "model", "arm"]).median(numeric_only=True).drop(columns="seed").to_csv(
        a.out / "efficiency.csv")

    for (ds, model), grp in pd.DataFrame(runs).groupby(["dataset", "model"]):
        by_arm = {k: g.sort_values("seed").to_dict("records") for k, g in grp.groupby("arm_label")}
        for arm_x, arm_ref in comparisons_for(sorted(by_arm)):
            rx, rr = by_arm[arm_x], by_arm[arm_ref]
            seeds = sorted({r["seed"] for r in rx} & {r["seed"] for r in rr})
            px = {r["seed"]: unit_preds(r, a.level) for r in rx if r["seed"] in seeds}
            pref = {r["seed"]: unit_preds(r, a.level) for r in rr if r["seed"] in seeds}
            base = px[seeds[0]][["key", "patient_id", "label"]].sort_values("key").reset_index(drop=True)
            def aligned(d):
                return [d[s].set_index("key").loc[base.key, "prob"].to_numpy() for s in seeds]
            ax_, ar_ = aligned(px), aligned(pref)
            per_seed = [all_metrics(base.label, x)["auc"] - all_metrics(base.label, y)["auc"] for x, y in zip(ax_, ar_)]
            boot = paired_cluster_bootstrap(base, ax_, ar_, n_boot=a.n_boot, seed=0)
            d = float(np.mean(per_seed))
            lo95, hi95 = np.percentile(boot, [2.5, 97.5])
            lo90, hi90 = np.percentile(boot, [5, 95])
            nb = len(boot)  # add-one correction: a bootstrap p-value is never exactly 0
            p_boot = float(min(1.0, 2 * min(((boot <= 0).sum() + 1) / (nb + 1), ((boot >= 0).sum() + 1) / (nb + 1))))
            dl = delong_paired(base.label, np.mean(ax_, 0), np.mean(ar_, 0))
            comp_rows.append({"dataset": ds, "model": model, "arm": arm_x, "reference": arm_ref, "seeds": len(seeds),
                              "delta_auc": d, "delta_sd_over_seeds": float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else np.nan,
                              "ci95_lo": lo95, "ci95_hi": hi95, "ci90_lo": lo90, "ci90_hi": hi90, "p_boot": p_boot,
                              "equivalent_within_margin": bool(lo90 > -a.margin and hi90 < a.margin),
                              "delong_delta": dl[2], "delong_p": dl[5], "n_units": len(base),
                              "n_patients": base.patient_id.nunique()})
    comp = pd.DataFrame(comp_rows)
    if comp.empty:
        print(summ.to_string(index=False))
        print("[info] no comparisons (need 'none' or matching randcrop runs)")
        return
    comp["p_holm"] = holm(comp.p_boot)
    comp.to_csv(a.out / f"comparisons_{a.level}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()
        print(comp[["dataset", "model", "arm", "reference", "delta_auc", "ci95_lo", "ci95_hi", "p_holm",
                    "equivalent_within_margin"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    forest_plot(comp.reset_index(drop=True), a.out / f"forest_{a.level}", a.level, a.margin)
    print(f"[done] results in {a.out}")


PRETTY = {"none": "Standard", "rrc": "RandomResizedCrop", "cutmix": "CutMix", "roi": "ROI crop",
          "randcrop": "Random crop"}
MODEL_PRETTY = {"effb0": "EfficientNet-B0", "resnet50": "ResNet-50", "convnext_t": "ConvNeXt-T"}


def pretty_arm(label: str, all_labels) -> str:
    base = label.split("[")[0]
    name = PRETTY.get(base, base)
    variants = {x.split("[", 1)[1] for x in all_labels if x.startswith(base + "[")}
    if "[" in label and len(variants) > 1:
        name += " [" + label.split("[", 1)[1]
    return name


def pretty_model(m: str) -> str:
    m, _, lr = m.partition("@")
    bb, head, unit = m.split("/")
    s = MODEL_PRETTY.get(bb, bb)
    if lr:
        s += f" ({lr.replace('lr', 'lr ')})"
    if head == "attnmil":
        s += " + attention-MIL"
    if unit == "breast":
        s += " (two-view)"
    return s


def forest_plot(comp: pd.DataFrame, stem: Path, level: str, margin: float):
    """Main figure: one panel per dataset; rows = comparisons; one marker per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    labels = set(comp.arm) | set(comp.reference)
    comp = comp.assign(row=[f"{pretty_arm(r.arm, labels)} vs {pretty_arm(r.reference, labels)}" for r in comp.itertuples()],
                       mlabel=comp.model.map(pretty_model))
    rows = list(dict.fromkeys(comp.row))
    models = list(dict.fromkeys(comp.mlabel))
    datasets = sorted(dict.fromkeys(comp.dataset), key=lambda d: (d != "miniddsm", d))
    markers, colors = "osD^vP", plt.cm.tab10.colors
    off = np.linspace(-0.25, 0.25, len(models)) if len(models) > 1 else [0.0]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets) + 1.8, 0.42 * len(rows) + 1.4),
                             sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        ax.axvspan(-margin, margin, color="0.88", zorder=0)
        ax.axvline(0, color="0.3", lw=0.8)
        for mi, m in enumerate(models):
            d = comp[(comp.dataset == ds) & (comp.mlabel == m)]
            y = np.array([len(rows) - 1 - rows.index(r) for r in d.row]) + off[mi]
            ax.errorbar(d.delta_auc, y, xerr=[d.delta_auc - d.ci95_lo, d.ci95_hi - d.delta_auc], fmt=markers[mi % 6],
                        ms=4, color=colors[mi % 10], capsize=1.5, lw=1, label=m)
        ax.set_title({"miniddsm": "Mini-DDSM", "cmmd": "CMMD"}.get(ds, ds))
        ax.set_xlabel(f"$\\Delta$ {level}-level AUC (95% CI)")
        ax.grid(axis="x", alpha=0.3)
    axes[0][0].set_yticks(range(len(rows)), rows[::-1])
    h, l = axes[0][0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    h.append(Patch(color="0.88"))
    l.append(f"equivalence margin $\\pm${margin:g}")
    fig.legend(h, l, loc="upper center", ncol=min(4, len(l)), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--runs", type=Path, default=Path("runs"))
    s.add_argument("--dataset", required=True)
    s.add_argument("--backbone", default="effb0")
    s.add_argument("--head", default="gap")
    s.add_argument("--unit", default="image")
    s.add_argument("--level", default="breast", choices=["unit", "breast", "patient"])
    s.add_argument("--max-curves", type=int, default=5)
    s.add_argument("--out", type=Path, default=Path("results"))
    s2 = sub.add_parser("select-lr", help="addendum: choose a backbone's learning rate on development folds")
    s2.add_argument("--runs", type=Path, default=Path("runs"))
    s2.add_argument("--dataset", default="miniddsm")
    s2.add_argument("--backbone", default="convnext_t")
    s2.add_argument("--level", default="breast", choices=["unit", "breast", "patient"])
    s2.add_argument("--out", type=Path, default=Path("results"))
    f = sub.add_parser("final")
    f.add_argument("--study", default="main", choices=["main", "all"],
                   help="main: pre-registered runs only; all: include addendum runs (sensitivity analyses)")
    f.add_argument("--runs", type=Path, default=Path("runs"))
    f.add_argument("--level", default="breast", choices=["unit", "breast", "patient"])
    f.add_argument("--margin", type=float, default=0.01)
    f.add_argument("--n-boot", type=int, default=2000)
    f.add_argument("--out", type=Path, default=Path("results"))
    f.add_argument("--exclude", type=Path, default=None,
                   help="CSV of breasts (breast_id, patient_id) to leave out (label-audit sensitivity analysis)")
    a = ap.parse_args(argv)
    {"select": cmd_select, "select-lr": cmd_select_lr, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    main()
