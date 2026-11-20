"""Addendum C: training to convergence (ANALYSIS_PLAN.md, addendum C; registered before its runs).

Three subcommands, one per step of the addendum:

    python -m roimammo.convergence select-lr    # after --plan convergence-lr   (Step 1a)
    python -m roimammo.convergence select-len   # after --plan convergence-len  (Step 1b)
    python -m roimammo.convergence analyze      # after --plan convergence      (Step 3), once only

select-lr   For each backbone, the learning rate with the higher mean breast-level AUC over the four
            Mini-DDSM development folds after 30 epochs with standard augmentation; rates within
            0.002 of the best are tied and the lower rate is taken.
select-len  At the chosen rate, separate 30- and 50-epoch runs of the four arms on development folds
            0-1. 50 epochs are chosen if, for any arm, the mean development AUC of the 50-epoch runs
            exceeds that of the 30-epoch runs by 0.005 or more; otherwise 30. If 50 are chosen and any
            arm gained 0.01 or more, convergence is not claimed ("longer-budget comparison"). The
            length is converted to a number of training samples, S* = epochs x mean training-set size
            of the development folds; every final run trains for at least S* samples.
analyze     The registered analysis of the blinded final runs (seeds 4-6): primary family (6, Holm;
            the CMMD ResNet-50 ROI - random crop replication one-sided), secondary family (9),
            cross-budget contrasts against the ten-epoch runs (6), failed runs and the sensitivity
            analysis without them, per-seed effects, the CMMD within-fold AUC, training losses and the
            registered interpretation of each primary comparison. It refuses to run until all final
            runs are finished, so that the analysis is run once.
"""
from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import load_runs, pool_oof, unit_preds
from .controlarm import labels
from .metrics import auc, holm
from .reanalysis import clusters, draw

ARMS = ("std", "roi", "rand", "crb")
NAMES = {"std": "Standard", "roi": "ROI crop", "rand": "Random crop", "crb": "Central random box"}
SETTINGS = (("miniddsm", "effb0"), ("miniddsm", "resnet50"), ("cmmd", "resnet50"))
PRIMARY = (("roi", "std"), ("roi", "rand"))
SECONDARY = (("roi", "crb"), ("crb", "rand"), ("rand", "std"))
REPLICATION = ("cmmd", "resnet50", "roi", "rand")
LRS, LENGTHS, TUNE_FOLDS, LEN_FOLDS, SEEDS = (2e-4, 1e-4), (30, 50), (0, 1, 2, 3), (0, 1), (4, 5, 6)
TIE, GAIN, LONGER = 0.002, 0.005, 0.01
MARGIN, MARGIN_SENS = 0.01, 0.02
FAIL_FRAC = 0.9


def trivial_loss(pos_weight: float) -> float:
    """Class-weighted BCE of the best constant prediction when pos_weight = negatives/positives."""
    return math.log(2) * 2 * pos_weight / (1 + pos_weight)


def breast_auc(run) -> float:
    u = unit_preds(run, "breast")
    return auc(u.label.to_numpy().astype(int), u.prob.to_numpy())


def epoch_curves(runs, arm_of) -> pd.DataFrame:
    rows = []
    for r in runs:
        lg = pd.read_csv(r["dir"] / "log.csv")
        for f in sorted(r["dir"].glob("preds_ep*.csv")):
            ep = int(f.stem[-2:])
            u = unit_preds(r, "breast", f.name)
            rows.append({"backbone": r["backbone"], "arm": arm_of(r), "lr": float(r["lr"]), "epochs": r["epochs"],
                         "fold": r["fold"], "epoch": ep, "auc": auc(u.label.to_numpy().astype(int), u.prob.to_numpy()),
                         "loss": float(lg.loss[lg.epoch == ep].iloc[0])})
    return pd.DataFrame(rows)


def dev_runs(a):
    return [r for r in load_runs(a.runs, "cv", "convergence") if r["dataset"] == "miniddsm"
            and r["head"] == "gap" and r["unit"] == "image"]


# ----------------------------------------------------------------------------- Step 1a

def cmd_select_lr(a):
    out = a.results / "convergence"
    out.mkdir(parents=True, exist_ok=True)
    lengths, folds = tuple(a.lengths), tuple(range(a.folds))
    runs = [r for r in dev_runs(a) if r["arm"] == "none" and r["epochs"] == lengths[0]]
    for bb in a.backbones:
        rs = [r for r in runs if r["backbone"] == bb and int(r["fold"]) in folds
              and any(np.isclose(float(r["lr"]), x) for x in LRS)]
        df = pd.DataFrame([{"lr": float(r["lr"]), "fold": int(r["fold"]), "auc": breast_auc(r),
                            "final_loss": float(pd.read_csv(r["dir"] / "log.csv").loss.iloc[-1]),
                            "n_train": int(r["n_train"])} for r in rs])
        if df.empty or df.groupby("lr").fold.nunique().reindex(list(LRS)).fillna(0).min() < len(folds):
            raise SystemExit(f"{bb}: Step 1a incomplete (need {len(LRS)} rates x {len(folds)} folds)")
        s = df.groupby("lr").agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"), final_loss=("final_loss", "mean"))
        best = s.auc_mean.max()
        lr = float(min(s.index[s.auc_mean >= best - TIE]))  # ties within 0.002 go to the lower rate
        df.to_csv(out / f"lr_runs_{bb}.csv", index=False)
        epoch_curves(rs, lambda r: "std").to_csv(out / f"lr_curves_{bb}.csv", index=False)
        json.dump({"backbone": bb, "lr": lr, "rule": "higher mean breast-level AUC on development folds 0-3 after "
                   f"{lengths[0]} epochs, standard augmentation; ties within {TIE} go to the lower rate",
                   "mean_auc": {f"{k:g}": round(v, 5) for k, v in s.auc_mean.items()},
                   "dev_train_images_mean": float(df.n_train.mean())},
                  open(out / f"chosen_lr_{bb}.json", "w"), indent=1)
        print(f"== {bb}\n{s.to_string(float_format=lambda v: f'{v:.4f}')}\n[chosen] {bb} lr={lr:g}")


# ----------------------------------------------------------------------------- Step 1b

def cmd_select_len(a):
    out = a.results / "convergence"
    lab = labels(json.load(open(a.results / "chosen_miniddsm.json")))
    inv = {v: k for k, v in lab.items()}
    runs = dev_runs(a)
    L = tuple(a.lengths)
    for bb in a.backbones:
        ch = json.load(open(out / f"chosen_lr_{bb}.json"))
        rs = [r for r in runs if r["backbone"] == bb and np.isclose(float(r["lr"]), ch["lr"])
              and r["arm_label"] in inv and int(r["fold"]) in LEN_FOLDS and r["epochs"] in L]
        df = pd.DataFrame([{"arm": inv[r["arm_label"]], "epochs": r["epochs"], "fold": int(r["fold"]),
                            "auc": breast_auc(r),
                            "final_loss": float(pd.read_csv(r["dir"] / "log.csv").loss.iloc[-1])} for r in rs])
        need = len(ARMS) * len(L) * len(LEN_FOLDS)
        if len(df) != need:
            raise SystemExit(f"{bb}: Step 1b incomplete ({len(df)} of {need} runs)")
        m = df.groupby(["arm", "epochs"]).auc.mean().unstack()
        gain = (m[L[1]] - m[L[0]]).reindex(list(ARMS))
        epochs = L[1] if (gain >= GAIN).any() else L[0]
        longer = bool(epochs == L[1] and (gain >= LONGER).any())
        samples = int(math.ceil(epochs * ch["dev_train_images_mean"]))
        df.to_csv(out / f"len_runs_{bb}.csv", index=False)
        curves = epoch_curves(rs, lambda r: inv[r["arm_label"]])
        curves.to_csv(out / f"len_curves_{bb}.csv", index=False)
        _plot_curves(curves, out / f"len_curves_{bb}.png", bb, L)
        json.dump({"backbone": bb, "lr": ch["lr"], "epochs": epochs, "samples": samples,
                   "longer_budget_comparison": longer,
                   "rule": f"50 epochs if any arm's mean development AUC (folds 0-1) gains >= {GAIN} from 30 to 50 "
                           f"epochs; 'longer-budget comparison' if 50 are chosen and any arm gains >= {LONGER}",
                   "gain_30_to_50": {k: round(float(v), 5) for k, v in gain.items()},
                   "mean_auc": {f"{k}@{e}": round(float(m.loc[k, e]), 5) for k in ARMS for e in L}},
                  open(out / f"chosen_conv_{bb}.json", "w"), indent=1)
        print(f"== {bb} (lr {ch['lr']:g})\n{m.to_string(float_format=lambda v: f'{v:.4f}')}\n"
              f"gain 30->50: {gain.round(4).to_dict()}\n[chosen] {bb}: {epochs} epochs, S* = {samples} samples"
              + (" (longer-budget comparison)" if longer else ""))


def _plot_curves(curves, path, title, L=LENGTHS):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for (arm, ep), g in curves.groupby(["arm", "epochs"]):
        mc = g.groupby("epoch")[["auc", "loss"]].mean()
        ls = "-" if ep == L[1] else "--"
        axes[0].plot(mc.index, mc.auc, ls, label=f"{NAMES[arm]}, {ep} ep")
        axes[1].plot(mc.index, mc.loss, ls, label=f"{NAMES[arm]}, {ep} ep")
    axes[0].set_ylabel("development breast-level AUC")
    axes[1].set_ylabel("weighted training loss")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- Step 3

def _auc_masked(y, s):
    m = ~np.isnan(s)
    return auc(y[m], s[m])


def _delta(y, X, R, idx, seeds):
    return np.mean([_auc_masked(y[idx], X[s][idx]) - _auc_masked(y[idx], R[s][idx]) for s in seeds])


def _draw_ok(starts, counts, y, rng):
    while True:
        idx = draw(starts, counts, rng)
        if 0 < y[idx].sum() < len(idx):
            return idx


def boot(job):
    """job: key, y, X, R, pids, n, mode ('hier' | 'patients'), seed, [X0, R0] for cross-budget."""
    key, y, X, R, pids, n, mode, seed, old = job
    rng = np.random.default_rng(seed)
    starts, counts = clusters(pids)
    k, out = len(X), np.empty(n)
    for b in range(n):
        idx = _draw_ok(starts, counts, y, rng)
        ss = rng.integers(0, k, k) if mode == "hier" else range(k)
        d = _delta(y, X, R, idx, ss)
        if old is not None:  # cross-budget: same patients, seeds of each recipe resampled independently
            X0, R0 = old
            s0 = rng.integers(0, len(X0), len(X0))
            d -= _delta(y, X0, R0, idx, s0)
        out[b] = d
    return key, out


def p_two(b):
    nb = len(b)
    return min(1.0, 2 * min(((b <= 0).sum() + 1) / (nb + 1), ((b >= 0).sum() + 1) / (nb + 1)))


def p_one_greater(b):
    return ((b <= 0).sum() + 1) / (len(b) + 1)


def p_tost(b, margin):
    nb = len(b)
    return max(((b <= -margin).sum() + 1) / (nb + 1), ((b >= margin).sum() + 1) / (nb + 1))


def interpret_std(row) -> str:
    if row.p_holm < 0.05:
        return "ROI cropping helps in this setting" if row.delta > 0 else "ROI cropping hurts in this setting"
    if row.ci95_hi < MARGIN:
        return "no meaningful benefit (a benefit of 0.01 or more is excluded)"
    if row.ci95_hi < MARGIN_SENS:
        return "a benefit of 0.02 or more is excluded; smaller benefits remain possible"
    return "inconclusive"


def interpret_rep(row) -> str:
    if row.delta > 0 and row.p_holm < 0.05:
        return "advantage over random crops replicated with fresh seeds"
    if row.ci95_hi < MARGIN:
        return "not replicated: a gain of 0.01 or more is excluded under the new recipe; the mechanism remains open"
    return "replication inconclusive"


def cmd_analyze(a):
    SETTINGS_ = tuple(tuple(x.split(":")) for x in a.settings)
    out = a.results / "convergence"
    out.mkdir(parents=True, exist_ok=True)
    lab = labels(json.load(open(a.results / "chosen_miniddsm.json")))
    inv = {v: k for k, v in lab.items()}
    chosen = {bb: json.load(open(out / f"chosen_conv_{bb}.json")) for bb in {b for _, b in SETTINGS_}}

    def keep(r, bb):
        return (r["backbone"] == bb and r["head"] == "gap" and r["unit"] == "image" and r["arm_label"] in inv)

    new_raw = load_runs(a.runs, "final", "convergence")
    new = pool_oof(new_raw)
    old = pool_oof(load_runs(a.runs, "final", "all"))
    lines = []

    def say(t=""):
        print(t, flush=True)
        lines.append(t)

    # completeness (blinding: the analysis is run once, on the full set of registered runs)
    sp = {ds: pd.read_csv(a.data_root / ds / "split.csv") for ds, _ in SETTINGS_}
    missing = []
    for ds, bb in SETTINGS_:
        n_units = 1 if (sp[ds].split == "test").any() else sp[ds].fold[sp[ds].fold >= 0].nunique()
        have = [r for r in new_raw if r["dataset"] == ds and keep(r, bb)]
        need = len(ARMS) * len(SEEDS) * n_units
        if len(have) != need:
            missing.append(f"{ds}/{bb}: {len(have)} of {need} runs")
    if missing and not a.allow_incomplete:
        raise SystemExit("not all registered runs are finished (" + "; ".join(missing) + "); the analysis runs once, "
                         "on the complete set")

    # failed runs (registered rule: final training loss > 0.9 x loss of a constant prediction)
    frows = []
    for r in new_raw:
        bb = r["backbone"]
        if not any(r["dataset"] == ds and bb == b for ds, b in SETTINGS_) or not keep(r, bb):
            continue
        lg = pd.read_csv(r["dir"] / "log.csv")
        thr = FAIL_FRAC * trivial_loss(float(r["pos_weight_value"]))
        frows.append({"dataset": r["dataset"], "backbone": bb, "arm": inv[r["arm_label"]], "seed": r["seed"],
                      "fold": r.get("oof_fold"), "epochs": r["epochs"], "final_loss": float(lg.loss.iloc[-1]),
                      "threshold": thr, "failed": bool(lg.loss.iloc[-1] > thr)})
    fr = pd.DataFrame(frows)
    fr.to_csv(out / "runs_loss_failures.csv", index=False)
    say("Failed runs per setting and arm (final training loss above 90% of a constant prediction):")
    say(fr.groupby(["dataset", "backbone", "arm"]).failed.agg(["sum", "size"]).to_string())
    curves = []
    for r in new_raw:
        if keep(r, r["backbone"]):
            lg = pd.read_csv(r["dir"] / "log.csv")
            curves += [{"dataset": r["dataset"], "backbone": r["backbone"], "arm": inv[r["arm_label"]],
                        "seed": r["seed"], "fold": r.get("oof_fold"), "epoch": int(e), "loss": float(l)}
                       for e, l in zip(lg.epoch, lg.loss)]
    pd.DataFrame(curves).to_csv(out / "loss_curves.csv", index=False)

    def preds(runs, ds, bb, arm):
        rs = {r["seed"]: r for r in runs if r["dataset"] == ds and keep(r, bb) and inv[r["arm_label"]] == arm}
        return {s: unit_preds(rs[s], "breast") for s in sorted(rs)}

    def arrays(P, base, mask_units=None):
        """Score arrays aligned to base, one per seed; in the sensitivity analysis the units of failed runs
        are set to NaN, and a seed with no units left is dropped (identically for both arms)."""
        out_ = []
        for s, u in P.items():
            v = u.set_index("key").loc[base.key, "prob"].to_numpy().astype(float)
            if mask_units is not None and s in mask_units:
                if mask_units[s].all():
                    continue
                v = np.where(mask_units[s], np.nan, v)
            out_.append(v)
        return out_

    failed_units = {}  # (ds, bb) -> {seed: set(folds) or {None}}
    for _, f in fr[fr.failed].iterrows():
        failed_units.setdefault((f.dataset, f.backbone), {}).setdefault(int(f.seed), set()).add(f.fold)

    jobs, meta = [], []
    fold_of = {ds: sp[ds].set_index("patient_id")["fold"] for ds, _ in SETTINGS_}
    for ds, bb in SETTINGS_:
        P = {arm: preds(new, ds, bb, arm) for arm in ARMS}
        P0 = {arm: preds(old, ds, bb, arm) for arm in ("std", "roi", "rand")}
        any_arm = next(iter(P["std"].values()))
        base = any_arm[["key", "patient_id", "label"]].sort_values(["patient_id", "key"]).reset_index(drop=True)
        y, pids = base.label.to_numpy().astype(int), base.patient_id.to_numpy()
        folds = base.patient_id.map(fold_of[ds]).to_numpy()
        # sensitivity mask: drop failed seed (Mini-DDSM) or failed seed-and-fold units (CMMD), all arms alike
        fu = failed_units.get((ds, bb), {})
        has_test = bool((sp[ds].split == "test").any())
        mask = {s: (np.ones(len(base), bool) if has_test else np.isin(folds, [int(f) for f in fl if f is not None]))
                for s, fl in fu.items()}
        for fam, comps in (("primary", PRIMARY), ("secondary", SECONDARY)):
            for x, r in comps:
                X, R = arrays(P[x], base), arrays(P[r], base)
                key = dict(family=fam, dataset=ds, backbone=bb, comparison=f"{NAMES[x]} - {NAMES[r]}", x=x, ref=r)
                jobs.append((dict(key, kind="hier"), y, X, R, pids, a.n_boot, "hier", 0, None))
                jobs.append((dict(key, kind="patients"), y, X, R, pids, a.n_boot, "patients", 1, None))
                meta.append((key, y, X, R, folds))
                if fam == "primary" and mask:
                    Xs, Rs = arrays(P[x], base, mask), arrays(P[r], base, mask)
                    jobs.append((dict(key, kind="hier_sensitivity"), y, Xs, Rs, pids, a.n_boot, "hier", 2, None))
        for x, r in PRIMARY:
            X, R = arrays(P[x], base), arrays(P[r], base)
            X0, R0 = arrays(P0[x], base), arrays(P0[r], base)
            key = dict(family="cross-budget", dataset=ds, backbone=bb,
                       comparison=f"({NAMES[x]} - {NAMES[r]}) new recipe - ten epochs", x=x, ref=r)
            jobs.append((dict(key, kind="hier"), y, X, R, pids, a.n_boot, "hier", 3, (X0, R0)))
            meta.append((key, y, X, R, folds, X0, R0))

    with Pool(a.workers) as pool:
        res = pool.map(boot, jobs)
    dist = {}
    for key, b in res:
        k = tuple((kk, vv) for kk, vv in key.items() if kk != "kind")
        dist.setdefault(k, {})[key["kind"]] = b

    rows = []
    for m in meta:
        key, y, X, R, folds = m[:5]
        k = tuple(key.items())
        d = dist[k]
        per_seed = [_auc_masked(y, xx) - _auc_masked(y, rr) for xx, rr in zip(X, R)]
        delta = float(np.mean(per_seed))
        if key["family"] == "cross-budget":
            X0, R0 = m[5], m[6]
            old_seed = [_auc_masked(y, xx) - _auc_masked(y, rr) for xx, rr in zip(X0, R0)]
            delta -= float(np.mean(old_seed))
        h = d["hier"]
        rep = (key["dataset"], key["backbone"], key["x"], key["ref"]) == REPLICATION and key["family"] == "primary"
        row = dict(key, delta=delta, per_seed=" ".join(f"{v:+.4f}" for v in per_seed),
                   per_seed_ten_epochs=(" ".join(f"{v:+.4f}" for v in old_seed)
                                        if key["family"] == "cross-budget" else ""),
                   ci95_lo=np.percentile(h, 2.5), ci95_hi=np.percentile(h, 97.5),
                   ci90_lo=np.percentile(h, 5), ci90_hi=np.percentile(h, 95),
                   p=(p_one_greater(h) if rep else p_two(h)), test=("one-sided (ROI > random)" if rep else "two-sided"),
                   p_equiv_001=p_tost(h, MARGIN), p_equiv_002=p_tost(h, MARGIN_SENS))
        if "patients" in d:
            row.update(pat_ci95_lo=np.percentile(d["patients"], 2.5), pat_ci95_hi=np.percentile(d["patients"], 97.5))
        if "hier_sensitivity" in d:
            row.update(sens_ci95_lo=np.percentile(d["hier_sensitivity"], 2.5),
                       sens_ci95_hi=np.percentile(d["hier_sensitivity"], 97.5),
                       sens_p=(p_one_greater if rep else p_two)(d["hier_sensitivity"]))
        if key["dataset"] == "cmmd" and key["family"] != "cross-budget":
            from .reanalysis import strat_auc
            row["within_fold_delta"] = float(np.mean([strat_auc(y, xx, folds) - strat_auc(y, rr, folds)
                                                      for xx, rr in zip(X, R)]))
        rows.append(row)
    t = pd.DataFrame(rows)
    for fam in ("primary", "secondary", "cross-budget"):
        m_ = t.family == fam
        t.loc[m_, "p_holm"] = holm(t.loc[m_, "p"].to_numpy())
        t.loc[m_, "p_equiv_001_holm"] = holm(t.loc[m_, "p_equiv_001"].to_numpy())
    prim = t[t.family == "primary"].copy()
    prim["interpretation"] = [interpret_rep(r) if (r.dataset, r.backbone, r.x, r.ref) == REPLICATION
                              else (interpret_std(r) if r.ref == "std" else "") for r in prim.itertuples()]
    t = t.merge(prim[["dataset", "backbone", "comparison", "interpretation"]], how="left",
                on=["dataset", "backbone", "comparison"])
    t.to_csv(out / "convergence_breast.csv", index=False)
    cols = ["family", "dataset", "backbone", "comparison", "delta", "ci95_lo", "ci95_hi", "per_seed", "p", "p_holm",
            "p_equiv_001_holm", "interpretation"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80):
        say(f"\nAddendum C (breast-level AUC; 95% CI: seeds then patients, {a.n_boot} replicates):")
        say(t[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    say("\nRecipes: " + "; ".join(f"{bb}: lr {c['lr']:g}, {c['epochs']} epochs, S* = {c['samples']} samples"
                                   + (" (longer-budget comparison)" if c["longer_budget_comparison"] else "")
                                   for bb, c in chosen.items()))
    say("CMMD: step-matched longer-budget comparison (convergence on CMMD is not established by this design).")
    (out / "convergence_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"[done] {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select-lr", "select-len", "analyze"):
        s = sub.add_parser(name)
        s.add_argument("--runs", type=Path, default=Path("runs"))
        s.add_argument("--results", type=Path, default=Path("results"))
        s.add_argument("--data-root", type=Path, default=Path("data"))
        s.add_argument("--backbones", nargs="+", default=["effb0", "resnet50"])
        s.add_argument("--lengths", type=int, nargs=2, default=list(LENGTHS), help="registered: 30 50 (testing only)")
        s.add_argument("--folds", type=int, default=len(TUNE_FOLDS), help="development folds for select-lr")
        if name == "analyze":
            s.add_argument("--n-boot", type=int, default=20000)
            s.add_argument("--workers", type=int, default=4)
            s.add_argument("--allow-incomplete", action="store_true", help="testing only")
            s.add_argument("--settings", nargs="+", default=[f"{d}:{b}" for d, b in SETTINGS],
                           help="dataset:backbone (registered: all three; fewer for testing only)")
    a = ap.parse_args(argv)
    {"select-lr": cmd_select_lr, "select-len": cmd_select_len, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
