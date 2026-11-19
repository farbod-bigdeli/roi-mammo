"""Statistics added after the second internal review (no retraining; uses the saved predictions).

For every comparison of the main analysis (study main) and of the addendum (study all):
    * patient-cluster bootstrap with 20,000 replicates (the main analysis used 2,000, which puts
      a floor of 1/2001 under each p-value), with the same paired, seed-averaged statistic;
    * where several seeds exist, a two-level (hierarchical) bootstrap that resamples seeds and
      then patients, so that the interval also reflects retraining variance;
    * the per-seed differences;
    * a bootstrap TOST p-value for equivalence within +-margin (max of the two one-sided
      bootstrap tail fractions), so that equivalence can be multiplicity-adjusted;
    * the role of each comparison in the plan (primary, secondary, or random crop vs standard,
      which the plan included in the family without naming it).
Holm adjustment is reported for: the pre-specified family (all 28 comparisons of the main
analysis), the primary comparisons alone (12), the addendum family as run (48), and the addendum
family without the 28 comparisons that repeat the main analysis.

For the primary EfficientNet-B0 and ResNet-50 contrasts on Mini-DDSM it also reports
    * a site-stratified AUC difference (AUC over positive-negative pairs from the same digitizer,
      weighted by the number of such pairs) with a bootstrap that resamples patients within sites;
    * a patient-level analysis that scores a patient by the maximum of the breast scores.

    python -m roimammo.reanalysis --runs runs --out results/reanalysis
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import comparisons_for, load_runs, pool_oof, unit_preds
from .metrics import auc, holm

MARGIN = 0.01


def role(model: str, arm: str, ref: str) -> str:
    if arm.startswith("roi") and (ref == "none" or ref.startswith("randcrop")) and model.endswith("gap/image"):
        return "primary"
    if arm.startswith("randcrop") and ref == "none":
        return "random crop vs standard"
    return "secondary"


def clusters(pids: np.ndarray):
    """Units must be sorted by patient. Returns start offsets and lengths per patient."""
    _, starts, counts = np.unique(pids, return_index=True, return_counts=True)
    order = np.argsort(starts)
    return starts[order], counts[order]


def draw(starts, counts, rng):
    pick = rng.integers(0, len(starts), len(starts))
    s, c = starts[pick], counts[pick]
    rep = np.repeat(s - np.concatenate([[0], np.cumsum(c)[:-1]]), c)
    return rep + np.arange(c.sum())


def boot_patients(y, X, R, pids, n, seed):
    """Seed-averaged paired AUC difference; X, R: lists (one per seed) of score arrays."""
    rng = np.random.default_rng(seed)
    starts, counts = clusters(pids)
    out = np.empty(n)
    for b in range(n):
        while True:
            idx = draw(starts, counts, rng)
            yy = y[idx]
            if 0 < yy.sum() < len(yy):
                break
        out[b] = np.mean([auc(yy, x[idx]) - auc(yy, r[idx]) for x, r in zip(X, R)])
    return out


def boot_hier(y, X, R, pids, n, seed):
    rng = np.random.default_rng(seed + 1)
    starts, counts = clusters(pids)
    k = len(X)
    out = np.empty(n)
    for b in range(n):
        ss = rng.integers(0, k, k)
        while True:
            idx = draw(starts, counts, rng)
            yy = y[idx]
            if 0 < yy.sum() < len(yy):
                break
        out[b] = np.mean([auc(yy, X[s][idx]) - auc(yy, R[s][idx]) for s in ss])
    return out


def strat_auc(y, s, g):
    num = den = 0.0
    for v in np.unique(g):
        m = g == v
        n1 = y[m].sum()
        n0 = m.sum() - n1
        if n1 and n0:
            num += auc(y[m], s[m]) * n1 * n0
            den += n1 * n0
    return num / den


def boot_strat(y, X, R, pids, g, n, seed):
    rng = np.random.default_rng(seed + 2)
    groups = []
    for v in np.unique(g):
        idx_v = np.nonzero(g == v)[0]
        st, ct = clusters(pids[idx_v])
        groups.append((idx_v, st, ct))
    out = np.empty(n)
    for b in range(n):
        idx = np.concatenate([iv[draw(st, ct, rng)] for iv, st, ct in groups])
        yy, gg = y[idx], g[idx]
        out[b] = np.mean([strat_auc(yy, x[idx], gg) - strat_auc(yy, r[idx], gg) for x, r in zip(X, R)])
    return out


def pvals(boot):
    nb = len(boot)
    p = min(1.0, 2 * min(((boot <= 0).sum() + 1) / (nb + 1), ((boot >= 0).sum() + 1) / (nb + 1)))
    p_eq = max(((boot <= -MARGIN).sum() + 1) / (nb + 1), ((boot >= MARGIN).sum() + 1) / (nb + 1))
    return p, p_eq


def one(job):
    key, y, X, R, pids, site, n_boot, n_hier = job
    d_seed = [auc(y, x) - auc(y, r) for x, r in zip(X, R)]
    b = boot_patients(y, X, R, pids, n_boot, 0)
    p, p_eq = pvals(b)
    row = dict(key, seeds=len(X), delta=float(np.mean(d_seed)),
               per_seed=" ".join(f"{v:+.4f}" for v in d_seed),
               ci95_lo=np.percentile(b, 2.5), ci95_hi=np.percentile(b, 97.5),
               ci90_lo=np.percentile(b, 5), ci90_hi=np.percentile(b, 95), p_boot=p, p_tost=p_eq)
    if len(X) > 1:
        h = boot_hier(y, X, R, pids, n_hier, 0)
        row.update(hier_ci95_lo=np.percentile(h, 2.5), hier_ci95_hi=np.percentile(h, 97.5),
                   hier_ci90_lo=np.percentile(h, 5), hier_ci90_hi=np.percentile(h, 95))
    if site is not None:
        s = boot_strat(y, X, R, pids, site, n_hier, 0)
        row.update(site_delta=float(np.mean([strat_auc(y, x, site) - strat_auc(y, r, site) for x, r in zip(X, R)])),
                   site_ci95_lo=np.percentile(s, 2.5), site_ci95_hi=np.percentile(s, 97.5))
    return row


def jobs_for(runs, level, n_boot, n_hier, with_site=False, patient_max=False):
    jobs = []
    for (ds, model), grp in pd.DataFrame(runs).groupby(["dataset", "model"]):
        by_arm = {k: g.sort_values("seed").to_dict("records") for k, g in grp.groupby("arm_label")}
        for arm_x, arm_ref in comparisons_for(sorted(by_arm)):
            rx = {r["seed"]: r for r in by_arm[arm_x]}
            rr = {r["seed"]: r for r in by_arm[arm_ref]}
            seeds = sorted(set(rx) & set(rr))
            if patient_max:
                px = {s: unit_preds(rx[s], "breast") for s in seeds}
                pr = {s: unit_preds(rr[s], "breast") for s in seeds}
                agg = lambda u: u.groupby("patient_id").agg(label=("label", "max"), prob=("prob", "max")).reset_index().assign(key=lambda d: d.patient_id)
                px = {s: agg(u) for s, u in px.items()}
                pr = {s: agg(u) for s, u in pr.items()}
            else:
                px = {s: unit_preds(rx[s], level) for s in seeds}
                pr = {s: unit_preds(rr[s], level) for s in seeds}
            base = px[seeds[0]][["key", "patient_id", "label"]].sort_values(["patient_id", "key"]).reset_index(drop=True)
            y = base.label.to_numpy().astype(int)
            X = [px[s].set_index("key").loc[base.key, "prob"].to_numpy() for s in seeds]
            R = [pr[s].set_index("key").loc[base.key, "prob"].to_numpy() for s in seeds]
            site = base.patient_id.str[0].to_numpy() if (with_site and ds == "miniddsm") else None
            key = dict(dataset=ds, model=model, arm=arm_x, reference=arm_ref, role=role(model, arm_x, arm_ref),
                       n_units=len(base), n_pos=int(y.sum()))
            jobs.append((key, y, X, R, base.patient_id.to_numpy(), site, n_boot, n_hier))
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results/reanalysis"))
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--n-hier", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    main_runs = pool_oof(load_runs(a.runs, "final", "main"))
    all_runs = pool_oof(load_runs(a.runs, "final", "all"))
    with Pool(a.workers) as pool:
        m = pd.DataFrame(pool.map(one, jobs_for(main_runs, "breast", a.n_boot, a.n_hier, with_site=True)))
        m["p_holm_prespecified"] = holm(m.p_boot)
        prim = m.role == "primary"
        m.loc[prim, "p_holm_primary_only"] = holm(m.p_boot[prim])
        m["p_tost_holm"] = holm(m.p_tost)
        m.to_csv(a.out / "main_breast.csv", index=False)
        print(f"[info] main analysis done ({len(m)} comparisons)", flush=True)

        ad = pd.DataFrame(pool.map(one, jobs_for(all_runs, "breast", a.n_boot, a.n_hier)))
        ad["p_holm_addendum_as_run"] = holm(ad.p_boot)
        k = ["dataset", "model", "arm", "reference", "seeds"]
        rep = ad.merge(m[k + ["delta"]], on=k, how="left", suffixes=("", "_main"))
        ad["repeats_main"] = (rep.delta_main - rep.delta).abs().lt(1e-12).fillna(False).to_numpy()
        new = ~ad.repeats_main
        ad.loc[new, "p_holm_addendum_new_only"] = holm(ad.p_boot[new])
        ad.loc[new, "p_tost_holm_new_only"] = holm(ad.p_tost[new])
        ad.to_csv(a.out / "addendum_breast.csv", index=False)
        print(f"[info] addendum done ({len(ad)} comparisons, {int(new.sum())} new)", flush=True)

        pm = pd.DataFrame(pool.map(one, [j for j in jobs_for(main_runs, "patient", a.n_hier, a.n_hier, patient_max=True)
                                         if j[0]["role"] == "primary"]))
        pm.to_csv(a.out / "patient_max_primary.csv", index=False)
        print("[info] patient max-score analysis done", flush=True)
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
