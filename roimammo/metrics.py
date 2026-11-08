"""Metrics and statistics: AUC, PR-AUC, Brier, ECE, aggregation, DeLong, cluster bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score


def auc(y, s) -> float:
    """ROC-AUC via the Mann-Whitney statistic (ties count one half)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pr_auc(y, s) -> float:
    y = np.asarray(y).astype(int)
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def aggregate(preds: pd.DataFrame, level: str) -> pd.DataFrame:
    """Average probabilities to breast or patient level; the label is positive if any unit is."""
    if level == "patient":
        g = preds.groupby("patient_id").agg(label=("label", "max"), prob=("prob", "mean")).reset_index()
        g.insert(0, "key", g["patient_id"])
        return g
    g = preds.groupby("breast_id").agg(label=("label", "max"), prob=("prob", "mean"),
                                       patient_id=("patient_id", "first")).reset_index()
    return g.rename(columns={"breast_id": "key"})


def all_metrics(y, p) -> dict:
    return {"auc": auc(y, p), "pr_auc": pr_auc(y, p), "brier": brier(y, p), "ece": ece(y, p),
            "n": int(len(y)), "n_pos": int(np.sum(y))}


# ----------------------------------------------------------------------------- DeLong (paired)

def _midrank(x):
    return stats.rankdata(x)


def delong_paired(y, s1, s2):
    """Paired DeLong test for two correlated ROC-AUCs (DeLong et al. 1988; Sun & Xu 2014).

    Assumes independent units; with two breasts per patient use the cluster bootstrap as the
    primary inference and DeLong as a secondary check.
    Returns (auc1, auc2, diff, se, z, p).
    """
    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    S = np.vstack([s1, s2]).astype(float)
    v10, v01, aucs = [], [], []
    for s in S:
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(s)
        a = (tz[pos].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(a)
        v10.append((tz[pos] - tx) / n)
        v01.append(1 - (tz[neg] - ty) / m)
    v10, v01 = np.array(v10), np.array(v01)
    cov = np.cov(v10) / m + np.cov(v01) / n
    d = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 1e-12)))
    z = d / se
    return aucs[0], aucs[1], d, se, z, float(2 * stats.norm.sf(abs(z)))


# ----------------------------------------------------------------------------- cluster bootstrap

def paired_cluster_bootstrap(units: pd.DataFrame, probs_a: list[np.ndarray], probs_b: list[np.ndarray],
                             n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of mean_seed AUC(A) - mean_seed AUC(B), resampling patients.

    ``units`` has columns patient_id, label (one row per evaluated unit, same order as the
    probability arrays). probs_a / probs_b hold one array per seed. Seeds are averaged inside
    each replicate, so the interval reflects patient sampling for the seed-averaged effect.
    """
    rng = np.random.default_rng(seed)
    pats, inv = np.unique(units["patient_id"].to_numpy(), return_inverse=True)
    members = [np.nonzero(inv == i)[0] for i in range(len(pats))]
    y = units["label"].to_numpy().astype(int)
    out = np.empty(n_boot)
    for bi in range(n_boot):
        for _ in range(100):  # redraw the rare replicate that contains only one class
            pick = rng.integers(0, len(pats), len(pats))
            idx = np.concatenate([members[i] for i in pick])
            yy = y[idx]
            if 0 < yy.sum() < len(yy):
                break
        da = np.mean([auc(yy, p[idx]) for p in probs_a])
        db = np.mean([auc(yy, p[idx]) for p in probs_b])
        out[bi] = da - db
    return out


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (len(p) - r) * p[i])
        adj[i] = min(1.0, running)
    return adj
