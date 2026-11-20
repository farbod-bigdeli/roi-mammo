"""Train one model and save its predictions.

Modes:
    cv     train on development folds != --fold, predict the held-out development fold every epoch
    final  train on all development patients, predict the locked test set once, after the last epoch.
           If the dataset has no test set (splits made with --test-frac 0), train on folds != --fold
           and predict fold --fold once ("final-oof"); the analysis pools all folds.
The reported model is always the LAST epoch (no checkpoint picking on evaluation data).

Example:
    python -m roimammo.train --data-dir data/miniddsm --mode cv --fold 0 --backbone effb0 \
        --arm roi --bank data/miniddsm/roi_bank_blend.json --p-crop 0.1 --jitter 0.1 --seed 1

Outputs in <out-root>/<run-name>/: config.json, env.json, log.csv (per epoch: loss, time,
throughput, data-wait, peak memory, eval AUC), preds.csv (last epoch), preds_epXX.csv
(cv mode), DONE.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ARMS, MammoDataset
from .metrics import aggregate, auc as fast_auc
from .models import Classifier


def set_seed(s: int, deterministic: bool):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def collate(batch):
    return {"x": torch.stack([b["x"] for b in batch]), "y": torch.stack([b["y"] for b in batch]),
            "id": [b["id"] for b in batch], "patient_id": [b["patient_id"] for b in batch],
            "breast_id": [b["breast_id"] for b in batch]}


def cutmix(x, y, rng: np.random.Generator, p=0.5, alpha=1.0):
    if rng.random() >= p or x.shape[0] < 2:
        return x, y
    lam = rng.beta(alpha, alpha)
    H, W = x.shape[-2:]
    ch, cw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = rng.integers(H), rng.integers(W)
    y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
    x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
    perm = torch.from_numpy(rng.permutation(x.shape[0]))
    x = x.clone()
    x[..., y0:y1, x0:x1] = x[perm][..., y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    return x, lam * y + (1 - lam) * y[perm]


def has_test_set(data_dir: Path) -> bool:
    return (pd.read_csv(data_dir / "split.csv").split == "test").any()


DEFAULT_LR = 2e-4


def run_name(a) -> str:
    crop = ""
    if a.arm in ("roi", "randcrop"):
        crop = f"_p{a.p_crop:g}_j{a.jitter:g}_k{a.roi_k}_{Path(a.bank).stem.replace('roi_bank_', '')}"
        if getattr(a, "crop_fit", "letterbox") == "fill":
            crop += "_fill"
    if getattr(a, "lr", DEFAULT_LR) != DEFAULT_LR:  # non-default recipes get their own name (addendum)
        crop += f"_lr{a.lr:g}"
    if getattr(a, "study", "main") == "convergence":  # addendum C runs never share a name with earlier runs
        crop += "_conv"
    if a.mode == "cv":
        part = f"cv{a.fold}"
    else:
        part = "final" if has_test_set(a.data_dir) else f"final-oof{a.fold}"
    return (f"{a.data_dir.name}_{a.backbone}_{a.head}_{a.unit}_{a.arm}{crop}_"
            f"{a.img_h}x{a.img_w}_e{a.epochs}_{part}_s{a.seed}{a.tag}")


def epochs_for_samples(a) -> int:
    """Addendum C: epochs = max(--epochs, ceil(N / samples presented per epoch)) for this run's
    training set (drop_last batches, so samples per epoch = floor(n / batch) * batch)."""
    if a.unit != "image":
        sys.exit("--epochs-from-steps supports --unit image only")
    index = pd.read_csv(a.data_dir / "index.csv")
    split = pd.read_csv(a.data_dir / "split.csv")
    df = index.merge(split, on="patient_id")
    dev = df[df.split == "dev"]
    oof = a.mode == "final" and not (df.split == "test").any()
    tr = dev[dev.fold != a.fold] if (a.mode == "cv" or oof) else dev
    batches = len(tr) // a.batch
    if a.max_steps:
        batches = min(batches, a.max_steps)
    per_epoch = max(1, batches * a.batch)
    return max(a.epochs, math.ceil(a.epochs_from_steps / per_epoch))


@torch.no_grad()
def predict(model, loader, device, amp, mf=torch.contiguous_format):
    model.eval()
    rows = []
    for b in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            xb = b["x"].to(device, non_blocking=True)
            if xb.dim() == 4:
                xb = xb.contiguous(memory_format=mf)
            logit = model(xb).float().cpu()
        for i in range(len(b["id"])):
            rows.append({"id": b["id"][i], "patient_id": b["patient_id"][i], "breast_id": b["breast_id"][i],
                         "label": int(b["y"][i]), "logit": float(logit[i]),
                         "prob": float(torch.sigmoid(logit[i]))})
    return pd.DataFrame(rows)


def quick_metrics(p: pd.DataFrame) -> dict:
    out = {"auc_unit": fast_auc(p.label, p.prob)}
    for lvl in ("breast", "patient"):
        g = aggregate(p, lvl)
        out[f"auc_{lvl}"] = fast_auc(g.label, g.prob)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--mode", choices=["cv", "final"], default="cv")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--backbone", default="effb0")
    ap.add_argument("--head", choices=["gap", "attnmil"], default="gap")
    ap.add_argument("--unit", choices=["image", "breast"], default="image",
                    help="breast = two-view model (CC+MLO, learned fusion)")
    ap.add_argument("--arm", choices=ARMS, default="none")
    ap.add_argument("--bank", type=Path, default=None, help="ROI bank JSON (needed for roi and randcrop)")
    ap.add_argument("--p-crop", type=float, default=0.1)
    ap.add_argument("--jitter", type=float, default=0.1)
    ap.add_argument("--roi-k", type=int, default=5)
    ap.add_argument("--img-h", type=int, default=896)  # breast crops are ~2:1 (h:w)
    ap.add_argument("--img-w", type=int, default=448)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    ap.add_argument("--pos-weight", default="auto", help="'auto' = negatives/positives in training data")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--grad-ckpt", type=int, default=0, help="gradient checkpointing (saves memory)")
    ap.add_argument("--channels-last", type=int, default=1, help="NHWC memory layout (faster convnets on GPU)")
    ap.add_argument("--deterministic", type=int, default=0)
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--save-ckpt", type=int, default=-1, help="-1 = only in final mode")
    ap.add_argument("--out-root", type=Path, default=Path("runs"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--crop-fit", choices=["letterbox", "fill"], default="letterbox",
                    help="fill: widen roi/randcrop boxes to the input aspect ratio (addendum sensitivity analysis)")
    ap.add_argument("--study", choices=["main", "addendum", "control", "convergence"], default="main",
                    help="addendum runs are kept out of the pre-registered main analysis")
    ap.add_argument("--epochs-from-steps", type=int, default=0,
                    help="addendum C: minimum number of training samples presented; the number of epochs becomes "
                         "max(--epochs, ceil(N / samples per epoch)), computed from this run's training set")
    ap.add_argument("--eval-metrics", type=int, default=1,
                    help="0 = save held-out predictions but compute and log no held-out metric (blinded runs)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop each epoch after N batches (testing)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    if a.arm in ("roi", "randcrop") and not a.bank:
        sys.exit("--bank is required for the roi and randcrop arms")
    if a.epochs_from_steps:
        a.epochs = epochs_for_samples(a)
    out = a.out_root / run_name(a)
    if (out / "DONE").exists():
        print(f"[skip] {out} already finished")
        return
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed, bool(a.deterministic))
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")

    index = pd.read_csv(a.data_dir / "index.csv")
    split = pd.read_csv(a.data_dir / "split.csv")
    df = index.merge(split, on="patient_id")
    dev = df[df.split == "dev"]
    oof = a.mode == "final" and not (df.split == "test").any()
    if a.mode == "cv" or oof:
        tr, ev = dev[dev.fold != a.fold], dev[dev.fold == a.fold]
    else:
        tr, ev = dev, df[df.split == "test"]
    if ev.empty:
        sys.exit(f"empty evaluation set (fold {a.fold}?)")
    assert not set(tr.patient_id) & set(ev.patient_id), "patient overlap between train and evaluation"

    bank = json.load(open(a.bank))["boxes"] if a.bank else None
    common = dict(data_dir=a.data_dir, img_hw=(a.img_h, a.img_w), unit=a.unit, seed=a.seed)
    tr_ds = MammoDataset(tr, train=True, arm=a.arm, p_crop=a.p_crop, jitter=a.jitter, roi_bank=bank,
                         roi_k=a.roi_k, crop_fit=a.crop_fit, **common)
    ev_ds = MammoDataset(ev, train=False, **common)
    g = torch.Generator().manual_seed(a.seed)
    pin = device.type == "cuda"
    tr_dl = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate,
                       pin_memory=pin, drop_last=True, generator=g, persistent_workers=False)
    ev_dl = DataLoader(ev_ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate,
                       pin_memory=pin)

    ytr = tr.groupby("breast_id").label.max() if a.unit == "breast" else tr.label
    pw = (ytr == 0).sum() / max((ytr == 1).sum(), 1) if a.pos_weight == "auto" else float(a.pos_weight)
    model = Classifier(a.backbone, a.head, twoview=a.unit == "breast", pretrained=bool(a.pretrained)).to(device)
    if a.grad_ckpt:
        model.set_grad_checkpointing(True)
    mf = torch.channels_last if (a.channels_last and device.type == "cuda") else torch.contiguous_format
    model = model.to(memory_format=mf)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = (a.max_steps or len(tr_dl)) // a.accum
    total = max(1, steps_per_epoch * a.epochs)
    warm = max(1, int(a.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device, dtype=torch.float32))
    use_amp = bool(a.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cm_rng = np.random.default_rng([a.seed, 99])

    cfg = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "run_name": out.name, "n_train": len(tr_ds), "n_eval": len(ev_ds),
           "n_train_patients": int(tr.patient_id.nunique()), "n_eval_patients": int(ev.patient_id.nunique()),
           "pos_weight_value": float(pw), "eval_set": ("fold%d" % a.fold) if (a.mode == "cv" or oof) else "test",
           "oof_fold": a.fold if oof else None}
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "platform": platform.platform()}
    try:
        import timm
        env["timm"] = timm.__version__
    except Exception:
        pass
    json.dump(env, open(out / "env.json", "w"), indent=1)
    print(f"[run] {out.name} | train {len(tr_ds)} ({cfg['n_train_patients']} pts) | eval {len(ev_ds)} "
          f"({cfg['n_eval_patients']} pts) | device {env['device']} | pos_weight {pw:.2f}")

    log = []
    imgs_per_sample = 2 if a.unit == "breast" else 1
    for ep in range(1, a.epochs + 1):
        tr_ds.set_epoch(ep)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0, wait, n, loss_sum = time.perf_counter(), 0.0, 0, 0.0
        tw = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(tr_dl):
            wait += time.perf_counter() - tw
            if a.max_steps and i >= a.max_steps:
                break
            x, y = b["x"].to(device, non_blocking=True), b["y"].to(device, non_blocking=True)
            if x.dim() == 4:
                x = x.contiguous(memory_format=mf)
            if a.arm == "cutmix":
                x, y = cutmix(x, y, cm_rng)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = crit(model(x).float(), y) / a.accum
            scaler.scale(loss).backward()
            if (i + 1) % a.accum == 0:
                scale0 = scaler.get_scale() if use_amp else None
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if not use_amp or scaler.get_scale() >= scale0:  # skipped steps (inf grads) do not advance the LR
                    sched.step()
            loss_sum += loss.item() * a.accum * len(y)
            n += len(y)
            tw = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.perf_counter() - t0
        mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
        row = {"epoch": ep, "loss": loss_sum / max(n, 1), "train_s": t_train, "data_wait_s": wait,
               "train_img_per_s": n * imgs_per_sample / t_train, "peak_alloc_mib": mem,
               "lr_end": sched.get_last_lr()[0]}
        if a.mode == "cv" or ep == a.epochs:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t1 = time.perf_counter()
            p = predict(model, ev_dl, device, use_amp, mf)
            row["eval_s"] = time.perf_counter() - t1
            row["eval_img_per_s"] = len(p) * imgs_per_sample / row["eval_s"]
            row["eval_peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
            if a.eval_metrics:
                row.update(quick_metrics(p))
            if a.mode == "cv":
                p.to_csv(out / f"preds_ep{ep:02d}.csv", index=False)
            if ep == a.epochs:
                p.to_csv(out / "preds.csv", index=False)
        log.append(row)
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)
        msg = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        print(f"[ep {ep}/{a.epochs}] {msg}", flush=True)

    if a.save_ckpt == 1 or (a.save_ckpt == -1 and a.mode == "final"):
        torch.save(model.state_dict(), out / "last.pt")
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
