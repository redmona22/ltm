"""Pretrain LTM v0 on synthetic tables from the SCM prior.

    python -m ltm.train --steps 5000 --out ltm_v0.pt
"""
import argparse
import time

import numpy as np
import torch

from .model import LTM, episode_loss
from .prior import generate_table


def make_episode(rng, max_rows, max_feats):
    n_rows = int(np.exp(rng.uniform(np.log(50), np.log(max_rows))))
    n_feats = int(np.exp(rng.uniform(np.log(2), np.log(max_feats))))
    task = "classification" if rng.random() < 0.6 else "regression"
    t = generate_table(rng, n_rows, n_feats, task)
    perm = rng.permutation(n_rows)
    n_ctx = max(10, int(0.7 * n_rows))
    X = torch.tensor(t.X[perm], dtype=torch.float32)
    y = torch.tensor(t.y[perm], dtype=torch.long if task == "classification" else torch.float32)
    return X, torch.tensor(t.cat_mask), y, n_ctx, task, t.n_classes


@torch.no_grad()
def evaluate(model, n_tables=30, seed=12345, max_rows=300, max_feats=20):
    """Compare LTM with HistGradientBoosting on fresh prior tables."""
    from sklearn.ensemble import HistGradientBoostingClassifier as C
    from sklearn.ensemble import HistGradientBoostingRegressor as R
    from sklearn.metrics import accuracy_score, r2_score
    rng = np.random.default_rng(seed)
    model.eval()
    res = {"classification": ([], []), "regression": ([], [])}
    for _ in range(n_tables):
        X, cm, y, n_ctx, task, k = make_episode(rng, max_rows, max_feats)
        Xn, yn = X.numpy(), y.numpy()
        if task == "classification" and len(np.unique(yn[:n_ctx])) < 2:
            continue
        out = model(X, cm, y[:n_ctx], task, k)
        gb = (C() if task == "classification" else R()).fit(Xn[:n_ctx], yn[:n_ctx])
        if task == "classification":
            res[task][0].append(accuracy_score(yn[n_ctx:], out.argmax(1).numpy()))
            res[task][1].append(accuracy_score(yn[n_ctx:], gb.predict(Xn[n_ctx:])))
        else:
            ym, ys = yn[:n_ctx].mean(), max(yn[:n_ctx].std(), 1e-6)
            res[task][0].append(r2_score(yn[n_ctx:], out.numpy() * ys + ym))
            res[task][1].append(r2_score(yn[n_ctx:], gb.predict(Xn[n_ctx:])))
    model.train()
    for task, (a, b) in res.items():
        metric = "accuracy" if task == "classification" else "R2"
        print(f"  {task:14s} {metric}: LTM {np.mean(a):.3f} | HistGB {np.mean(b):.3f} (n={len(a)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--max_rows", type=int, default=300)
    p.add_argument("--max_feats", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="ltm_v0.pt")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = LTM(d=a.d, layers=a.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    hist = {"classification": [], "regression": []}
    t0 = time.time()
    for step in range(1, a.steps + 1):
        for _ in range(a.accum):
            X, cm, y, n_ctx, task, k = make_episode(rng, a.max_rows, a.max_feats)
            loss, _ = episode_loss(model, X, cm, y, n_ctx, task, k)
            (loss / a.accum).backward()
            hist[task].append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if step % 50 == 0:
            c, r = np.mean(hist["classification"][-100:]), np.mean(hist["regression"][-100:])
            print(f"step {step} | cls loss {c:.3f} | reg mse {r:.3f} (1.0 = predict mean) | {time.time() - t0:.0f}s")
        if step % 500 == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "d": a.d, "layers": a.layers}, a.out)
            evaluate(model)


if __name__ == "__main__":
    main()
