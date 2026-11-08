"""Train the action-value net on gen_q.py data and export numpy weights for the bot (training only).

    python train/train_q.py --data "data/q/*.npz" --out bot/q_weights.npz --epochs 15

Loss: Huber on the action actually taken (other outputs get no gradient). FOLD rows are skipped: folding is
exactly 0 from now. Holdout = whole files listed in --val (or the last file), so no hand leaks into both sets.
Reports holdout loss per opponent so we can see whether it generalises (leave-one-opponent-out: --exclude).
"""
import argparse
import glob

import numpy as np
import torch
from torch import nn


def load(files, exclude=()):
    parts = [np.load(f) for f in files]
    X = np.concatenate([p["X"] for p in parts]); A = np.concatenate([p["A"] for p in parts]).astype(np.int64)
    Y = np.concatenate([p["Y"] for p in parts]); O = np.concatenate([[str(p["names"][o]) for o in p["O"]] for p in parts])
    keep = (A != 0) & ~np.isin(O, list(exclude))
    return X[keep], A[keep], Y[keep], O[keep]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/q/*.npz")
    ap.add_argument("--val", nargs="*", help="holdout files (default: the last file)")
    ap.add_argument("--exclude", nargs="*", default=[], help="opponents left out of training (generalisation test)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="bot/q_weights.npz")
    a = ap.parse_args()
    files = sorted(glob.glob(a.data))
    val_files = a.val or files[-1:]
    train_files = [f for f in files if f not in val_files]
    X, A, Y, _ = load(train_files, a.exclude)
    Xv, Av, Yv, Ov = load(val_files)
    mean, std = X.mean(0), X.std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"train {len(Y)} rows from {len(train_files)} files, holdout {len(Yv)} rows, device {dev}")

    t = lambda z, dt=torch.float32: torch.tensor(z, dtype=dt, device=dev)  # noqa: E731
    X, A, Y = t((X - mean) / std), t(A, torch.long), t(Y)
    Xv, Av, Yv = t((Xv - mean) / std), t(Av, torch.long), t(Yv)
    h = a.hidden
    net = nn.Sequential(nn.Linear(X.shape[1], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 4)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=10.0, reduction="none")

    def evaluate(x, act, y):
        with torch.no_grad():
            return loss_fn(net(x).gather(1, act[:, None])[:, 0], y)

    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Y), device=dev)
        for i in range(0, len(Y), a.batch):
            idx = perm[i:i + a.batch]
            loss = loss_fn(net(X[idx]).gather(1, A[idx, None])[:, 0], Y[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        base = loss_fn(torch.full_like(Yv, float(Y.mean())), Yv).mean().item()  # predict-the-average baseline
        print(f"epoch {ep + 1}: holdout loss {evaluate(Xv, Av, Yv).mean().item():.3f} (baseline {base:.3f})", flush=True)

    per = evaluate(Xv, Av, Yv).cpu().numpy()
    for name in sorted(set(Ov)):
        s = Ov == name
        print(f"  holdout {name:<22} rows {s.sum():>7}  loss {per[s].mean():.3f}"
              + ("   <- never seen in training" if name in a.exclude else ""))
    lin = [m for m in net if isinstance(m, nn.Linear)]
    w = {f"W{i}": m.weight.detach().cpu().numpy().T.astype(np.float32) for i, m in enumerate(lin)}
    w.update({f"b{i}": m.bias.detach().cpu().numpy().astype(np.float32) for i, m in enumerate(lin)})
    np.savez(a.out, mean=mean.astype(np.float32), std=std.astype(np.float32), **w)
    print(f"saved {a.out}")
