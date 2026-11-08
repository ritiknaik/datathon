"""Train the decision ensemble on rows from train/policy_data.py and export bot/policy_weights.npz.

    python train/policy_train.py data/policy/pilot.npz --out bot/policy_weights.npz
Members: ridge regression + N bagged numpy MLPs (30 -> 64 -> 64 -> 4). Target = advantage per legal action
(q minus the mean over legal actions), illegal actions masked out of the loss. Train/test split BY SEED.
Reference implementation for the Databricks run (databricks/ENSEMBLE_PLAN.md, prompt G4).
"""
import argparse
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_data import offline_value, strategy  # noqa: E402


def load(path):
    d = np.load(path)
    q = d["q"]
    mask = ~np.isnan(q)
    adv = np.where(mask, q - np.nanmean(q, axis=1, keepdims=True), 0.0)
    test = np.array([zlib.crc32(s.encode()) % 5 == 0 for s in d["seed"]])  # 20% of seeds held out
    return d, d["features"], adv, mask, test


def ridge(X, Y, M, lam=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    W = np.zeros((X.shape[1] + 1, 4))
    for j in range(4):
        A = Xb[M[:, j]]
        W[:, j] = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y[M[:, j], j])
    return [(W[:-1], W[-1])]


def mlp(X, Y, M, seed, hidden=64, epochs=40, lr=1e-3, l2=1e-4, batch=256):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), len(X))  # bootstrap sample (bagging)
    X, Y, M = X[idx], Y[idx], M[idx]
    sizes = [X.shape[1], hidden, hidden, 4]
    P = []
    for i in range(3):
        P += [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])), np.zeros(sizes[i + 1])]
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            b = order[s:s + batch]
            a0 = X[b]
            z1 = a0 @ P[0] + P[1]
            a1 = np.maximum(z1, 0)
            z2 = a1 @ P[2] + P[3]
            a2 = np.maximum(z2, 0)
            out = a2 @ P[4] + P[5]
            g = 2 * (out - Y[b]) * M[b] / max(1, M[b].sum())
            grads = [None] * 6
            grads[4], grads[5] = a2.T @ g, g.sum(0)
            g = (g @ P[4].T) * (z2 > 0)
            grads[2], grads[3] = a1.T @ g, g.sum(0)
            g = (g @ P[2].T) * (z1 > 0)
            grads[0], grads[1] = a0.T @ g, g.sum(0)
            t += 1
            for i in range(6):
                gi = grads[i] + (l2 * P[i] if i % 2 == 0 else 0)
                m[i] = 0.9 * m[i] + 0.1 * gi
                v[i] = 0.999 * v[i] + 0.001 * gi ** 2
                P[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return [(P[0], P[1]), (P[2], P[3]), (P[4], P[5])]


def forward(layers, X):
    a = X
    for i, (W, b) in enumerate(layers):
        a = a @ W + b
        if i < len(layers) - 1:
            a = np.maximum(a, 0)
    return a


def rows_of(d, sel):
    return [dict(q=list(d["q"][i]), ev=list(d["ev"][i]), pot=float(d["pot"][i])) for i in np.where(sel)[0]]


def report(name, pred, Y, M, d, test, temp):
    mae = np.abs(pred - Y)[M].mean()
    big = np.where(M, pred, -np.inf).argmax(1) == np.where(M, Y, -np.inf).argmax(1)
    rows = rows_of(d, test)
    vals = {w: offline_value(rows, pred, w, temp) for w in (0.0, 0.25, 0.5, 1.0, 2.0)}
    print(f"{name:10} adv MAE {mae:6.2f} | best-action match {big.mean():.1%} | offline value (chips) by w: "
          + "  ".join(f"{w}: {v:+.3f}" for w, v in vals.items()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="data/policy/policy_weights.npz")
    ap.add_argument("--bags", type=int, default=3)
    a = ap.parse_args()
    d, X, Y, M, test = load(a.data)
    mu, sd = X[~test].mean(0), X[~test].std(0)
    sd[sd < 1e-6] = 1.0
    Z = (X - mu) / sd
    tr = ~test
    print(f"train rows {tr.sum()}, test rows {test.sum()} (split by seed)")
    temp = strategy.load_params()["temperature"]
    members = {"lin": ridge(Z[tr], Y[tr], M[tr])}
    for i in range(a.bags):
        members[f"mlp{i}"] = mlp(Z[tr], Y[tr], M[tr], seed=i + 1)
    preds = {k: forward(v, Z[test]) for k, v in members.items()}
    for k, p in preds.items():
        report(k, p, Y[test], M[test], d, test, temp)
    report("ensemble", np.mean(list(preds.values()), axis=0), Y[test], M[test], d, test, temp)
    out = {"mu": mu, "sd": sd, "members": np.array(list(members)), "alpha": np.ones(len(members))}
    for k, layers in members.items():
        for i, (W, b) in enumerate(layers):
            out[f"{k}/W{i}"], out[f"{k}/b{i}"] = W, b
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, **out)
    print("wrote", a.out)
