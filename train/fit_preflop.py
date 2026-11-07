"""Fit preflop_equity weights: MC equity of random hands vs a random hand, linear least squares.
Writes the weights into bot/params.json. Scale --hands/--samples up on Grace.
    python train/fit_preflop.py --hands 3000 --samples 200
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import warmup  # noqa: E402
from bot.equity import FEATURES, mc_equity, preflop_features  # noqa: E402
from bot.strategy import HERE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hands", type=int, default=3000)
ap.add_argument("--samples", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
warmup()
rng = random.Random(args.seed)
X, y = [], []
for _ in range(args.hands):
    hole = sorted(rng.sample(range(36), 5))
    X.append([1.0] + preflop_features(hole))
    y.append(mc_equity(hole, (), args.samples, rng))
X, y = np.array(X), np.array(y)
w, *_ = np.linalg.lstsq(X, y, rcond=None)
r2 = 1 - ((X @ w - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"R^2 = {r2:.3f}")

path = os.path.join(HERE, "params.json")
with open(path, encoding="utf-8") as f:
    P = json.load(f)
P["preflop"] = {"bias": round(float(w[0]), 4), **{k: round(float(v), 4) for k, v in zip(FEATURES, w[1:])}}
with open(path, "w", encoding="utf-8") as f:
    json.dump(P, f, indent=2)
print(P["preflop"])
