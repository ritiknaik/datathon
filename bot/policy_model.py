"""Numpy-only inference for the learned decision ensemble (see databricks/ENSEMBLE_PLAN.md).

predict(features) -> 4 advantages in chips (FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT); the caller
centres them over the legal actions and blends them with the rule EV.

Weights file bot/policy_weights.npz (allow_pickle=False), written by train/policy_train.py or Databricks:
    mu, sd                 feature mean and scale, shape (30,)
    members                string array of member names, e.g. ["lin", "mlp0", "mlp1"]
    alpha                  float array, one blend weight per member (they are normalised to sum to 1)
    <name>/W0, <name>/b0, <name>/W1, ...   dense layers; ReLU between layers, linear output of size 4
Importing this module raises if the weights are missing; bot/strategy.py then runs without it.
"""
import os

import numpy as np

_W = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_weights.npz"))
_MU, _SD = _W["mu"], _W["sd"]
_ALPHA = _W["alpha"] / _W["alpha"].sum()
_MEMBERS = []
for _name in _W["members"]:
    _n = sum(1 for k in _W.files if k.startswith(f"{_name}/W"))
    _MEMBERS.append([(_W[f"{_name}/W{i}"], _W[f"{_name}/b{i}"]) for i in range(_n)])


def predict(features) -> np.ndarray:
    x = (np.asarray(features, dtype=np.float64) - _MU) / _SD
    out = np.zeros(4)
    for alpha, layers in zip(_ALPHA, _MEMBERS):
        a = x
        for i, (W, b) in enumerate(layers):
            a = a @ W + b
            if i < len(layers) - 1:
                a = np.maximum(a, 0.0)
        out += alpha * a
    return out
