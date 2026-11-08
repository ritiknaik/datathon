"""Learned action-value net (ported from spoc_db): q_features -> expected bb from now for
FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT. numpy only.

Default weights bot/q_weights.npz; a params file may name another file ("qnet_file", relative to bot/),
so different nets can play each other in one process. A missing/broken file is never an error:
predict() raises and the caller keeps its formula EV.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}


def _load(name):
    if name not in _CACHE:
        _CACHE[name] = dict(np.load(os.path.join(HERE, name)))
    return _CACHE[name]


try:
    _load("q_weights.npz")
    LOADED = True
except Exception:
    LOADED = False


def predict(features, name="q_weights.npz"):
    W = _load(name)
    x = (np.asarray(features, dtype=np.float32) - W["mean"]) / W["std"]
    i = 0
    while f"W{i}" in W:
        x = x @ W[f"W{i}"] + W[f"b{i}"]
        if f"W{i + 1}" in W:
            x = np.maximum(x, 0.0)
        i += 1
    return x.tolist()
