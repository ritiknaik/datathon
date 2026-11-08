"""Learned action-value net: features -> expected bb from now for FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT.

Weights come from train/train_q.py (bot/q_weights.npz). numpy only, ~50 us per call.
If the file is missing or broken, LOADED is False and the bot uses its formula (never an error).
"""
import os

try:
    import numpy as np
    _W = dict(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_weights.npz")))
    LOADED = True
except Exception:
    _W, LOADED = {}, False


def predict(features):
    """Return 4 floats (bb from now per action) for one feature list."""
    x = (np.asarray(features, dtype=np.float32) - _W["mean"]) / _W["std"]
    i = 0
    while f"W{i}" in _W:
        x = x @ _W[f"W{i}"] + _W[f"b{i}"]
        if f"W{i + 1}" in _W:
            x = np.maximum(x, 0.0)  # ReLU on hidden layers
        i += 1
    return x.tolist()
