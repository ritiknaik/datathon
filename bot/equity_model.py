"""Numpy-only fast equity approximation (exported from the Databricks training run).

predict(hole_cards, board) mirrors bot.equity.mc_equity(hole, board, n, rng):
    5 hole-card ints + a board of 0/3/4/5 card ints -> equity vs one random hand, in [0, 1].
Trained on 10k MC-labeled deals (n=2000) x verified suit/order augmentations.
Held-out MAE vs MC: 0.045 overall (preflop 0.019 / flop 0.041 / turn 0.050 / river 0.069).
~0.3 ms per call on 1 CPU. NOT yet wired into strategy.py.
"""
import os
from collections import Counter
from itertools import combinations

import numpy as np

_W = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "equity_weights.npz"))
_MU, _SD = _W["mu"], _W["sd"]


def featurize(hole, board):
    f = []
    ranks_h = [c >> 2 for c in hole]; suits_h = [c & 3 for c in hole]
    ranks_b = [c >> 2 for c in board]; suits_b = [c & 3 for c in board]
    for src in (ranks_h, ranks_b):
        h = [0.0] * 9
        for r in src: h[r] += 1
        f += [x / 2 for x in h]
    for src in (suits_h, suits_b):
        s = [0.0] * 4
        for x in src: s[x] += 1
        f += [x / 5 for x in s]
    rc_h, rc_b = Counter(ranks_h), Counter(ranks_b)
    f += [sum(v >= 2 for v in rc_h.values()),
          float(max(rc_h.values()) >= 3),
          sum(v >= 2 for v in rc_b.values()),
          sum(v >= 3 for v in rc_b.values()),
          float(max(rc_b.values()) >= 4 if rc_b else 0)]
    def conn(rs):
        s = set(rs)
        if 8 in s: s.add(-1)
        return max(sum(r in s for r in range(lo, lo + 5)) for lo in range(-1, 5)) / 5
    f += [conn(ranks_h), conn(ranks_h + ranks_b), sum(ranks_h) / 40, max(ranks_h) / 8]
    f += [sum(min(v, rc_b.get(r, 0)) for r, v in rc_h.items()) / 3,
          max(v * rc_b.get(r, 0) for r, v in rc_h.items()) / 4]
    sc_h, sc_b = Counter(suits_h), Counter(suits_b)
    f += [max(sc_h[s] + sc_b.get(s, 0) for s in sc_h) / 8]
    f += [{0: 0, 3: 1, 4: 2, 5: 3}[len(board)] == i for i in range(4)]
    best1 = best2 = 0.0
    straight_share = max_suit = max_window = 0.0
    fd_share = sd_share = 0.0
    nb = len(board)
    if nb >= 3:
        n_combo = n_straight = n_fd = n_sd = 0
        for i, j in combinations(range(5), 2):
            for t in combinations(range(nb), 3):
                cards = [hole[i], hole[j]] + [board[k] for k in t]
                r5 = sorted((c >> 2 for c in cards), reverse=True)
                sc = Counter(c & 3 for c in cards)
                flush_len = max(sc.values())
                m = len(set(r5))
                if m == 5:
                    is_str = r5[0] - r5[4] == 4 or tuple(r5) == (8, 3, 2, 1, 0)
                    cat = (8 if is_str else 6) if flush_len == 5 else (3 if is_str else 0)
                else:
                    cnt = Counter(r5); mx = max(cnt.values())
                    cat = 7 if mx == 4 else 5 if mx == 3 and len(cnt) == 2 else 4 if mx == 3 else 2 if len(cnt) == 3 else 1
                sc_cat = (cat + r5[0] / 9) / 9
                if sc_cat > best1: best1, best2 = sc_cat, best1
                elif sc_cat > best2: best2 = sc_cat
                s = set(r5)
                if 8 in s: s.add(-1)
                w = max(sum(r in s for r in range(lo, lo + 5)) for lo in range(-1, 5))
                n_combo += 1
                if cat >= 3: n_straight += 1
                if flush_len == 4: n_fd += 1
                if w == 4: n_sd += 1
                max_suit = max(max_suit, flush_len / 5)
                max_window = max(max_window, w / 5)
        straight_share = n_straight / n_combo; fd_share = n_fd / n_combo; sd_share = n_sd / n_combo
    f += [best1, best2, best1 - best2, straight_share, max_suit, max_window, fd_share, sd_share]
    return np.asarray(f, dtype=np.float32)


def predict(hole_cards, board) -> float:
    x = (featurize(list(hole_cards), list(board)) - _MU) / _SD
    a = x
    for l in range(3):
        z = a @ _W[f"W{l}"] + _W[f"b{l}"]
        a = z if l == 2 else np.maximum(z, 0)
    return float(np.clip(a[0], 0.0, 1.0))
