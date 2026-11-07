"""Hand strength: preflop features (fitted offline) and Monte Carlo equity."""
import math
from collections import Counter
from itertools import combinations

from evaluator import omaha_score

try:  # numpy is in the tournament base image; without it (or without get_table) use the pure-Python path
    import numpy as np
    from evaluator import get_table
except ImportError:  # pragma: no cover
    np = None

FEATURES = ("pairs", "trips", "suits2", "suits4", "ace_suited", "conn", "high")
_PAIRS = np.array(list(combinations(range(5), 2))) if np else None
_TRIPLES = np.array(list(combinations(range(5), 3))) if np else None
_NP = {}  # built once by _np_tables(): colex-rank -> score array and binomial table


def preflop_features(hole) -> list[float]:
    ranks = [c >> 2 for c in hole]
    rc, sc = Counter(ranks), Counter(c & 3 for c in hole)
    rs = set(ranks) | ({-1} if 8 in ranks else set())  # ace also plays low (A-6-7-8-9)
    conn = max(sum(r in rs for r in range(lo, lo + 5)) for lo in range(-1, 5))
    return [
        sum(v >= 2 for v in rc.values()),
        float(max(rc.values()) >= 3),
        sum(v >= 2 for v in sc.values()),
        float(max(sc.values()) >= 4),
        float(any(c >> 2 == 8 and sc[c & 3] >= 2 for c in hole)),
        conn / 5,
        sum(ranks) / 40,
    ]


def preflop_equity(hole, weights: dict) -> float:
    f = preflop_features(hole)
    return min(0.95, max(0.05, weights["bias"] + sum(weights[k] * v for k, v in zip(FEATURES, f))))


def _np_tables():
    """Score of every 5-card hand in an int32 array indexed by the hand's combination rank
    (sum of C(card_i, i+1) over the ascending cards), so scoring is one array lookup."""
    if not _NP:
        binom = np.array([[math.comb(n, k) for n in range(36)] for k in range(6)], dtype=np.int64)
        combos = np.array(list(combinations(range(36), 5)), dtype=np.int64)
        rank = sum(binom[i + 1][combos[:, i]] for i in range(5))
        table = get_table()
        masks = (np.int64(1) << combos).sum(axis=1).tolist()
        score = np.empty(len(combos), dtype=np.int32)
        score[rank] = np.fromiter((table[m] for m in masks), dtype=np.int32, count=len(masks))
        _NP.update(binom=binom, score=score)
    return _NP["binom"], _NP["score"]


def warmup_np() -> None:
    global np
    if np is not None:
        try:
            _np_tables()
        except Exception:  # the evaluator changed under us: never fail at startup, use the pure-Python path
            np = None


def omaha_scores(hole, board):
    """omaha_score for many hands at once: hole and board are int arrays of shape (n, 5)."""
    binom, score = _np_tables()
    n = hole.shape[0]
    h = np.broadcast_to(hole[:, _PAIRS][:, :, None, :], (n, 10, 10, 2))
    b = np.broadcast_to(board[:, _TRIPLES][:, None, :, :], (n, 10, 10, 3))
    five = np.sort(np.concatenate([h, b], axis=3), axis=3)
    idx = sum(binom[i + 1][five[..., i]] for i in range(5))
    return score[idx].reshape(n, 100).max(axis=1)


def mc_equity(hole, board, n: int, rng) -> float:
    """Equity vs one random hand, sampling the rest of the board (vectorised with numpy)."""
    if np is None:
        return _mc_equity_py(hole, board, n, rng)
    deck = np.array([c for c in range(36) if c not in set(hole) | set(board)])
    need = 5 - len(board)
    g = np.random.default_rng(rng.getrandbits(64))
    draw = deck[g.random((n, len(deck))).argsort(axis=1)[:, :5 + need]]  # distinct cards per sample
    full = np.concatenate([np.broadcast_to(np.array(board, dtype=np.int64), (n, len(board))), draw[:, 5:]], axis=1)
    me = omaha_scores(np.broadcast_to(np.array(hole, dtype=np.int64), (n, 5)), full)
    op = omaha_scores(draw[:, :5], full)
    return float(((me > op) + 0.5 * (me == op)).mean())


def _mc_equity_py(hole, board, n: int, rng) -> float:
    """Pure-Python version (reference and fallback)."""
    dead = set(hole) | set(board)
    deck = [c for c in range(36) if c not in dead]
    board, need = tuple(board), 5 - len(board)
    mine = omaha_score(hole, board) if need == 0 else None
    wins = 0.0
    for _ in range(n):
        s = rng.sample(deck, 5 + need)
        b = board + tuple(s[5:])
        me = mine if mine is not None else omaha_score(hole, b)
        op = omaha_score(s[:5], b)
        wins += 1.0 if me > op else 0.5 if me == op else 0.0
    return wins / n
