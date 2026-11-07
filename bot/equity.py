"""Hand strength: preflop features (fitted offline) and Monte Carlo equity."""
from collections import Counter

from evaluator import omaha_score

FEATURES = ("pairs", "trips", "suits2", "suits4", "ace_suited", "conn", "high")


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


def mc_equity(hole, board, n: int, rng) -> float:
    """Equity vs one random hand, sampling the rest of the board."""
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
