"""Features for the learned decision ensemble (see databricks/ENSEMBLE_PLAN.md).

ONE definition, used both when the training data is generated (train/policy_data.py) and when the bot
plays (bot/strategy.py), so the model always sees the same numbers. Do not re-implement it elsewhere.
Action order everywhere: FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT (= int(Action)).
"""
from game import Action, Street

N_FEATURES = 30
STREETS = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)


def _board(board) -> list[float]:
    ranks = [c >> 2 for c in board]
    suits = [c & 3 for c in board]
    rs = set(ranks) | ({-1} if 8 in ranks else set())  # the ace also plays low (A-6-7-8-9)
    window = max((sum(r in rs for r in range(lo, lo + 5)) for lo in range(-1, 5)), default=0)
    top_suit = max((suits.count(s) for s in range(4)), default=0)
    return [float(len(ranks) != len(set(ranks))), float(top_suit >= 3), top_suit / 5,
            float(window >= 3), len(board) / 5]


def build(obs, eq: float, ev: dict, agg: float, fold: float) -> list[float]:
    """30 numbers describing one decision: strength, price, situation, opponent, board."""
    pot, tc = obs.pot, obs.to_call
    total = max(1, obs.my_stack + obs.opp_stack + pot)          # all chips in play, for scaling
    need = tc / (pot + tc) if tc else 0.0
    last = 0.0
    for a in obs.action_history:
        if a.street == obs.street and a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            last = 1.0 if a.is_me else -1.0                     # who raised last on this street
    f = [eq, eq - need]
    f += [ev[a] / total if a in ev else 0.0 for a in Action]     # rule EV per action (4)
    f += [float(a in ev) for a in Action]                        # legal mask (4)
    f += [tc / max(pot, 1), pot / total, obs.my_stack / total, obs.opp_stack / total]
    f += [float(obs.street == s) for s in STREETS]               # street one-hot (4)
    f += [float(obs.is_button), obs.raises_this_street / 4, last, len(obs.action_history) / 10]
    f += [agg, fold, obs.hand_index / 1000]
    f += _board(obs.board)                                       # board texture (5)
    return f
