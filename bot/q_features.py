"""State features for the learned action-value net (shared by the bot and the training code).

Keep this the single source of truth: training data and live play must build features identically.
"""
from collections import Counter

from game import RAISE_ACTIONS, Action, Street

from .equity import preflop_features

ACTIONS = (Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)  # net output order
STACK = 80.0


def _board_texture(board):
    if not board:
        return [0.0, 0.0, 0.0, 0.0]
    ranks = [c >> 2 for c in board]
    rs = set(ranks) | ({-1} if 8 in ranks else set())
    return [
        float(max(Counter(ranks).values()) >= 2),                                   # paired board
        max(Counter(c & 3 for c in board).values()) / 5,                            # flush pressure
        max(sum(r in rs for r in range(lo, lo + 5)) for lo in range(-1, 5)) / 5,   # straight pressure
        max(ranks) / 8,                                                             # high card
    ]


def build(obs, eq, agg, fold, opp_acts):
    """Fixed-length feature list for one decision. eq = raw equity before any bet discount."""
    pot, tc = obs.pot, obs.to_call
    opp = [a for a in obs.action_history if not a.is_me]
    opp_street = [a for a in opp if a.street == obs.street]
    last = opp[-1].action if opp else None
    raise_frac = [((obs.raise_to.get(a, obs.my_bet) - obs.my_bet) / max(pot, 1)) for a in RAISE_ACTIONS]
    return ([float(obs.street == s) for s in Street]
            + [eq] + preflop_features(obs.hole_cards) + _board_texture(obs.board)
            + [pot / (2 * STACK), tc / max(pot, 1), tc / (pot + tc) if tc else 0.0,
               obs.my_stack / STACK, obs.opp_stack / STACK, min(obs.my_stack, obs.opp_stack) / max(pot, 1),
               obs.raises_this_street / 4, float(obs.is_button)]
            + [float(last == a) for a in (None, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)]
            + [sum(a.action in RAISE_ACTIONS for a in opp) / 4, sum(a.action in RAISE_ACTIONS for a in opp_street) / 4]
            + [agg, fold, min(opp_acts, 400) / 400]
            + raise_frac
            + [float(a in obs.legal_actions) for a in ACTIONS])


def legal_mask(obs):
    return [a in obs.legal_actions for a in ACTIONS]
