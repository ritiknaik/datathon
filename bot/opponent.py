"""Opponent model: smoothed counts of aggression and folding to our raises."""
from game import RAISE_ACTIONS, Action


def init(memory: dict) -> None:
    memory["opp"] = {"acts": 0, "raises": 0, "faced": 0, "folded": 0}
    memory["seen"] = (-1, 0)          # (hand_index, opp history entries processed)
    memory["pending_raise"] = False   # our last action was a raise, opp not yet responded


def observe(obs, memory: dict) -> None:
    """Count opponent actions we haven't seen yet; resolve our pending raise."""
    o = memory["opp"]
    hand, done = memory["seen"]
    if hand != obs.hand_index:
        done = 0
    opp_acts = [a for a in obs.action_history if not a.is_me]
    for a in opp_acts[done:]:
        o["acts"] += 1
        o["raises"] += a.action in RAISE_ACTIONS
    memory["seen"] = (obs.hand_index, len(opp_acts))
    if memory["pending_raise"]:
        o["faced"] += 1
        memory["pending_raise"] = False


def hand_end(result, memory: dict) -> None:
    if memory["pending_raise"]:
        o = memory["opp"]
        o["faced"] += 1
        o["folded"] += result.winner == "me" and not result.went_to_showdown
        memory["pending_raise"] = False


def rates(memory: dict, P: dict) -> tuple[float, float]:
    """(aggression, fold_to_raise), smoothed toward priors."""
    o, w = memory["opp"], P["prior_weight"]
    agg = (o["raises"] + P["agg_prior"] * w) / (o["acts"] + w)
    fold = (o["folded"] + P["fold_prior"] * w) / (o["faced"] + w)
    return agg, fold


def mark_action(action: Action, memory: dict) -> None:
    memory["pending_raise"] = action in RAISE_ACTIONS
