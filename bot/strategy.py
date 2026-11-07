"""Decision logic: equity -> EV per legal action -> softmax mixed strategy.

All numbers live in params.json; real-life rules live in scenarios.json.
"""
import json
import math
import os
import random
import time

from game import RAISE_ACTIONS, Action, Street

from . import opponent
from .equity import mc_equity, preflop_equity

HERE = os.path.dirname(os.path.abspath(__file__))


def load_params(path: str = os.path.join(HERE, "params.json")) -> dict:
    with open(path, encoding="utf-8") as f:
        P = json.load(f)
    with open(os.path.join(HERE, "scenarios.json"), encoding="utf-8") as f:
        P["scenarios"] = json.load(f)
    return P


def new_half(num_hands: int, memory: dict) -> None:
    opponent.init(memory)
    memory.update(rng=random.Random(random.SystemRandom().random()), hands=num_hands, t=0.0, eq=None)


def hand_end(result, memory: dict) -> None:
    opponent.hand_end(result, memory)


def _equity(obs, memory: dict, P: dict) -> float:
    key = (obs.hand_index, obs.street)
    if memory["eq"] and memory["eq"][0] == key:
        return memory["eq"][1]
    eq = preflop_equity(obs.hole_cards, P["preflop"])
    if obs.street != Street.PREFLOP:
        left = max(1, memory["hands"] - obs.hand_index)
        per_decision = (P["time_budget"] - memory["t"]) / (left * P["decisions_per_hand"])
        n = min(P["mc_n"], int(per_decision / P["sample_cost"]))
        if n >= P["mc_min"]:
            eq = mc_equity(obs.hole_cards, obs.board, n, memory["rng"])
    memory["eq"] = (key, eq)
    return eq


def _scenario(obs, eq: float, P: dict) -> dict | None:
    facts = {"street": obs.street.name, "facing": obs.to_call > 0, "is_button": obs.is_button}
    for rule in P["scenarios"]:
        w = rule["when"]
        if all(facts[k] == v for k, v in w.items() if k in facts) \
                and w.get("eq_min", 0) <= eq <= w.get("eq_max", 1):
            probs = {Action[a]: p for a, p in rule["probs"].items() if Action[a] in obs.legal_actions}
            if sum(probs.values()) > 0:
                return probs
    return None


def decide(obs, memory: dict, P: dict) -> Action:
    t0 = time.perf_counter()
    opponent.observe(obs, memory)
    eq = _equity(obs, memory, P)
    agg, fold = opponent.rates(memory, P)
    if obs.to_call > 0:  # a bet from a passive opponent means more than one from a maniac
        eq = max(0.0, eq - P["bet_discount"] * (1 - agg))

    pot, tc = obs.pot, obs.to_call
    ev = {Action.CHECK_CALL: eq * (pot + tc) - tc}
    if Action.FOLD in obs.legal_actions:
        ev[Action.FOLD] = 0.0
    fe = fold * P["fe_scale"][obs.street.name]
    eq_called = max(0.0, eq - P["called_penalty"])
    for a in RAISE_ACTIONS:
        if a in obs.legal_actions:
            opp_add = min(obs.raise_to[a] - obs.opp_bet, obs.opp_stack)
            add = min(obs.raise_to[a] - obs.my_bet, tc + opp_add)
            ev[a] = fe * pot + (1 - fe) * (eq_called * (pot + add + opp_add) - add)

    probs = _scenario(obs, eq, P)
    if probs is None:
        scale = P["temperature"] * max(pot, 1)
        top = max(ev.values())
        probs = {a: math.exp((v - top) / scale) for a, v in ev.items()}
    actions = list(probs)
    action = memory["rng"].choices(actions, weights=[probs[a] for a in actions])[0]

    opponent.mark_action(action, memory)
    memory["t"] += time.perf_counter() - t0
    return action
