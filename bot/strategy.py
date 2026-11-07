"""Decision logic: equity -> EV per legal action -> softmax mixed strategy.

All numbers live in params.json; real-life rules live in scenarios.json.
"""
import json
import math
import os
import random
import sys
import time

from evaluator import cards_to_str
from game import RAISE_ACTIONS, Action, Street

from . import opponent
from .equity import mc_equity, preflop_equity, warmup_np

try:  # optional: numpy-only learned equity model exported from Databricks (bot/equity_model.py, predict(hole, board))
    from . import equity_model
except Exception:  # missing file, missing/corrupt weights, anything: never fail at startup
    equity_model = None

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.environ.get("BOT_TRACE") == "1"  # watching only: one line per decision on stderr, off by default


def load_params(path: str = os.path.join(HERE, "params.json")) -> dict:
    warmup_np()  # build the numpy score table at startup, not inside the first request
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
    eq = None
    # opt-in: set in params.json after train/compare.py ACCEPTs it; "equity_model_streets" limits it to some streets
    if equity_model is not None and P.get("use_equity_model") \
            and obs.street.name in P.get("equity_model_streets", ("PREFLOP", "FLOP", "TURN", "RIVER")):
        try:
            eq = float(equity_model.predict(obs.hole_cards, obs.board))
            eq = eq if 0.0 <= eq <= 1.0 else None
        except Exception:
            eq = None  # a broken model must never cost an error: use the Monte Carlo path
    if eq is None:
        eq = preflop_equity(obs.hole_cards, P["preflop"])
        left = max(1, memory["hands"] - obs.hand_index)
        per_decision = (P["time_budget"] - memory["t"]) / (left * P["decisions_per_hand"])
        n_max = P.get("preflop_mc_n", 0) if obs.street == Street.PREFLOP else P["mc_n"]
        n = min(n_max, int(per_decision / P["sample_cost"]))
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
    # a bet from a passive opponent means more than one from a maniac; the small blind completing the
    # forced blind is not a bet, so "discount_needs_raise" skips the discount unless a raise was made
    if obs.to_call > 0 and (obs.raises_this_street > 0 or not P.get("discount_needs_raise")):
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
    from_scenario = probs is not None
    if probs is None:
        scale = P["temperature"] * max(pot, 1)
        top = max(ev.values())
        probs = {a: math.exp((v - top) / scale) for a, v in ev.items()}
    actions = list(probs)
    action = memory["rng"].choices(actions, weights=[probs[a] for a in actions])[0]

    if TRACE:
        total = sum(probs.values())
        print(f"[bot] hand {obs.hand_index} {obs.street.name} cards {cards_to_str(obs.hole_cards)} "
              f"board {cards_to_str(obs.board) or '-'} | equity {eq:.2f} | to_call {tc} pot {pot} "
              f"| opp aggr {agg:.2f} fold {fold:.2f} | EV " +
              " ".join(f"{a.name}={v:.1f}" for a, v in ev.items()) +
              " | mix " + " ".join(f"{a.name}={probs[a] / total:.2f}" for a in actions) +
              f" | {'scenario' if from_scenario else 'mixed'} -> {action.name}",
              file=sys.stderr, flush=True)

    opponent.mark_action(action, memory)
    memory["t"] += time.perf_counter() - t0
    return action
