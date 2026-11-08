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

from . import features, opponent, qnet
from .equity import mc_equity, preflop_equity, warmup_np

try:  # optional: numpy-only learned equity model exported from Databricks (bot/equity_model.py, predict(hole, board))
    from . import equity_model
except ImportError:
    equity_model = None

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.environ.get("BOT_TRACE") == "1"  # watching only: one line per decision on stderr, off by default


def load_params(path: str = os.path.join(HERE, "params.json")) -> dict:
    warmup_np()  # build the numpy score table at startup, not inside the first request
    with open(path, encoding="utf-8") as f:
        P = json.load(f)
    if "scenarios" not in P:  # a params file may carry its own rules (e.g. train/opps/*.json)
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
    if equity_model is not None:
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
        # Never fall off a cliff on a slow/shared CPU: keep a cheap floor of mc_min samples
        # until this half's time_budget is actually spent (the 8 s bank stays protected).
        n = min(n_max, max(P["mc_min"], int(per_decision / P["sample_cost"]))) if memory["t"] < P["time_budget"] else 0
        if n:
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
    use_net = qnet.LOADED and P.get("qnet_weight", 0) > 0
    if use_net or memory.get("log_features"):  # training code sets log_features; live play only if the net is on
        memory["feat"] = features.build(obs, eq, agg, fold, memory["opp"]["acts"])
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
    if use_net:  # blend the learned chips-from-now with the formula; any failure keeps the formula
        try:
            q, w = qnet.predict(memory["feat"]), P["qnet_weight"]
            for i, a in enumerate(features.ACTIONS):
                if a in ev and a != Action.FOLD:  # folding is exactly 0 from now
                    ev[a] = (1 - w) * ev[a] + w * q[i] * obs.big_blind
        except Exception:
            pass

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
