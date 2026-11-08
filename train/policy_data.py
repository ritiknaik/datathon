"""Training data for the learned decision ensemble (see databricks/ENSEMBLE_PLAN.md).

At sampled decisions of our bot, the same spot is replayed with EACH legal action forced and the rest of
the hand played out k times by both players' normal policies. The label q[a] is our average chips for the
hand. The deal fixes all 15 cards up front, so every action is replayed on the same cards.

    python train/policy_data.py --states 20000 --out data/policy/pilot.npz      # local, all cores
    from train import policy_data; rows = policy_data.generate_labeled("s1", "add3", 200)  # Spark unit
"""
import argparse
import copy
import math
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selfplay import BASELINES, HAND_END, opponent_pool, strategy, warmup  # noqa: E402
from tune import variants  # noqa: E402

from bot import opponent, policy_features  # noqa: E402
from game import Action, Game, Street, button_for_deal, make_deal  # noqa: E402

RATE = 0.1         # share of our decisions that get labelled
OVERSAMPLE = 3     # extra rate for turn/river spots facing a raise (the known leak)
STRONG = {"add1", "add2", "add3", "add4"}


def pool():
    """name -> (opponent spec, sampling weight). Strong opponents count double."""
    base = strategy.load_params()
    specs = {**opponent_pool(), **variants(base)}
    return {k: (v, 2.0 if (k in STRONG or not isinstance(v, str)) else 1.0) for k, v in specs.items()}


POOL_WEIGHTS = None  # filled lazily by pool(); kept as a name for the Genie prompts


def _act(spec, obs, m):
    return strategy.decide(obs, m, spec) if isinstance(spec, dict) else BASELINES[spec](obs, m)


def _end(spec, result, m):
    if isinstance(spec, dict):
        strategy.hand_end(result, m)
    elif spec in HAND_END:
        HAND_END[spec](result, m)


def _rollout(game, ours, mu, P, opp, mo, action, rng_seed):
    """Force our `action` now, then both policies play the hand out; return our chips for the hand."""
    g, mu, mo = copy.deepcopy(game), copy.deepcopy(mu), copy.deepcopy(mo)
    mu["rng"], mo["rng"] = random.Random(rng_seed), random.Random(rng_seed + 1)
    obs = g.observation(ours)
    opponent.observe(obs, mu)
    opponent.mark_action(action, mu)
    g.apply(action)
    while not g.is_terminal:
        s = g.to_act
        o = g.observation(s)
        g.apply(strategy.decide(o, mu, P) if s == ours else _act(opp, o, mo))
    return g.result().nets[ours]


def generate_labeled(seed: str, opp_name: str, n_hands: int = 200, k: int = 4, rate: float = RATE,
                     oversample: float = OVERSAMPLE, params: dict | None = None) -> list[dict]:
    """Play n_hands of our bot vs one pool opponent and return labelled decision rows."""
    warmup()
    P = dict(params or strategy.load_params())
    P["time_budget"] = 1e9  # fixed Monte Carlo count (mc_n) at every decision: deterministic, like runtime
    opp = pool()[opp_name][0]
    rng = random.Random(f"{seed}/{opp_name}")
    ours = rng.randrange(2)
    mu, mo = {}, {}
    strategy.new_half(n_hands, mu)
    mu["rng"] = random.Random(rng.random())
    if isinstance(opp, dict):
        strategy.new_half(n_hands, mo)
    mo["rng"] = random.Random(rng.random())
    rows = []
    for d in range(n_hands):
        g = Game(make_deal(f"{seed}/{opp_name}", d), button_for_deal(d), hand_index=d)
        while not g.is_terminal:
            s = g.to_act
            obs = g.observation(s)
            if s != ours:
                g.apply(_act(opp, obs, mo))
                continue
            facing = obs.raises_this_street > 0 and obs.street in (Street.TURN, Street.RIVER)
            take = rng.random() < rate * (oversample if facing else 1)
            snap = (copy.deepcopy(g), copy.deepcopy(mu), copy.deepcopy(mo)) if take else None
            action = strategy.decide(obs, mu, P)
            if take:
                eq, ev, agg, fold = mu["last"]
                q, q_std = [math.nan] * 4, [math.nan] * 4
                for a in obs.legal_actions:
                    vals = [_rollout(*snap[:1], ours, snap[1], P, opp, snap[2], a, rng.randrange(2**31))
                            for _ in range(k)]
                    q[int(a)], q_std[int(a)] = float(np.mean(vals)), float(np.std(vals))
                rows.append(dict(seed=seed, opponent=opp_name, hand_index=d, street=obs.street.name,
                                 facing_raise=int(obs.raises_this_street > 0), pot=obs.pot,
                                 features=policy_features.build(obs, eq, ev, agg, fold),
                                 ev=[ev.get(a, math.nan) for a in Action], q=q, q_std=q_std))
            g.apply(action)
        for s, (spec, m) in ((ours, (P, mu)), (1 - ours, (opp, mo))):
            _end(spec, g.hand_result(s), m)
    return rows


def offline_value(rows, adv_pred, w: float, temperature: float) -> float:
    """Average label value (chips) of the blended policy: softmax over ev + w*(adv - mean) per row."""
    total = 0.0
    for r, adv in zip(rows, adv_pred):
        legal = [i for i in range(4) if not math.isnan(r["q"][i])]
        mean = sum(adv[i] for i in legal) / len(legal)
        score = {i: r["ev"][i] + w * (adv[i] - mean) for i in legal}
        top, scale = max(score.values()), temperature * max(r["pot"], 1)
        p = {i: math.exp((v - top) / scale) for i, v in score.items()}
        z = sum(p.values())
        total += sum(p[i] / z * r["q"][i] for i in legal)
    return total / len(rows)


def to_arrays(rows):
    return {k: np.array([r[k] for r in rows], dtype=float if k not in ("seed", "opponent", "street") else str)
            for k in rows[0]}


def selftest():
    rows = generate_labeled("selftest", "add3", n_hands=20, k=2, rate=0.5)
    assert rows and all(len(r["features"]) == policy_features.N_FEATURES for r in rows)
    assert all(not math.isnan(r["q"][1]) for r in rows)  # CHECK_CALL is always legal
    return len(rows)


def _unit(args):
    return generate_labeled(*args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=20000, help="approximate number of labelled rows")
    ap.add_argument("--hands", type=int, default=200, help="hands per work unit")
    ap.add_argument("--k", type=int, default=4, help="playouts per forced action")
    ap.add_argument("--prefix", default="p", help="seed prefix (use a new one for held-out data)")
    ap.add_argument("--out", default="data/policy/pilot.npz")
    a = ap.parse_args()
    names, weights = zip(*[(n, w) for n, (_, w) in pool().items()])
    per_unit = a.hands * 2.5 * RATE * 1.3            # rough labelled rows per unit
    n_units = max(1, int(a.states / per_unit))
    r = random.Random(a.prefix)
    jobs = [(f"{a.prefix}{i}", r.choices(names, weights)[0], a.hands, a.k) for i in range(n_units)]
    with Pool(initializer=warmup) as p:
        rows = [row for chunk in p.imap_unordered(_unit, jobs) for row in chunk]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, **to_arrays(rows))
    print(f"wrote {len(rows)} rows from {n_units} units to {a.out}")
