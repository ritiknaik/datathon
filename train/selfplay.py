"""In-process duplicate match (no HTTP), mirroring judge_engine.play_half/combine_halves.

A player is a params dict (our strategy) or a baseline name: random | call | half | pot.
    python train/selfplay.py --opp random --deals 500
"""
import argparse
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import warmup  # noqa: E402
from game import Action, Game, button_for_deal, make_deal  # noqa: E402
from bot import strategy  # noqa: E402

BASELINES = {
    "random": lambda obs, m: m["rng"].choice(obs.legal_actions),
    "call": lambda obs, m: Action.CHECK_CALL,
    "half": lambda obs, m: Action.RAISE_HALF if Action.RAISE_HALF in obs.legal_actions else Action.CHECK_CALL,
    "pot": lambda obs, m: Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL,
}


class Player:
    def __init__(self, spec, rng_seed):
        self.spec, self.rng_seed = spec, rng_seed

    def start(self, deals):
        self.m = {}
        if isinstance(self.spec, dict):
            strategy.new_half(deals, self.m)
        self.m["rng"] = random.Random(self.rng_seed)

    def act(self, obs):
        if isinstance(self.spec, dict):
            return strategy.decide(obs, self.m, self.spec)
        return BASELINES[self.spec](obs, self.m)

    def end(self, result):
        if isinstance(self.spec, dict):
            strategy.hand_end(result, self.m)


def play_half(seats, seed, deals):
    for p in seats:
        p.start(deals)
    nets = ([], [])
    for d in range(deals):
        g = Game(make_deal(seed, d), button_for_deal(d), hand_index=d)
        while not g.is_terminal:
            g.apply(seats[g.to_act].act(g.observation(g.to_act)))
        for s in (0, 1):
            nets[s].append(g.result().nets[s])
            seats[s].end(g.hand_result(s))
    return nets


def match(a, b, deals=500, seed="selfplay", rng_seed=0):
    """Return (mean bb/deal for a, ci_low, ci_high)."""
    A, B = Player(a, rng_seed), Player(b, rng_seed + 1)
    h1 = play_half((A, B), seed, deals)[0]
    h2 = play_half((B, A), seed, deals)[1]
    bb = [(x + y) / 2 for x, y in zip(h1, h2)]
    mean = sum(bb) / deals
    se = statistics.stdev(bb) / math.sqrt(deals)
    return mean, mean - 1.96 * se, mean + 1.96 * se


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="random", help="baseline name or path to a params json")
    ap.add_argument("--params", default=os.path.join(strategy.HERE, "params.json"))
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--seed", default="selfplay")
    args = ap.parse_args()
    warmup()
    opp = strategy.load_params(args.opp) if args.opp.endswith(".json") else args.opp
    print("bb/deal %.3f  CI [%.3f, %.3f]" % match(strategy.load_params(args.params), opp, args.deals, args.seed))
