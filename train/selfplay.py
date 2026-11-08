"""In-process duplicate match (no HTTP), mirroring judge_engine.play_half/combine_halves.

A player is a params dict (our strategy) or a baseline name:
random | call | half | pot | add1 | add2 | add3 | add4 (add1-4 = baseline-bots/baseline-add-1..4).
    python train/selfplay.py --opp random --deals 500
"""
import argparse
import glob
import json
import math
import os
import random
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baseline-bots"))
from evaluator import warmup  # noqa: E402
from game import Action, Game, button_for_deal, make_deal  # noqa: E402
from bot import features, opponent, strategy  # noqa: E402
import opponents  # noqa: E402  (baseline-add-1..4: add1 | add2 | add3 | add4)

BASELINES = {
    "random": lambda obs, m: m["rng"].choice(obs.legal_actions),
    "call": lambda obs, m: Action.CHECK_CALL,
    "half": lambda obs, m: Action.RAISE_HALF if Action.RAISE_HALF in obs.legal_actions else Action.CHECK_CALL,
    "pot": lambda obs, m: Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL,
    **{k: act for k, (act, _) in opponents.STYLES.items()},
}
HAND_END = {k: end for k, (_, end) in opponents.STYLES.items() if end}


class Player:
    def __init__(self, spec, rng_seed, log=None, label="", qlog=None, explore=0.0):
        self.spec, self.rng_seed = spec, rng_seed
        self.log, self.label, self.rows = log, label, []  # log: list collecting one row per decision (training only)
        self.qlog, self.qrows, self.explore = qlog, [], explore  # Q-net training data + exploration (training only)
        self.xrng = random.Random(rng_seed + 7919)

    def start(self, deals):
        self.m = {}
        if isinstance(self.spec, dict):
            strategy.new_half(deals, self.m)
        self.m["rng"] = random.Random(self.rng_seed)
        self.m["log_features"] = self.qlog is not None

    def act(self, obs):
        if not isinstance(self.spec, dict):
            return BASELINES[self.spec](obs, self.m)
        a = strategy.decide(obs, self.m, self.spec)
        explored = 0
        if self.explore and self.xrng.random() < self.explore:  # training only: try a random legal action
            a, explored = self.xrng.choice(obs.legal_actions), 1
            opponent.mark_action(a, self.m)
        if self.qlog is not None:  # target (bb from now) is filled in at hand end
            self.qrows.append((self.m["feat"], features.ACTIONS.index(a), features.legal_mask(obs),
                               80 - obs.my_stack, explored))
        if self.log is not None:
            agg, fold = opponent.rates(self.m, self.spec)
            opp_hist = [h for h in obs.action_history if not h.is_me]
            self.rows.append({
                "opp": self.label, "hand": obs.hand_index, "street": obs.street.name,
                "is_button": int(obs.is_button), "eq": round(self.m["eq"][1], 4), "pot": obs.pot,
                "to_call": obs.to_call, "pot_odds": round(obs.to_call / (obs.pot + obs.to_call), 4),
                "raises_this_street": obs.raises_this_street,
                "opp_last": opp_hist[-1].action.name if opp_hist else "NONE",
                "opp_raises_hand": sum(h.action.name.startswith("RAISE") for h in opp_hist),
                "opp_agg": round(agg, 4), "opp_fold": round(fold, 4), "action": a.name})
        return a

    def end(self, result):
        if isinstance(self.spec, dict):
            strategy.hand_end(result, self.m)
            for r in self.rows:  # every decision gets the hand's outcome
                r.update(net_bb=result.my_net / 2, showdown=int(result.went_to_showdown),
                         won=int(result.winner == "me"))
            if self.log is not None:
                self.log.extend(self.rows)
            self.rows = []
            for f, ai, mask, put_in, ex in self.qrows:  # chips from this decision on = net + chips already in
                self.qlog.append((f, ai, mask, (result.my_net + put_in) / 2, ex))
            self.qrows = []
        elif self.spec in HAND_END:
            HAND_END[self.spec](result, self.m)


HALF = 1000  # tournament half length: bots restart every HALF deals, so long runs = several real halves


def play_half(seats, seed, deals):
    nets = ([], [])
    for d in range(deals):
        if d % HALF == 0:
            for p in seats:
                p.start(min(HALF, deals - d))
        g = Game(make_deal(seed, d), button_for_deal(d), hand_index=d % HALF)
        while not g.is_terminal:
            g.apply(seats[g.to_act].act(g.observation(g.to_act)))
        for s in (0, 1):
            nets[s].append(g.result().nets[s])
            seats[s].end(g.hand_result(s))
    return nets


def load_local(path):
    """Training-only overrides (train/local/, git-ignored). Never read by agent.py."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def match(a, b, deals=500, seed="selfplay", rng_seed=0, local=None, log=None, label="", qlog=None, explore=0.0):
    """Return (mean bb/deal for a, ci_low, ci_high). `local` overrides apply to every strategy player.
    `log` (a list) collects one row per decision of player a (training data)."""
    if local:
        a, b = ({**p, **local} if isinstance(p, dict) else p for p in (a, b))
    A, B = Player(a, rng_seed, log, label, qlog, explore), Player(b, rng_seed + 1)
    h1 = play_half((A, B), seed, deals)[0]
    h2 = play_half((B, A), seed, deals)[1]
    bb = [(x + y) / 2 for x, y in zip(h1, h2)]
    mean = sum(bb) / deals
    se = statistics.stdev(bb) / math.sqrt(deals)
    return mean, mean - 1.96 * se, mean + 1.96 * se


def opponent_pool():
    """name -> opponent spec: every baseline plus each saved past version in bot/pool/*.json."""
    pool = {k: k for k in BASELINES}
    for f in sorted(glob.glob(os.path.join(strategy.HERE, "pool", "*.json"))):
        pool["pool/" + os.path.basename(f)[:-5]] = strategy.load_params(f)
    for f in sorted(glob.glob(os.path.join(ROOT, "train", "opps", "*.json"))):  # extra test opponents
        pool["opps/" + os.path.basename(f)[:-5]] = strategy.load_params(f)
    return pool


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="random", help="baseline name or path to a params json")
    ap.add_argument("--params", default=os.path.join(strategy.HERE, "params.json"))
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--seed", default="selfplay")
    ap.add_argument("--local", help="training-only overrides json (keep it in train/local/)")
    args = ap.parse_args()
    warmup()
    opp = strategy.load_params(args.opp) if args.opp.endswith(".json") else args.opp
    print("bb/deal %.3f  CI [%.3f, %.3f]" % match(strategy.load_params(args.params), opp, args.deals, args.seed,
                                                    local=load_local(args.local)))
