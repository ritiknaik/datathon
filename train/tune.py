"""Random-search tuner over params.json (one Slurm array task = one --seed).

    python train/tune.py --iters 30 --deals 300 --seed 1 --out runs/1.json
    python train/tune.py --pick runs          # copy the best run into bot/params.json
"""
import argparse
import copy
import glob
import json
import os
import random

from selfplay import load_local, match, strategy, warmup

TUNABLE = ["temperature", "bet_discount", "called_penalty", "agg_prior", "fold_prior"]
POOL = ["random", "call", "half", "pot"]


def perturb(P, rng, sigma):
    Q = copy.deepcopy(P)
    for k in TUNABLE:
        Q[k] = round(Q[k] * rng.lognormvariate(0, sigma), 4)
    for k in Q["fe_scale"]:
        Q["fe_scale"][k] = round(min(1.5, Q["fe_scale"][k] * rng.lognormvariate(0, sigma)), 4)
    return Q


def score(P, base, deals, seed, local=None):
    """Mean bb/deal vs the baseline pool plus the incumbent params.
    `local` (training-only) applies during play only; the saved params never contain it."""
    return sum(match(P, o, deals, seed, local=local)[0] for o in POOL + [base]) / (len(POOL) + 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/0.json")
    ap.add_argument("--pick", help="dir of run outputs: install the best into bot/params.json")
    ap.add_argument("--local", help="training-only overrides json (train/local/); never saved")
    a = ap.parse_args()
    L = load_local(a.local)
    path = os.path.join(strategy.HERE, "params.json")
    if a.pick:
        best = max((json.load(open(f)) for f in glob.glob(os.path.join(a.pick, "*.json"))), key=lambda r: r["score"])
        json.dump(best["params"], open(path, "w"), indent=2)
        print("installed score", best["score"])
        raise SystemExit
    warmup()
    rng = random.Random(a.seed)
    base = strategy.load_params()
    best, best_s = base, score(base, base, a.deals, f"tune{a.seed}", L)
    for i in range(a.iters):
        cand = perturb(best, rng, a.sigma)
        s = score(cand, base, a.deals, f"tune{a.seed}", L)  # same deals for every candidate = fair comparison
        if s > best_s:
            best, best_s = cand, s
        print(i, round(s, 3), "best", round(best_s, 3), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    best.pop("scenarios", None)
    assert not any(best.get(k) == v and base.get(k) != v for k, v in L.items()), "local override leaked"
    json.dump({"score": best_s, "params": best}, open(a.out, "w"), indent=2)
