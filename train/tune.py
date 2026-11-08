"""Random-search tuner over params.json (one Slurm array task = one --seed).

    python train/tune.py --iters 30 --deals 300 --seed 1 --out runs/1.json
    python train/tune.py --pick runs          # copy the best run into bot/params.json

A candidate is scored against the whole opponent pool: the 8 baselines, every past version in
bot/pool/*.json, and style variants of the incumbent (loose / sticky / tight). The score is the
mean of the 4 WORST results, so a change cannot win by beating weak bots. Always accept a winner
with train/compare.py on fresh seeds (held out), never on this score alone.
"""
import argparse
import copy
import glob
import json
import os
import random
from multiprocessing import Pool

from selfplay import load_local, match, opponent_pool, strategy, warmup

TUNABLE = ["temperature", "bet_discount", "called_penalty", "agg_prior", "fold_prior"]
WORST_N = 4


def perturb(P, rng, sigma):
    Q = copy.deepcopy(P)
    for k in TUNABLE:
        Q[k] = round(Q[k] * rng.lognormvariate(0, sigma), 4)
    for k in Q["fe_scale"]:
        Q["fe_scale"][k] = round(min(1.5, Q["fe_scale"][k] * rng.lognormvariate(0, sigma)), 4)
    return Q


def variants(P):
    """Style variants of a params dict, used as extra opponents."""
    def mod(**kw):
        Q = copy.deepcopy(P)
        Q.update(kw)
        return Q
    return {
        "var/loose": mod(temperature=P["temperature"] * 4, called_penalty=0.0),
        "var/sticky": mod(bet_discount=0.0, fe_scale={k: 0.3 for k in P["fe_scale"]}),
        "var/tight": mod(bet_discount=P["bet_discount"] * 3, called_penalty=P["called_penalty"] * 2),
    }


def opponents(base):
    return {**opponent_pool(), **variants(base)}


def _job(args):
    P, spec, deals, seed, local = args
    return match(P, spec, deals, seed, local=local)[0]


def score(P, opps, deals, seed, pool=None, local=None):
    """Mean bb/deal over the WORST_N opponents (a robust score), plus all per-opponent results.
    `local` (training-only overrides) applies during play only; saved params never contain it."""
    jobs = [(P, spec, deals, seed, local) for spec in opps.values()]
    res = pool.map(_job, jobs) if pool else [_job(j) for j in jobs]
    return sum(sorted(res)[:WORST_N]) / WORST_N, dict(zip(opps, res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=1, help="processes for scoring one candidate (1 on Slurm)")
    ap.add_argument("--out", default="runs/0.json")
    ap.add_argument("--pick", help="dir of run outputs: install the best into bot/params.json")
    ap.add_argument("--local", help="training-only overrides json (train/local/); never saved")
    a = ap.parse_args()
    L = load_local(a.local)
    path = os.path.join(strategy.HERE, "params.json")
    if a.pick:
        best = max((json.load(open(f)) for f in glob.glob(os.path.join(a.pick, "*.json"))), key=lambda r: r["score"])
        json.dump(best["params"], open(path, "w"), indent=2)
        print("installed score", best["score"], "- now accept it with train/compare.py on fresh seeds")
        raise SystemExit
    warmup()
    rng = random.Random(a.seed)
    base = strategy.load_params()
    opps = opponents(base)
    pool = Pool(a.procs, initializer=warmup) if a.procs > 1 else None
    seed = f"tune{a.seed}"
    best, (best_s, per) = base, score(base, opps, a.deals, seed, pool, L)
    print("incumbent", round(best_s, 3), {k: round(v, 1) for k, v in per.items()}, flush=True)
    for i in range(a.iters):
        cand = perturb(best, rng, a.sigma)
        s, per = score(cand, opps, a.deals, seed, pool, L)  # same deals for every candidate = fair comparison
        if s > best_s:
            best, best_s = cand, s
        print(i, round(s, 3), "best", round(best_s, 3), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    best.pop("scenarios", None)
    assert not any(best.get(k) == v and base.get(k) != v for k, v in L.items()), "local override leaked"
    json.dump({"score": best_s, "params": best}, open(a.out, "w"), indent=2)
