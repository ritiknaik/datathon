"""Generate Q-net training data (training only): our bot + exploration vs the pool and random-style opponents.

One compressed .npz per run (Grace scratch has a file-count limit). Each row = one decision:
X features, A action index, M legal mask, Y bb from this decision on, E explored flag, O opponent id.

    python train/gen_q.py --matches 40 --deals 1000 --explore 0.1 --seed 0 --out data/q/part0.npz
"""
import argparse
import copy
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selfplay import match, opponent_pool, strategy, warmup  # noqa: E402


def random_style(base, rng):
    """Our strategy with randomly drawn settings: a different opponent personality every match."""
    P = copy.deepcopy(base)
    for k, lo, hi in (("temperature", 0.005, 0.3), ("bet_discount", 0.0, 0.5), ("called_penalty", 0.0, 0.3),
                      ("agg_prior", 0.05, 0.7), ("fold_prior", 0.05, 0.7), ("prior_weight", 2, 40)):
        P[k] = rng.uniform(lo, hi)
    P["fe_scale"] = {k: rng.uniform(0.1, 1.6) for k in P["fe_scale"]}
    return P


def run(job):
    opp_name, opp, params, deals, seed, explore = job
    warmup()
    rows = []
    match(params, opp, deals, seed, qlog=rows, explore=explore)
    f, ai, mask, y, e = zip(*rows)  # compact arrays, not millions of Python tuples
    return opp_name, (np.array(f, np.float32), np.array(ai, np.int8), np.array(mask, bool),
                      np.array(y, np.float32), np.array(e, np.int8))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=40, help="matches in this run (pool + random styles, round-robin)")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--explore", type=float, default=0.1, help="chance per decision of a random legal action")
    ap.add_argument("--random-share", type=float, default=0.5, help="share of matches vs random-style opponents")
    ap.add_argument("--params", default=os.path.join(strategy.HERE, "params.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    ap.add_argument("--out", default="data/q/part0.npz", help="written as part0_000.npz, part0_001.npz, ...")
    ap.add_argument("--chunk", type=int, default=60, help="save every N finished matches (a crash loses at most one chunk)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    base = strategy.load_params(a.params)
    pool = list(opponent_pool().items())
    jobs = []
    for i in range(a.matches):
        name, opp = ("random_style", random_style(base, rng)) if rng.random() < a.random_share else pool[i % len(pool)]
        jobs.append((name, opp, base, a.deals, f"q{a.seed}-{i}", a.explore))
    names, parts, k = [], [], 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def save():
        X, A, M, Y, E, O = (np.concatenate(c) for c in zip(*parts))
        path = a.out.replace(".npz", f"_{k:03d}.npz")
        np.savez_compressed(path, X=X, A=A, M=M, Y=Y, E=E, O=O, names=np.array(names))
        print(f"saved {len(Y)} rows, {X.shape[1]} features -> {path}", flush=True)

    with ProcessPoolExecutor(a.workers) as ex:
        for i, (name, arrs) in enumerate(ex.map(run, jobs), 1):
            if name not in names:
                names.append(name)
            parts.append(arrs + (np.full(len(arrs[3]), names.index(name), np.int16),))
            print(f"[{i}/{len(jobs)}] {name}: {len(arrs[3])} rows", flush=True)
            if len(parts) == a.chunk or i == len(jobs):
                save()
                parts, k = [], k + 1
