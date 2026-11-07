"""Accept test for a tuned candidate: run it and the incumbent over the whole pool on FRESH seeds.

    python train/compare.py bot/params.json runs/best.json --deals 1000 --seeds hold1 hold2

The first file is the incumbent. For every other file it prints bb/deal vs each pool opponent
(mean over the seeds), the pool mean and worst result, and a head-to-head match against the
incumbent with its 95% interval. ACCEPT needs: head-to-head interval entirely above 0, pool mean
not lower, and worst result not lower. Use seeds that tune.py never saw.
"""
import argparse
import json
from multiprocessing import Pool

from selfplay import match, opponent_pool, strategy, warmup


def load(path):
    d = json.load(open(path))
    if "params" in d:  # a tune.py run file
        return {**d["params"], "scenarios": strategy.load_params()["scenarios"]}
    return strategy.load_params(path)


def _job(args):
    P, spec, deals, seed = args
    return match(P, spec, deals, seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="params files; the first is the incumbent")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seeds", nargs="+", default=["hold1", "hold2"])
    a = ap.parse_args()
    cands = [load(f) for f in a.files]
    opps = opponent_pool()
    with Pool(initializer=warmup) as pool:
        res = []
        for P in cands:
            jobs = [(P, spec, a.deals, s) for spec in opps.values() for s in a.seeds]
            r = pool.map(_job, jobs)
            n = len(a.seeds)
            res.append({k: sum(x[0] for x in r[i * n:(i + 1) * n]) / n for i, k in enumerate(opps)})
        h2h = [None] + [pool.map(_job, [(P, cands[0], a.deals, s) for s in a.seeds]) for P in cands[1:]]
    print(f"{'file':<22}" + "".join(f"{k[:9]:>10}" for k in opps) + f"{'mean':>8}{'worst':>8}")
    for f, r in zip(a.files, res):
        v = list(r.values())
        print(f"{f[-21:]:<22}" + "".join(f"{x:>10.2f}" for x in v) + f"{sum(v) / len(v):>8.2f}{min(v):>8.2f}")
    base = res[0]
    for f, r, h in zip(a.files[1:], res[1:], h2h[1:]):
        mean = sum(x[0] for x in h) / len(h)
        se = (sum(((x[2] - x[1]) / 3.92) ** 2 for x in h) ** 0.5) / len(h)  # pooled over the seeds
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        ok = lo > 0 and sum(r.values()) >= sum(base.values()) and min(r.values()) >= min(base.values())
        print(f"{f}: head-to-head vs incumbent {mean:+.2f} bb/deal, 95% CI [{lo:+.2f}, {hi:+.2f}] -> "
              f"{'ACCEPT' if ok else 'REJECT'}")
