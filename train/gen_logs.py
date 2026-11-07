"""Generate decision-level training data: our bot vs every test opponent, one CSV per opponent.
Rows = one decision each, with the hand's outcome (net_bb, showdown, won). Training only.

    python train/gen_logs.py --deals 3000 --out data/decisions
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from selfplay import match, strategy, warmup  # noqa: E402

OPPS = ["random", "call", "half", "pot"] + [
    os.path.join(HERE, "opps", f) for f in ("past_best.json", "adaptor.json", "hyper_pass.json", "hyper_agg.json")]


def run(job):
    opp, params, deals, seed, out = job
    warmup()
    label = os.path.basename(opp).replace(".json", "")
    rows = []
    res = match(strategy.load_params(params), strategy.load_params(opp) if opp.endswith(".json") else opp,
                deals, f"{seed}-{label}", log=rows, label=label)
    path = os.path.join(out, f"{label}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return f"{label}: {len(rows)} rows, bb/deal {res[0]:.2f} CI [{res[1]:.2f}, {res[2]:.2f}] -> {path}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--params", default=os.path.join(strategy.HERE, "params.json"))
    ap.add_argument("--seed", default="logs")
    ap.add_argument("--out", default="data/decisions")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for line in ex.map(run, [(o, a.params, a.deals, a.seed, a.out) for o in OPPS]):
            print(line, flush=True)
