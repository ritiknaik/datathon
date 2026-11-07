"""Play judge matches with hand logs on and flatten them into two CSVs (for Databricks etc).

    python tools/gen_hands.py                           # 8 baselines (random always_pot calling_station always_half add1..add4) x seeds 7 11 42
    python tools/gen_hands.py --deals 200 --seeds 7 --opps random
Output (git-ignored): data/hands.csv (one row per hand), data/actions.csv (one row per action).
Local only: it starts agent.py on :5008 and a baseline on :5009 for each match.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPPS = ["random", "always_pot", "calling_station", "always_half", "add1", "add2", "add3", "add4"]
HAND_COLS = ["match_id", "opponent", "seed", "half", "deal_index", "button_seat", "our_seat", "our_is_button",
             "our_hole", "opp_hole", "board", "pot", "went_to_showdown", "ended_by_fold", "final_street",
             "num_actions", "our_net", "opp_net"]
ACT_COLS = ["match_id", "half", "deal_index", "action_index", "street", "actor", "action", "amount", "street_total"]


def flatten(opp, seed, data, hands, acts):
    mid = f"{opp}_{seed}"
    for half, key in ((1, "half1"), (2, "half2")):
        our = 0 if half == 1 else 1  # agent1 (our bot) sits in seat 0 in half 1 and seat 1 in half 2
        for h in data["logs"][key]:
            last = h["actions"][-1] if h["actions"] else None
            folder = ""
            if last and last["action"] == "FOLD":
                folder = "us" if last["seat"] == our else "opp"
            hands.append([mid, opp, seed, half, h["deal_index"], h["button_seat"], our, int(h["button_seat"] == our),
                          " ".join(h["hole_cards"][our]), " ".join(h["hole_cards"][1 - our]),
                          " ".join(h["board_revealed"]), h["pot"], int(h["went_to_showdown"]), folder,
                          last["street"] if last else "", len(h["actions"]), h["nets"][our], h["nets"][1 - our]])
            for i, a in enumerate(h["actions"]):
                acts.append([mid, half, h["deal_index"], i, a["street"], "us" if a["seat"] == our else "opp",
                             a["action"], a["amount"], a["street_total"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seeds", nargs="+", default=["7", "11", "42"])
    ap.add_argument("--opps", nargs="+", default=OPPS)
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()
    hands, acts = [], []
    for opp in a.opps:
        for seed in a.seeds:
            env = {**os.environ, "BOT_TRACE": ""}
            bot = subprocess.Popen([sys.executable, "agent.py"], cwd=ROOT, env={**env, "PORT": "5008"},
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            script = f"baseline-add-{opp[3:]}.py" if opp.startswith("add") else f"baseline_{opp}.py"
            ob = subprocess.Popen([sys.executable, f"baseline-bots/{script}"], cwd=ROOT,
                                  env={**env, "PORT": "5009"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "m.json")
                    r = subprocess.run([sys.executable, "judge_engine.py", "--agent1", "http://127.0.0.1:5008",
                                        "--agent2", "http://127.0.0.1:5009", "--seed", seed, "--deals",
                                        str(a.deals), "--json", path, "--logs"],
                                       cwd=ROOT, capture_output=True, text=True)
                    result = [ln for ln in r.stdout.splitlines() if ln.startswith(("RESULT", "agent1:"))]
                    print(opp, seed, " | ".join(result), flush=True)
                    flatten(opp, seed, json.load(open(path)), hands, acts)
            finally:
                bot.kill()
                ob.kill()
                bot.wait()
                ob.wait()
    os.makedirs(a.out, exist_ok=True)
    for name, cols, rows in (("hands.csv", HAND_COLS, hands), ("actions.csv", ACT_COLS, acts)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        print(f"wrote {name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
