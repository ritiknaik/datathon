"""Offline decision audit of our bot, from the CSVs written by tools/gen_hands.py.

    python tools/audit.py                  # all hands (a few minutes, uses all cores)
    python tools/audit.py --sample 0.1     # quick look at 10% of hands

Replays every hand through game.Game, and at each of OUR decisions that faced a bet it computes
equity vs a random hand (mc_equity) and the pot odds. The four baselines ignore their cards, so
a random opponent hand is the right range against them. Immediate pot odds only: later streets,
implied odds and fold equity are NOT counted, so treat the numbers as a ranking, not exact EV.
Writes data/audit_decisions.csv (one row per decision facing a bet).
"""
import argparse
import collections
import csv
import os
import random
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bot.equity import mc_equity  # noqa: E402
from evaluator import warmup  # noqa: E402
from game import Action, Game, button_for_deal, make_deal  # noqa: E402

MARGIN = 0.05  # "clear" mistake = equity beyond the pot-odds requirement by more than this


def _eq(job):
    hole, board, n = job
    return mc_equity(hole, board, n, random.Random(f"{hole}{board}"))


def load(data):
    hands = {(r["match_id"], r["half"], r["deal_index"]): r for r in csv.DictReader(open(f"{data}/hands.csv"))}
    acts = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{data}/actions.csv")):
        acts[(r["match_id"], r["half"], r["deal_index"])].append(r)
    return hands, acts


def replay(hands, acts, sample):
    rng, out, bad = random.Random(0), [], 0
    for key, h in hands.items():
        if sample < 1 and rng.random() > sample:
            continue
        our, d = int(h["our_seat"]), int(h["deal_index"])
        g = Game(make_deal(h["seed"], d), int(h["button_seat"]), hand_index=d)
        for a in sorted(acts[key], key=lambda r: int(r["action_index"])):
            seat = g.to_act
            if (seat == our) != (a["actor"] == "us"):
                bad += 1
                break
            if seat == our:
                o = g.observation(seat)
                if o.to_call > 0:
                    out.append(dict(opponent=h["opponent"], match_id=key[0], half=key[1], deal_index=d,
                                    street=a["street"], action=a["action"], pot=o.pot, to_call=o.to_call,
                                    hole=tuple(o.hole_cards), board=tuple(o.board),
                                    showdown=int(h["went_to_showdown"]), net=int(h["our_net"])))
            g.apply(Action[a["action"]])
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--samples", type=int, default=200, help="Monte Carlo samples per equity")
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of hands to audit")
    a = ap.parse_args()
    hands, acts = load(a.data)
    dec, bad = replay(hands, acts, a.sample)
    keys = sorted({(d["hole"], d["board"]) for d in dec})
    print(f"{len(dec)} decisions facing a bet, {len(keys)} distinct equities to compute"
          f" ({bad} hands skipped on replay mismatch)", flush=True)
    with Pool(initializer=warmup) as p:
        eqs = dict(zip(keys, p.map(_eq, [(h, b, a.samples) for h, b in keys], chunksize=50)))
    n_hands = collections.Counter(h["opponent"] for h in hands.values())
    scale = a.sample if a.sample < 1 else 1.0
    rows = []
    for d in dec:
        eq = eqs[(d["hole"], d["board"])]
        req = d["to_call"] / (d["pot"] + d["to_call"])
        d.update(equity=eq, required=req, ev_call=eq * (d["pot"] + d["to_call"]) - d["to_call"])
        rows.append(d)
    with open(os.path.join(a.data, "audit_decisions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["match_id", "opponent", "half", "deal_index", "street", "action", "pot", "to_call",
                    "equity", "required", "ev_call_chips"])
        for d in rows:
            w.writerow([d["match_id"], d["opponent"], d["half"], d["deal_index"], d["street"], d["action"],
                        d["pot"], d["to_call"], round(d["equity"], 3), round(d["required"], 3),
                        round(d["ev_call"], 2)])

    def per100(chips, opp):  # bb per 100 hands
        return chips / 2 / max(1, n_hands[opp] * scale) * 100

    print("\n== Pot-odds audit, per opponent (bb per 100 hands, immediate pot odds only) ==")
    print(f"{'opponent':<16}{'folds':>6}{'fold leaks':>11}{'clear':>6}{'leak bb/100':>12}"
          f"{'calls':>7}{'-EV calls':>10}{'clear':>6}{'cost bb/100':>12}")
    for opp in sorted(n_hands):
        r = [d for d in rows if d["opponent"] == opp]
        fo = [d for d in r if d["action"] == "FOLD"]
        ca = [d for d in r if d["action"] == "CHECK_CALL"]
        fl = [d for d in fo if d["ev_call"] > 0]
        bc = [d for d in ca if d["ev_call"] < 0]
        print(f"{opp:<16}{len(fo):>6}{len(fl):>11}{sum(d['equity'] - d['required'] > MARGIN for d in fo):>6}"
              f"{per100(sum(d['ev_call'] for d in fl), opp):>12.2f}{len(ca):>7}{len(bc):>10}"
              f"{sum(d['required'] - d['equity'] > MARGIN for d in ca):>6}"
              f"{per100(-sum(d['ev_call'] for d in bc), opp):>12.2f}")

    print("\n== Folds with positive pot-odds EV, by street (all opponents) ==")
    for st in ("PREFLOP", "FLOP", "TURN", "RIVER"):
        fo = [d for d in rows if d["action"] == "FOLD" and d["street"] == st]
        fl = [d for d in fo if d["ev_call"] > 0]
        if fo:
            print(f"{st:<8} folds {len(fo):>6}  with +EV call {len(fl):>6} ({len(fl) / len(fo):.0%})"
                  f"  avg equity {sum(d['equity'] for d in fo) / len(fo):.2f}"
                  f"  avg required {sum(d['required'] for d in fo) / len(fo):.2f}")

    print("\n== Big showdown losses (net <= -40 chips): unlucky or bad commit? ==")
    print("   equity at our LAST call/raise decision vs a random hand; >=0.50 = unlucky, <0.35 = bad commit")
    last = {}
    for d in rows:
        if d["action"] != "FOLD":
            last[(d["match_id"], d["half"], d["deal_index"])] = d
    for opp in sorted(n_hands):
        big = [d for d in last.values() if d["opponent"] == opp and d["showdown"] and d["net"] <= -40]
        if big:
            print(f"{opp:<16} {len(big):>5} losses  unlucky {sum(d['equity'] >= .5 for d in big):>4}"
                  f"  middling {sum(.35 <= d['equity'] < .5 for d in big):>4}"
                  f"  bad commit {sum(d['equity'] < .35 for d in big):>4}")


if __name__ == "__main__":
    main()
