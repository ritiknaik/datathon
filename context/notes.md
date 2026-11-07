# Notes (shared hackathon context)

Last updated: 2026-11-07 (watch tools, hand data, decision audit)

## Problem statement
- TAMU Datathon Poker Bot Challenge: Heads-Up Short Deck (36 cards) 5-card Pot-Limit Omaha.
- Only `agent.py` (and what it imports) is ours. Its `choose_action` must return one of `obs.legal_actions`.
- Match = 1000 duplicate deals × 2 halves. Win only if the 95% CI on bb/deal is entirely above 0; otherwise chop (0.5 points).
- Limits: 8 s thinking per half, 1 s per request, 5 errors = forfeit = match loss. 1 CPU, 4 GB, no network, server must start in under 15 s.
- Full rules: the organisers' PDF guide (Heads_up.pdf, not in the repo) and `README.md`.

## Goal / winning idea
- Equity + pot-odds EV for each action, then a softmax **mixed strategy** (deliberately unpredictable to opponent-modelling bots).
- An opponent model (aggression, fold-to-raise) shifts call and bluff decisions.
- Trainable: every number is in `bot/params.json`, tuned by self-play (`train/tune.py`, Grace via `train/grace.slurm`).
- Real-life play experience goes in `bot/scenarios.json` as override rules.

## Team
- Team: **Straight Flush**. Bot: **Patrick Jane**.
- Work split: TBD.

## Current state of the code
- `agent.py`: names set, and the 3 functions delegate to `bot/strategy.py` inside try/except (fallback CHECK_CALL). Server plumbing untouched.
- `bot/equity.py`: preflop features (linear weights fitted by `train/fit_preflop.py` on 5000 hands × 300 samples, R²≈0.77) and Monte Carlo equity.
- `bot/opponent.py`: opponent stats kept in `memory`. `bot/strategy.py`: EV, softmax, scenarios, and a time-budget throttle.
- `train/`: `selfplay.py` (fast in-process duplicate match), `fit_preflop.py`, `tune.py`, `grace.slurm`.
- Results (judge, 1000 deals, seeds 11 and 42): random +12.5/+11.2, always_pot +7.3/+7.0, calling_station +8.1/+8.6, always_half +11.2/+12.2 bb/deal. 8/8 wins, 0 errors, ≤5.5 s thinking per half.
- Tuning round 1 (6 local chains × 15 iters): no candidate beat the current params on held-out deals (one lost significantly). Tuning against the baselines alone overfits, so the next round must tune against stronger opponents (past bests, aggressive/passive variants).
- Time budget matters: `time_budget` 3.0 lost −2.0 bb/deal (significant) to 4.0. A faster equity engine (more MC samples) is the clearest strength lever.
- How to run everything: `RUNNING.md`.
- Watch tools (all off by default, local only): judge `--verbose` prints every hand; `BOT_TRACE=1` prints the bot's equity/EV/mix per decision. `tools/gen_hands.py` writes `data/hands.csv` + `data/actions.csv` (git-ignored; loaded into Databricks as `workspace.default.hands/actions`). `tools/audit.py` replays them and compares our equity with pot odds.
- Audit findings (24k hands vs the 4 baselines; immediate pot odds only, so upper bounds): we fold too many hands worth calling, mostly vs always_pot (≈40 bb/100 hands) and always_half (≈27); 91% of preflop folds were +EV calls. Most big showdown losses were bad luck (equity ≥0.5), not bad commits. Part of the leak is the softmax noise (`temperature`).
- Not done yet: opponent pool in tuning, faster/vectorised `mc_equity`, learned equity model (Databricks), Docker build check (Docker CLI + Colima installed locally, daemon not started).

## Key decisions
- Keep the plumbing in `agent.py` unchanged. Strategy code lives in `bot/`.
- `time_budget` is 4.0 s. The judge also counts ~1.5 s per half of request overhead, which the bot's own timer can't see, so the total is ≈5.5 of 8 s.
- `train/tune.py` scores every candidate in a chain on the same deals (fair comparison). Always validate winners on held-out seeds; don't trust `--pick` alone.
- Use `127.0.0.1`, not `localhost`, for judge URLs (the judge failed to reach the bots through `localhost` on Windows).
- numpy is a dev-only dependency (training). The base image already has it, and `requirements.txt` is unchanged.
