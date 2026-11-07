# Notes (shared hackathon context)

Last updated: 2026-11-07 (bot v1: equity + mixed strategy + opponent model)

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
- `bot/equity.py`: preflop features (linear weights fitted by `train/fit_preflop.py`, R²≈0.66) and Monte Carlo equity.
- `bot/opponent.py`: opponent stats kept in `memory`. `bot/strategy.py`: EV, softmax, scenarios, and a time-budget throttle.
- `train/`: `selfplay.py` (fast in-process duplicate match), `fit_preflop.py`, `tune.py`, `grace.slurm`.
- Results (judge, 1000 deals, seed 7): beats random +9.6, always_pot +6.3, calling_station +7.4, always_half +11.4 bb/deal. All wins, 0 errors, ≤5 s thinking per half.
- How to run everything: `RUNNING.md`.

## Key decisions
- Keep the plumbing in `agent.py` unchanged. Strategy code lives in `bot/`.
- `time_budget` is 4.0 s, leaving margin under the 8 s limit for a slower tournament machine.
- Use `127.0.0.1`, not `localhost`, for judge URLs (the judge failed to reach the bots through `localhost` on Windows).
- numpy is a dev-only dependency (training). The base image already has it, and `requirements.txt` is unchanged.
