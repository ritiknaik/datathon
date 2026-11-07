# Notes (shared hackathon context)

Last updated: 2026-11-07 (stronger baselines, opponent pool, numpy equity, params v2)

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
- `bot/equity.py`: preflop features (linear weights fitted by `train/fit_preflop.py` on 5000 hands × 300 samples, R²≈0.77) and Monte Carlo equity, now **vectorised with numpy** (`omaha_scores`, exact match with `evaluator.omaha_score`; pure-Python fallback if numpy is missing). The score table is built at startup by `strategy.load_params()`.
- `bot/opponent.py`: opponent stats kept in `memory`. `bot/strategy.py`: EV, softmax, scenarios, a time-budget throttle, and an **optional learned equity model hook** (`bot/equity_model.py` with `predict(hole, board)`; ignored if absent, and any error falls back to Monte Carlo).
- `baseline-bots/baseline-add-1..4.py`: stronger practice opponents (1 ValueBettor, 2 Maniac, 3 Nit, 4 Adaptive). Logic in `baseline-bots/opponents.py`, also used in-process by `train/selfplay.py` as `add1`..`add4`.
- `train/`: `selfplay.py` (fast in-process duplicate match), `fit_preflop.py`, `tune.py` (scores vs the whole pool: 8 baselines + `bot/pool/*.json` + 3 style variants; score = mean of the 4 worst), `compare.py` (held-out accept test), `grace.slurm`.
- **Params v2 installed** (tune chain seed 1, accepted by `compare.py` on 4 fresh seeds: head-to-head vs v1 +1.61 bb/deal, CI [+1.10, +2.11]; pool mean equal, worst result −2.47 → −0.03). Changes: temperature 0.05→0.079, bet_discount 0.12→0.24, called_penalty 0.08→0.14, agg_prior 0.3→0.44, fold_prior 0.3→0.17, fe_scale lower on late streets. Old version kept as `bot/pool/v1.json`, new as `bot/pool/v2.json`.
- Results of v2 (judge over HTTP, 1000 deals, seed 7): random +7.1, always_pot +5.8, calling_station +4.9, always_half +12.2, add1 +1.3, add2 +2.3, add4 +2.0 (all wins), add3 −0.5 (CHOP, was a significant loss with v1). 0 errors. v2 wins less than v1 vs the weakest baselines (random, calling_station) in exchange for much better results vs strong ones.
- Thinking time is now ≈0.1 s of the 8 s bank (each request gets 1 ms free and numpy equity mostly fits inside it).
- Experiments (selfplay, 8 opponents): lowering `bet_discount` HURTS vs the stronger bots (so the audit's fold "leak" is not that); lower `temperature` helped slightly (mean ≈ +0.5); more MC samples (`mc_n` 300) neutral; Monte Carlo preflop (`preflop_mc_n`) worse with the current params, so left off (default 0).
- How to run everything: `RUNNING.md`.
- Watch tools (all off by default, local only): judge `--verbose` prints every hand; `BOT_TRACE=1` prints the bot's equity/EV/mix per decision. `tools/gen_hands.py` writes `data/hands.csv` + `data/actions.csv` (git-ignored; loaded into Databricks as `workspace.default.hands/actions`). `tools/audit.py` replays them and compares our equity with pot odds.
- Audit findings (24k hands vs the 4 baselines; immediate pot odds only, so upper bounds): we fold too many hands worth calling, mostly vs always_pot (≈40 bb/100 hands) and always_half (≈27); 91% of preflop folds were +EV calls. Most big showdown losses were bad luck (equity ≥0.5), not bad commits. Part of the leak is the softmax noise (`temperature`).
- Not done yet: learned equity model from Databricks (hook is ready), more tuning rounds (accept only via `train/compare.py`), `scenarios.json` rules, Docker build check (Docker CLI + Colima installed locally, daemon not started).

## Key decisions
- Keep the plumbing in `agent.py` unchanged. Strategy code lives in `bot/`.
- `time_budget` is 4.0 s. The judge also counts ~1.5 s per half of request overhead, which the bot's own timer can't see, so the total is ≈5.5 of 8 s.
- `train/tune.py` scores every candidate on the same deals against the whole pool. Accept a winner only if `train/compare.py` says ACCEPT on fresh seeds, then copy it to `bot/pool/vN.json`.
- Use `127.0.0.1`, not `localhost`, for judge URLs (the judge failed to reach the bots through `localhost` on Windows).
- numpy is now also used by the bot at runtime (`bot/equity.py`). The tournament base image already has it and `requirements.txt` is unchanged; there is a pure-Python fallback. Confirm with `tools/check_build.sh`.
