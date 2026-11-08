# Notes (shared hackathon context)

Last updated: 2026-11-07 (branch ml-ensemble: learned decision ensemble, Phase 0 GO)

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
- **Params v3 installed** (tune chain seed 7 started from v2 + `discount_needs_raise`; accepted by `compare.py` on 4 then 6 fresh seeds: head-to-head vs v2 +0.83 [+0.32, +1.34] then +0.70 [+0.29, +1.11]; pool mean 3.61 → 4.37, worst result −0.06 → +0.40). Saved as `bot/pool/v3.json` (v1 and v2 also kept there). Why: the audit showed v2 folded 14% of hands preflop (98% worth calling) because `bet_discount` was applied even when the small blind only completes the forced blind; `discount_needs_raise: 1` applies it only against a real raise, then the other numbers were re-tuned (bet_discount 0.24→0.37, temperature 0.079→0.052, called_penalty 0.14→0.089, agg_prior 0.44→0.14, fe_scale PREFLOP 1.5).
- Results of v3 (judge over HTTP, 1000 deals, seed 7): random +8.8, always_pot +6.3, calling_station +7.7, always_half +12.3, add1 +1.5, add2 +2.0, add3 +1.0, add4 +1.7 — all 8 wins, 0 errors (add3 on seed 21 was a chop, +0.5). v2 chopped add3 and v1 lost to it.
- Databricks (branch `db-equity-model`, merged here): equity model = numpy MLP trained on 10k Monte-Carlo-labelled deals with verified suit/order augmentations, MLflow experiment `poker_equity_model`, notebook in `databricks/`. Held-out MAE 0.045 (preflop 0.017 / flop 0.040 / turn 0.045 / river 0.069 vs our 60-sample MC 0.046 / 0.046 / 0.040 / 0.038); 0.2–0.5 ms per call. **Not used in the shipped bot**: `compare.py` on 11 fresh seeds showed no bb/deal gain (all variants within ±0.3), so `use_equity_model` stays off (opt-in param, any import/predict error falls back to Monte Carlo).
- Thinking time is now ≈0.1 s of the 8 s bank (each request gets 1 ms free and numpy equity mostly fits inside it).
- Experiments (selfplay, 8 opponents; all rejected unless stated): lowering `bet_discount` HURTS vs the stronger bots (so the audit's fold "leak" is not that); lower `temperature` helped slightly (mean ≈ +0.5); more MC samples (`mc_n` 300) neutral; Monte Carlo preflop (`preflop_mc_n`) worse with the current params, so left off (default 0); per-street fold tracking in the opponent model was significantly worse (−0.5 to −0.7) and was reverted; tuning rounds 2 (3 chains from v2) found nothing better than v2.
- How to run everything: `RUNNING.md`.
- Watch tools (all off by default, local only): judge `--verbose` prints every hand; `BOT_TRACE=1` prints the bot's equity/EV/mix per decision. `tools/gen_hands.py` writes `data/hands.csv` + `data/actions.csv` (git-ignored; loaded into Databricks as `workspace.default.hands/actions`). `tools/audit.py` replays them and compares our equity with pot odds.
- Audit findings (24k hands vs the 4 baselines; immediate pot odds only, so upper bounds): we fold too many hands worth calling, mostly vs always_pot (≈40 bb/100 hands) and always_half (≈27); 91% of preflop folds were +EV calls. Most big showdown losses were bad luck (equity ≥0.5), not bad commits. Part of the leak is the softmax noise (`temperature`).
- Not done yet: more tuning rounds (accept only via `train/compare.py`), `scenarios.json` rules, Docker build check (Docker CLI + Colima installed locally, daemon not started).

- **Branches:** `r-poc` = rule-based bot v3 (backup, also tag `rule-based-v3`, Docker check passed). `ml-ensemble` = rule-based v3 + learned decision ensemble (plan and results in `databricks/ENSEMBLE_PLAN.md`). `spoc` = teammate bot (v2 params + throttle floor). Over 50 HTTP matches, `r-poc` vs `spoc` was 18 wins / 32 chops / 0 losses, pooled +0.73 [+0.58, +0.87] bb/deal.
- **ml-ensemble bot:** score = rule EV + `policy_w` × ensemble advantage (ridge + 3 bagged MLPs on 30 features from `bot/policy_features.py`; weights `bot/policy_weights.npz`). Labels come from replaying each sampled decision with every action forced (`train/policy_data.py`); training is in `train/policy_train.py`. Opt-in via `use_policy_model` / `policy_w` (1.0); any model error falls back to the rules. `sample_cost` is 2e-5, so equity always uses 60 samples (training/match consistency).
- **ml-ensemble results:** compare.py ACCEPT twice (+0.60, +0.55 bb/deal vs v3). Over HTTP: vs the rule-based backup 3 wins / 17 chops / 0 losses, pooled +0.41 [+0.22, +0.59]; vs `spoc` 14 wins / 6 chops / 0 losses, pooled +0.83. Against unseen styles (`spoc`'s train/opps) it is about even: better vs adaptive, worse vs passive. Next: Databricks rounds with more varied opponents. Not Docker-checked on this branch yet.

## Key decisions
- Keep the plumbing in `agent.py` unchanged. Strategy code lives in `bot/`.
- `time_budget` is 4.0 s. The judge also counts ~1.5 s per half of request overhead, which the bot's own timer can't see, so the total is ≈5.5 of 8 s.
- `train/tune.py` scores every candidate on the same deals against the whole pool. Accept a winner only if `train/compare.py` says ACCEPT on fresh seeds, then copy it to `bot/pool/vN.json`.
- Use `127.0.0.1`, not `localhost`, for judge URLs (the judge failed to reach the bots through `localhost` on Windows).
- numpy is now also used by the bot at runtime (`bot/equity.py`). The tournament base image already has it and `requirements.txt` is unchanged; there is a pure-Python fallback. Confirm with `tools/check_build.sh`.
