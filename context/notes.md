# Notes (shared hackathon context)

Last updated: 2026-11-07 (merge of r-poc into spoc: v2 params kept, numpy equity, both opponent sets, both data pipelines)

## Problem statement
- TAMU Datathon Poker Bot Challenge: Heads-Up Short Deck (36 cards) 5-card Pot-Limit Omaha.
- Only `agent.py` (and what it imports) is ours. Its `choose_action` must return one of `obs.legal_actions`.
- Match = 1000 duplicate deals × 2 halves (1 deal = the same cards played twice with seats swapped = 2 hands). Win only if the 95% CI on bb/deal is entirely above 0; otherwise chop (0.5 points). Margin only matters for tiebreaks.
- **Hard limits (strict):** 8 s thinking per half, 1 s per request, 5 errors in a half = forfeit = match loss, server answers within 15 s. 1 CPU, 4 GB, no network.
- **Not a limit:** average ms per decision. The PDF's "2–4 ms" and README's "4–7 ms" are only the organisers' rough estimates (confirmed by organisers); we choose how to spend the 8 s.
- Tournament hardware: Kubernetes on a GCP VM with 96 vCPUs shared by all bots, so expect a busy, slower-than-laptop CPU.
- Full rules: the organisers' PDF guide (Heads_up.pdf, not in the repo) and `README.md`.

## Goal / winning idea
- Equity + pot-odds EV for each action, then a softmax **mixed strategy** (deliberately unpredictable to opponent-modelling bots).
- An opponent model (aggression, fold-to-raise) shifts call and bluff decisions.
- Trainable: every number is in `bot/params.json`, tuned by self-play (`train/tune.py`, Grace via `train/grace.slurm`), accepted only via `train/compare.py`.
- Real-life play experience goes in `bot/scenarios.json` as override rules.

## Team
- Team: **Straight Flush**. Bot: **Patrick Jane**.
- Work split: TBD. Branches: `r-poc` (Ritik), `spoc` (Sanket; merged r-poc on 2026-11-07), `db-equity-model` (learned equity model).

## Current state of the code
- `agent.py`: names set, and the 3 functions delegate to `bot/strategy.py` inside try/except (fallback CHECK_CALL). Server plumbing untouched.
- `bot/equity.py`: preflop features (linear weights fitted by `train/fit_preflop.py`, R²≈0.77) and Monte Carlo equity **vectorised with numpy** (`omaha_scores`, exact match with `evaluator.omaha_score`; pure-Python fallback). The score table is built at startup by `strategy.load_params()`.
- `bot/strategy.py`: EV, softmax, scenarios, an **optional learned equity model hook** (`bot/equity_model.py` with `predict(hole, board)`; ignored if absent, and errors fall back to Monte Carlo), and the time throttle. The throttle keeps a floor of `mc_min` post-flop samples until `time_budget` is spent (no cliff on a busy CPU).
- **Params: v2 kept** (`bot/params.json` = `bot/pool/v2.json`, `time_budget` 4.0). Merge comparison (`compare.py`, 14 opponents × 3 seeds × 1000 deals): v2 vs the adaptor params head-to-head −0.01, CI [−0.56, +0.54] (tie). The adaptor has a higher pool mean (3.91 vs 3.56) but loses to the Nit (−1.46 vs −0.29), so it was REJECTED. A tie scores more than a loss.
- **Judge over HTTP (verified fresh server), v2, seeds 21 and 23, 1000 deals each:** random +7.4/+6.5, always_pot +6.1/+5.7, calling_station +4.6/+4.8, always_half +11.5/+12.8, add1 ValueBettor +1.8/+1.5, add2 Maniac +2.0/+2.7, add4 Adaptive +2.1/+3.4 (all wins); add3 Nit 0.0/+0.0 (CHOP). 14 wins, 2 chops, 0 errors, thinking 1.2–4.5 s of 8 s per half.
- Simulator (3000 deals) agrees: v2 beats v1 +1.32, add1 +1.16, add2 +2.19, add4 +2.07, hyper_pass +2.32, calling station +4.59; ties add3 (−0.15) and the adaptor (−0.14).
- **Opponents:** 4 starter baselines; `add1..add4` (`baseline-bots/opponents.py`, servers `baseline-add-N.py`); `train/opps/{past_best,adaptor,hyper_pass,hyper_agg}.json`; past versions in `bot/pool/*.json`. All of these are in `selfplay.opponent_pool()`, used by tune.py and compare.py. The Nit (add3) is our hardest opponent (only a chop).
- **Data:** `tools/gen_hands.py` (full hand histories into Databricks `workspace.default.hands/actions`; `tools/audit.py` replays them) and `train/gen_logs.py` (one row per decision with our equity and opponent read). Both are documented: RUNNING.md and `DATABRICKS.md`. `data/` is git-ignored.
- Watch tools (off by default): judge `--verbose`; `BOT_TRACE=1` prints equity/EV/mix per decision.
- Audit findings (24k hands vs the 4 starter baselines; immediate pot odds only, so upper bounds): we fold too many hands worth calling, mostly vs always_pot and always_half. Lowering `bet_discount` hurt vs the stronger bots, so that is not the fix.
- How to run everything: `RUNNING.md`.
- **Branch `spoc_db` (work in progress): learned action-value net.** `bot/features.py` (39 inputs, shared by bot and training), `bot/qnet.py` (numpy inference ~14 µs; missing/broken `bot/q_weights.npz` = plays exactly like v2), blended into the EV via `qnet_weight` (0 = off, the default). Training only: exploration + per-decision "bb from now" logging in `train/selfplay.py`, `train/gen_q.py` (pool + random-style opponents, one .npz per run), `train/train_q.py` (PyTorch, holdout file, per-opponent loss, `--exclude` for leave-one-opponent-out), `train/grace_gen.slurm` (data, CPU array) and `train/grace_job.slurm` (training, any GPU; queue both with `sbatch --dependency=afterok:$(sbatch --parsable train/grace_gen.slurm) train/grace_job.slurm`) (HPRC QuickStart format; `module restore poker`, venv in `$SCRATCH/pokervenv`, data in `$SCRATCH/pokerdata`). Ship a net only after `compare.py` ACCEPTs it vs v2 and an HTTP judge check on fresh servers.

## Key decisions
- Keep the plumbing in `agent.py` unchanged. Strategy code lives in `bot/`.
- More thinking barely helps: `mc_n` 300 was neutral, and a training-only 20 s budget tied. Too little hurts: adaptor at 1.5 s lost −1.7 to 3 s.
- Accept new params only if `train/compare.py` says ACCEPT on fresh seeds (head-to-head CI > 0, pool mean and worst result not lower), then copy them to `bot/pool/vN.json`.
- **Training-only overrides** (which may exceed competition limits) go in `train/local/*.json` (git-ignored), passed with `--local` to `selfplay.py`/`tune.py`. They must never reach `bot/params.json` or the repo; `test_shipped_params_are_competition_legal` guards this.
- `selfplay.py` restarts the bots every 1000 deals (a real half), so `--deals 3000` = 3 real matches. Before this fix, runs over 1000 deals starved the time budget (results from such runs before the fix are invalid; tune/compare defaults were ≤1000, so v2's acceptance is valid).
- For close comparisons use ≥3000 deals: even a mirror match can look "significant" over 1000.
- **HTTP testing pitfall:** if a port is still taken, a new bot server exits with a bind error and the judge silently plays the OLD bot. This produced several wrong results on 2026-11-07. Always kill leftover servers and check for bind errors (RUNNING.md §2).
- Use `127.0.0.1`, not `localhost`, for judge URLs (the judge failed to reach the bots through `localhost` on Windows).
- numpy is used by the bot at runtime (`bot/equity.py`). The tournament base image already has it and `requirements.txt` is unchanged; there is a pure-Python fallback. Confirm with `tools/check_build.sh`.
