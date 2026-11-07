# Notes (shared hackathon context)

Last updated: 2026-11-07 (bot v1.2: adaptor params, 3 s budget, no-cliff throttle, training-only overrides)

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
- An opponent model (aggression, fold-to-raise) shifts call and bluff decisions. The current params adapt fast (`prior_weight` 4).
- Trainable: every number is in `bot/params.json`, tuned by self-play (`train/tune.py`, Grace via `train/grace.slurm`).
- Real-life play experience goes in `bot/scenarios.json` as override rules.

## Team
- Team: **Straight Flush**. Bot: **Patrick Jane**.
- Work split: TBD.

## Current state of the code
- `agent.py`: names set, and the 3 functions delegate to `bot/strategy.py` inside try/except (fallback CHECK_CALL). Server plumbing untouched.
- `bot/equity.py`: preflop features (linear weights fitted by `train/fit_preflop.py`, R²≈0.77) and Monte Carlo equity.
- `bot/opponent.py`: opponent stats kept in `memory`. `bot/strategy.py`: EV, softmax, scenarios, and the time-budget throttle.
- `bot/params.json` = the "adaptor" settings plus `time_budget` 3.0. Head-to-head vs the previous params (`train/opps/past_best.json`): +1.41 bb/deal over 3000 deals, CI [+0.83, +1.99].
- Trade-off: the new params beat smarter opponents but take less from simple ones (vs calling_station ≈ +6 alone vs the old params' +8). Every baseline is still a significant win, and points only need CI > 0.
- Judge (1000 deals, seed 7): random +10.2, always_pot +5.9, calling_station +7.4, always_half +11.9 bb/deal. 4/4 wins, 0 errors, ≤5.05 s thinking per half.
- Test opponents: the 4 starter baselines, plus `train/opps/{past_best,adaptor,hyper_pass,hyper_agg}.json` (our strategy code with other params; use them via `selfplay.py --opp <json>`). Hardest: adaptor and hyper_pass.
- How to run everything: `RUNNING.md`.
- Data for analysis: `train/gen_logs.py` writes one row per decision (with the hand outcome) vs all 8 opponents. The Databricks workflow (Delta table, Genie leak-finding, logistic fit of "what an opponent's bet means") is in `DATABRICKS.md`. `data/` is git-ignored.

## Key decisions
- Keep the plumbing in `agent.py` unchanged. Strategy code lives in `bot/`.
- `time_budget` 3.0 s. Tests: 3 vs 4 s was a tie, 1.5 s lost −1.7, and 20 s (training-only) was a tie, so extra thinking barely helps. The judge also charges ~2 s per half of request overhead the bot can't see, so local total ≈5 of 8 s.
- Throttle never drops post-flop simulation to 0 while budget remains (keeps a floor of `mc_min` samples). A busy CPU previously made the bot fall back to preflop-only equity and lose ~2.5 bb/deal.
- **Training-only overrides** (which may exceed competition limits) go in `train/local/*.json` (git-ignored), passed with `--local` to `selfplay.py`/`tune.py`. They must never reach `bot/params.json` or the repo; `test_shipped_params_are_competition_legal` guards this.
- `selfplay.py` restarts the bots every 1000 deals (a real half), so `--deals 3000` = 3 real matches. Before this fix, runs over 1000 deals starved the time budget; the old claims "3 s budget loses −2.0" and "a faster equity engine is the main lever" came from that bug and are **wrong**.
- For close comparisons, use ≥3000 deals: even a mirror match can look "significant" over 1000.
- `train/tune.py` scores every candidate in a chain on the same deals. Validate winners on held-out seeds; don't trust `--pick` alone.
- Use `127.0.0.1`, not `localhost`, for judge URLs (the judge failed to reach the bots through `localhost` on Windows).
- numpy is a dev-only dependency (training). The base image already has it, and `requirements.txt` is unchanged.
