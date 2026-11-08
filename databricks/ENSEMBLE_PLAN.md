# Ensemble decision model: plan (Databricks training)

Team Straight Flush, bot Patrick Jane. Branch `ml-ensemble`. The rule-based backup is tag `rule-based-v3` (= `r-poc` at 4661f22).

**Status: Phase 0 done, result GO.** Next: the Databricks rounds (section 8).

## Phase 0 results (local pilot)

- **Data:** 21,627 labelled decisions from 307 work units (200 hands each, k = 4 playouts per action), generated in about 3 minutes on 10 cores (`train/policy_data.py`).
- **Model:** ridge + 3 bagged MLPs (`train/policy_train.py`), 0.03 ms per decision, 161 KB weights file. On held-out seeds, the offline value per labelled decision went from +3.78 chips (rules only) to +4.29 (w = 1).
- **Accept gate (`train/compare.py`, in-process, fresh seeds):** w = 1 gave +0.60 [+0.20, +1.01] (4 seeds), then +0.55 [+0.22, +0.88] (6 more). ACCEPT both times.
- **Real HTTP matches (`tools/versus.py`, 1000 deals per half, fresh servers):**
  - vs the rule-based backup: 3 wins, 17 chops, 0 losses; pooled +0.41 [+0.22, +0.59] bb/deal over 20,000 deals.
  - vs `spoc`: 14 wins, 6 chops, 0 losses; pooled +0.83 [+0.67, +0.98]. (The rule-based bot won 18 of 50 vs `spoc`.)
  - 0 errors; thinking 1.9 s median and 2.7 s max per half, while running 5 matches at once.
- **Unseen opponent styles (`spoc`'s `train/opps`, not in the training pool), mixed:**

  | Opponent | Rules | Ensemble |
  |---|---|---|
  | hyper_agg | +9.96 | +9.67 |
  | hyper_pass | **+3.46** | +2.56 |
  | adaptor | +0.67 | **+1.93** |

  About even on average. The model has partly learned our pool's styles, so the Databricks rounds must add **more varied opponents** (especially passive ones). Keep `spoc`'s three styles as a held-out test, never as training opponents.
- **Prerequisite done:** `sample_cost` set to 2e-5, so equity always uses `mc_n` = 60 samples (neutral, +0.11 not significant). Without it, the model's inputs at match time would differ from training.
- **Not run:** the Track B pilot (opponent-type experts). The ensemble showed a gain, so Track A goes first.


## 1. Why, and what "success" means

The bot today is rule-based: equity + pot odds give an EV per action, a softmax picks the action, and the numbers in `bot/params.json` are tuned by self-play. Two findings set the direction:

- **Hand strength is not the bottleneck.** The Databricks equity model was as accurate as our Monte Carlo and made no difference in bb/deal (11 fresh seeds, all within ±0.3).
- **Decisions are.** Every real gain came from decision logic (v1→v2 +1.6 bb/deal from tuning against strong opponents; v2→v3 +0.7 from fixing the small-blind discount bug). The 50-match battle vs `spoc` shows the next leak: we fold to a raise 53% on the turn and 60% on the river, and our biggest losses are folds of hands that would have won.

So the model learns **how good each action is in each situation**, and the bot blends that with its current rules.

**Success = all three, on seeds never used in training:**
1. `train/compare.py` says ACCEPT vs the current params (pooled head-to-head interval above 0, pool mean and worst result not lower).
2. `tools/versus.py` vs `origin/spoc`: no worse than today's 18 wins / 32 chops / 0 losses over 50 matches.
3. 0 errors, thinking ≤ 2 s per half under load, `bash tools/check_build.sh` passes.

**Stop rule:** if the local pilot (Phase 0) shows no gain at any blend weight, stop and spend the time elsewhere.

## 2. The model: an ensemble on top of the rules

For each legal action `a` in situation `x`:

```
score(a) = EV_rule(a)  +  w * sum_m  alpha_m * ( Q_m(x, a) - mean_a Q_m(x, a) )
action   = softmax(score / (temperature * pot))      # same sampling as today
```

| Member | What it is | Why it's in |
|---|---|---|
| Rules (`EV_rule`) | Today's bot | Safe floor: `w = 0` is exactly the current bot |
| `Q_mlp` x3-5 | Small numpy MLPs (about 30 → 64 → 64 → 4), each trained on a different bootstrap of the data | Main learner; bagging cuts variance |
| `Q_lin` | Ridge regression on the same features | Stable, hard to overfit; anchors the ensemble |
| `Q_gbm` (optional) | Gradient-boosted trees, **distilled** into an extra MLP student | Usually the most accurate on tabular data, but too slow to ship directly (3.6 ms per call last time) |

- `w` and the `alpha_m` are chosen by `train/compare.py`, not by training loss.
- **Fallback:** any import or predict error → `w = 0` for that decision (same pattern as the equity model hook).
- **Budget:** whole ensemble under 0.5 ms per decision on 1 CPU, files under 50 MB. Size is not the limit: about 4,000 decisions per half in 8 s is about 2 ms each, and the first 1 ms of every request is free. An 8-billion-parameter model would need about 16 GB (4 GB allowed) and seconds per call (1 s allowed), so it can't run at match time; bigger also wouldn't help, because label noise, not capacity, limits us.

### Alternative or complement (Track B): opponent-type experts

Cheaper and robust, lower ceiling. Tune a separate params set per opponent style (maniac, station, nit, balanced), then switch between them at match time from the opponent's stats after about 50 hands (with hysteresis, so it doesn't flip back and forth).

- Databricks fits well: each style's tuning runs in parallel.
- Run its pilot alongside Track A; it may beat the deep model on effort vs gain.

## 3. Data: what to generate, and why simple augmentation isn't enough

**States.** Self-play of the current bot vs a weighted opponent pool, recording a feature vector at each of our decisions.

- Pool: the 8 baselines (4 originals + `add1`-`add4`), past versions `bot/pool/v1..v3`, `spoc`'s params, and the style variants from `train/tune.py` (loose, sticky, tight).
- Weight the stronger opponents more, because the tournament field is unknown.
- Sample states across the whole half (`hand_index` 0-999), so the opponent stats in the features range from fresh to mature.
- **Oversample the leak:** take 3x as many states facing a raise on the turn and river.

**Labels (the expensive part).** For each sampled state, replay to that exact point, force each legal action, then play the rest of the hand K times with both players' current policies.

- The label for action `a` is the average chips we end the hand with: `Q(x, a)`.
- The deal fixes all 15 cards up front, so every action is replayed on the *same* cards. Only policy randomness differs between replays, which keeps label noise low.
- The opponent's real cards are used. That is unbiased, because the real deal is a fair sample of what the opponent could hold given the history. The model's features never include the opponent's cards.

**Why suit-swap augmentation won't help here.** The features (equity, pot odds, position, opponent stats) barely depend on suits, so a suit-swapped copy is nearly a duplicate row. What helps is more real states, more replays (K), and more varied opponents. That is "more data" in this plan.

**Sizes**

| Stage | States | K | Approximate cost |
|---|---|---|---|
| Phase 0 local pilot | 20,000 | 4 | about 1 hour on 10 cores |
| Databricks round 1 | 200,000 | 8 | about 9 core-hours (scale to the free-tier quota) |
| Databricks round 2+ | 200,000 per round | 8 | same, with the improved policy |

## 4. Features (one shared file, used by training AND by the bot)

`bot/policy_features.py` builds about 30 numbers from what the bot can see at decision time:

- **Strength:** equity, equity minus the required pot-odds equity, the rule EV of each action (4)
- **Price and size:** to_call / pot, pot / starting stack, our stack and the opponent's stack / starting stack
- **Situation:** street (one-hot, 4), we are the button, raises this street, who raised last, number of actions this hand
- **Opponent:** aggression rate, fold-to-raise rate, hands seen this half / 1000
- **Board:** paired, flush possible (3+ of a suit), straight possible, number of board cards

**Avoiding train/serve skew.** Equity at runtime uses 16 Monte Carlo samples early in a half and 60 later (the throttle assumes slow samples). The features must be computed the same way in training and at match time, so:

- **Step 0a:** set a fixed sample count by recalibrating `sample_cost` for the numpy speed (an earlier test of more samples was neutral, so this should cost nothing). Confirm with `compare.py`.
- Use the same count in data generation.
- Genie must **import** `bot/policy_features.py` and must never re-write it.

## 5. Training (Databricks)

- **Target:** advantage per legal action, `Q(x, a) - mean over legal actions`, in big blinds. Illegal actions are masked out of the loss.
- **Split by deal seed, not by row,** so replays of the same deal never sit in both train and test.
- **Metrics:**
  - advantage MAE;
  - how often the model's best action matches the label's best action;
  - an offline "policy value": the average label value of the action the blended policy would pick, for several `w`.
- **MLflow:** one experiment (`poker_policy_ensemble`), one run per member and per round. Log the data table versions.
- **Export:** a single `bot/policy_weights.npz` (all members + feature mean/scale) and a numpy-only `bot/policy_model.py` with `predict(features) -> 4 advantages`. Also save a parity file of 1,000 rows of features and expected outputs for the local check.

## 6. Who does what

| Step | Where | Owner |
|---|---|---|
| 0a. Fix the equity sample count, check with `compare.py` | local | Claude |
| 0b. `bot/policy_features.py`, `train/policy_data.py` (state generator + replay labeller, deterministic seeds), `bot/policy_model.py` hook in `strategy.py` (off by default: `use_policy_model`, `policy_w`), tests | local | Claude |
| 0c. **Pilot:** 20k labelled states, quick ridge + MLP fit, `compare.py` at `w` = 0.25 / 0.5 / 1 (Track A); quick per-style experts (Track B) | local | Claude |
| Go / no-go | — | you |
| G1-G6. Scale data, train the ensemble, export | Databricks | you (Genie prompts below) |
| 7. Pull the branch, parity test, latency test, accept gate, 50 matches vs `spoc`, Docker check | local | Claude |
| 8. Round 2: regenerate data with the accepted policy, retrain | Databricks | you + Genie |

Estimate: Phase 0 is about 2-3 hours including code. Each Databricks round is about 2-3 hours, depending on quota.

## 7. Risks

- **Small or no gain** (as with the equity model). The pilot and stop rule limit the loss.
- **Overfitting to our own pool.** Mitigated by a varied, weighted pool, the blend weight `w`, and held-out seeds. It can't be removed entirely, because the tournament field is unknown.
- **Train/serve skew.** Mitigated by one shared feature file, a fixed sample count, and a parity check.
- **A runtime crash costs a match.** Every model call has a fallback to the rules, plus a startup test with a broken model file (as for the equity model).

---

## 8. Genie prompts (run in order, one at a time; stop after each and check the result)

Before G1, Phase 0 is done and pushed to `ml-ensemble`. In Databricks, pull the Git folder to the latest `ml-ensemble` (not `r-poc`, which is the rule-based backup without the policy code).

**G0: setup check**
```
This repo is a heads-up Short Deck 5-card Pot-Limit Omaha poker bot. Read CLAUDE.md, context/notes.md
and databricks/ENSEMBLE_PLAN.md first. Pull the Git folder to the latest ml-ensemble. Add the repo root and
baseline-bots/ to sys.path, call evaluator.warmup(), and run
train.selfplay.match(strategy.load_params(), "add3", 100, "check"). Then import train.policy_data
and run its small self-test (policy_data.selftest()). Report compute type, available cores, and how long
selftest took. Do not edit any repo file.
```

**G1+G2 (merged in the code): generate AND label decision states at scale**
```
Use train.policy_data.generate_labeled(seed, opp_name, n_hands=200, k=8) from the repo; it plays one
work unit and returns labelled rows (features[30], ev[4], q[4], q_std[4], street, facing_raise, pot,
opponent, seed, hand_index). Do not re-implement any game, feature or policy code. The opponents and
their sampling weights come from train.policy_data.pool(). Build a Spark job (mapInPandas, about 200
partitions) over work units (seed "d1".."dN", opponent drawn by weight), so the total is about 200,000
rows. Use NEW seed prefixes (not "p", which the pilot used). Write Delta table
workspace.default.policy_rows. Run 20 units first and report the time per unit, then the full job.
Report the row counts by street, opponent and facing_raise.
```

(The two prompts below are kept for reference only: the merged prompt above replaces them.)

**G1 (old): generate decision states at scale**
```
Use train.policy_data.generate_states(seed, n_hands, opponent_spec) from the repo. Do not re-implement
any game, feature or policy code. Build a Spark job over (seed, opponent) work units: opponents and
weights are policy_data.POOL_WEIGHTS; seeds "s1".."sN". Each unit returns rows of
(state_id, seed, opponent, street, facing_raise, features[], legal_mask[4], replay_key).
Target 200,000 states in total, with turn/river states facing a raise oversampled 3x as
policy_data.OVERSAMPLE says. Write Delta table workspace.default.policy_states. Report the row counts by
street, opponent and facing_raise, and the time taken. Start with 2,000 states to check cost first.
```

**G2: label states by replay**
```
For every row of workspace.default.policy_states, call train.policy_data.label_state(replay_key, k=8),
which replays the hand to that point, forces each legal action and plays the hand out k times, and
returns q[4] in big blinds (NaN for illegal actions). Run it with mapInPandas over about 200 partitions,
seeded so it is reproducible. Write Delta table workspace.default.policy_labels with state_id, q[4],
q_std[4]. Do 5,000 rows first and report the average q_std per street, then the full table. Do not
change the labeller.
```

**G3: data quality checks**
```
Join policy_states and policy_labels. Report: rows with missing labels; mean and spread of advantage
(q minus the mean over legal actions) per street and per action; how often FOLD is the best action when
facing a raise, per street; and the label-noise floor (q_std / sqrt(k)). Flag anything odd and stop if
more than 1% of rows are broken. Do not drop or alter rows without telling me.
```

**G4: train the ensemble members**
```
Reference implementation: train/policy_train.py (ridge + bagged numpy MLPs, split by seed, masked
advantage loss, offline_value report, export). Run it on the Delta table at scale, or port it to the
cluster without changing the method, then add the members below.
Target = advantage per legal action (mask illegal ones out of the loss). Split train/test BY seed, never
by row. Train with features exactly as stored (they came from bot/policy_features.py):
(a) ridge regression, (b) 5 numpy MLPs 30->64->64->4 on bootstrap samples (seeds 1..5),
(c) optional: a gradient-boosted model, then distil it into one more MLP of the same size.
Log every run to MLflow experiment poker_policy_ensemble with the Delta table versions. Report per
member and for the plain average: advantage MAE, best-action agreement with the label, and the offline
policy value for blend weights w in {0, 0.25, 0.5, 1} (computed with train.policy_data.offline_value).
Also report predict latency for the whole ensemble on 1 CPU (p50/p95 per call).
```

**G5: export and parity**
```
Export all chosen members and the feature mean/scale into one file bot/policy_weights.npz, in the
format bot/policy_model.py reads (see its docstring; do not change that file). Write
bot/policy_parity.npz with 1,000 test rows of features and the ensemble's expected outputs. In a fresh
process, import bot.policy_model, check it reproduces the expected outputs to 1e-5, and measure p95
latency. Report both. The bot must stay numpy-only: no sklearn, MLflow, Spark or torch imports in bot/.
```

**G6: hand over**
```
Commit bot/policy_weights.npz, bot/policy_parity.npz and the notebook (databricks/policy_ensemble.ipynb)
to a new branch db-policy-model and push it. Do not change any other file. Do not put workspace paths
or email addresses in notebook outputs. Report the branch name, commit id and the MLflow run ids.
```

After G6, tell Claude: "db-policy-model is pushed". Step 7 (local gate) follows.

## 9. Round 2: retrain the correction layer on the `league` bot

**Why.** The shipped bot is now `league`: v3 rules + `spoc_db`'s Q-net (weight 0.85) + this ensemble as a correction (weight 0.5). The ensemble was trained on decisions and playouts of the *v3-only* bot, so it is now used on states and continuations it never saw. Round 2 relabels with the `league` bot itself playing (both in the forced-action spot and in every playout), and `spoc_db`'s bot (`pool/qnet1`) is now in the opponent pool. Same method as round 1, so the risk is low. Accept only if the local HTTP gate vs `league` passes.

**Branch:** in Databricks, pull the Git folder to the latest **`league`**. The kernel is long-lived: after any pull or branch switch, remove every repo module (`bot`, `train`, `game`, `evaluator`, `selfplay`, `policy_data`, `opponents` and their submodules) from `sys.modules` before importing.

**R0: setup check**
```
Read CLAUDE.md, context/notes.md and databricks/ENSEMBLE_PLAN.md (section 9) first. Pull the Git folder to
the latest `league` branch and clear the repo modules from sys.modules. Add the repo root, train/ and
baseline-bots/ to sys.path, call evaluator.warmup(). Print strategy.load_params()["qnet_weight"] (must be 0.85)
and sorted(train.policy_data.pool()) (must include "pool/qnet1"). Run
train.selfplay.match(strategy.load_params(), "pool/qnet1" spec from opponent_pool(), 100, "check") and
policy_data.selftest(). Report compute type, cores, and selftest time. Do not edit any repo file.
```

**R1: generate and label (same as G1+G2, new names)**
```
Same as G1+G2 in section 8, with these changes: seed prefix "L2-" (never reuse "d" or "p"), Delta table
workspace.default.policy_rows_r2, about 200,000 rows, 20-unit cost pilot first. Report rows by street,
opponent and facing_raise, and the table version.
```

**R2: checks**: run G3 on `workspace.default.policy_rows_r2`.

**R3: train**: run G4 on `policy_rows_r2` **only** (round-1 rows were labelled by a different bot, so their values are stale). MLflow experiment `poker_policy_ensemble`, run names ending `_r2`. Report offline value for w in {0, 0.25, 0.5, 1} as before.

**R4: export, parity, hand over**: run G5, then G6 but push to a new branch **`db-league-r2`** (only `bot/policy_weights.npz`, `bot/policy_parity.npz`, `databricks/policy_ensemble_r2.ipynb`; placeholder `<your-databricks-user>` for workspace paths). Never force-push.

After R4, tell Claude: "db-league-r2 is pushed". Gate: parity, `compare.py`, then 50 HTTP matches vs `league` and 25 vs `spoc_db`.
