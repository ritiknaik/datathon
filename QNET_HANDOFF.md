# Handoff: the blended Q-net bot ("v2 + adaptive net")

**Audience:** a coding agent (or developer) on another branch who must (1) rebuild this exact bot from branch `spoc_db` and (2) compare it fairly against their own model.
**Read first:** `CLAUDE.md` and every file in `context/` (team rule). Follow the hard limits there.

---

## 1. What the model is (one paragraph)

The base bot (`bot/strategy.py`, params "v2") computes an EV per legal action from Monte Carlo equity, pot odds and an opponent model, then samples an action with a softmax. **The blended model adds a small neural net** that predicts, for each action, the big blinds we will win from this decision on. Final EV per action:

```
ev[a] = (1 - w) * formula_ev[a] + w * net_bb[a] * big_blind      (FOLD stays exactly 0)
w     = qnet_weight * clip((qnet_agg_hi - agg) / (qnet_agg_hi - qnet_agg_lo), 0, 1)
```

`agg` is the opponent's smoothed raise rate (already tracked by `bot/opponent.py`). With the shipped values, the net gets 50% weight against normal opponents and **fades to 0 against near-always-raising opponents** (raise rate 0.75 → 0.95), where the formula plays better.

## 2. Files (all on `spoc_db`)

| Path | Role |
|---|---|
| `bot/features.py` | Builds the 39 input features. **Single source of truth** for training and live play |
| `bot/qnet.py` | numpy-only inference (~18 µs/decision). Loads **`bot/q_weights.npz`** at import. If it's missing or broken, `qnet.LOADED = False` and the bot plays pure v2 |
| `bot/strategy.py` | `net_weight(P, agg)` (the fade) and the blend inside `decide()` |
| `models/grace_20261107_2122/q_weights.npz` | **The trained net** (39 → 128 → 128 → 4, ReLU) |
| `models/grace_20261107_2122/params_adaptive.json` | **The blended model's params** = v2 + `qnet_weight 0.5, qnet_agg_lo 0.75, qnet_agg_hi 0.95` |
| `models/grace_20261107_2122/train.log`, `data.log`, `summary.txt` | Training and data evidence |
| `bot/params.json` | v2 (the currently shipped bot). The net is **off** here (no `qnet_weight`) |
| `train/compare.py` | Acceptance test: candidates vs the 14-opponent pool on fresh seeds, plus head-to-head vs the first file |
| `train/selfplay.py` | Fast in-process duplicate match (no HTTP) |
| `train/gen_q.py`, `train/train_q.py`, `train/grace_*.slurm` | How the net was made (section 6) |

## 3. Rebuild the blended bot (exact steps)

```bash
git fetch origin && git checkout spoc_db && git pull origin spoc_db
python -m pip install -r requirements-dev.txt          # numpy, fastapi, uvicorn, pytest, httpx
cp models/grace_20261107_2122/q_weights.npz bot/q_weights.npz      # LOCAL ONLY, do not commit (see section 8)
python -c "from bot import qnet; print('net loaded:', qnet.LOADED)"  # must print True
python -m pytest -q                                     # all tests pass (21 at the time of writing)
```

Use it:
- **In-process** (simulator, `compare.py`): pass `models/grace_20261107_2122/params_adaptive.json` as the params file.
- **As an HTTP bot** (judge): `agent.py` always reads `bot/params.json`. Use a **separate worktree** (section 5B) and copy `params_adaptive.json` over `bot/params.json` **there only**.

Sanity check (should reproduce within noise):
```bash
python train/selfplay.py --params models/grace_20261107_2122/params_adaptive.json --opp add3 --deals 3000 --seed check
# expect roughly +1 bb/deal vs the Nit (v2 is about 0 there)
```

## 4. Results so far (simulator, in-process, fresh seeds)

14 opponents × 3 seeds × 1000 deals (bb/deal, positive = we win):

| Opponent | v2 | net 50% (no fade) | **blended (adaptive)** |
|---|---|---|---|
| random | 6.16 | 8.35 | 8.82 |
| call (calling station) | 4.44 | 8.90 | 8.88 |
| half (always half-pot) | 11.59 | 4.17 | 10.54 |
| pot (always pot) | 5.89 | 2.07 | 6.06 |
| add1 ValueBettor | 1.27 | 2.49 | 2.72 |
| add2 Maniac | 1.57 | 2.28 | 2.27 |
| add3 Nit | 0.02 | 1.10 | 1.35 |
| add4 Adaptive | 2.32 | 2.29 | 1.63 |
| pool/v1, pool/v2 | 1.31, −0.08 | 3.49, 1.14 | 3.01, 1.33 |
| opps/adaptor, hyper_agg, hyper_pass, past_best | 0.07, 9.62, 2.01, 2.09 | 2.52, 4.16, 3.57, 3.47 | 2.45, 5.28, 3.54, 3.19 |
| **mean / worst** | 3.45 / −0.08 | 3.57 / 1.10 | **4.36 / 1.33** |
| **head-to-head vs v2** | — | +2.04 [+1.48, +2.60] | **+1.02 [+0.44, +1.59]** |

- Blended vs plain net directly (3000 deals): +0.17 [−0.39, +0.73], a tie.
- Noise level: about ±0.6 bb/deal per cell. For example, vs add4 the two net versions play identically (add4 never reaches the fade), yet differ by 0.66.
- **Not yet done:** HTTP-judge verification of the blended bot (wins/chops, errors, time bank). Do this before shipping (section 5B).

## 5. How to compare it with YOUR model

Pick by how your model is built.

### 5A. Your model is a params file for this same engine (`bot/strategy.py`)
One command; it uses the same pool and fresh seeds:
```bash
python train/compare.py bot/params.json models/grace_20261107_2122/params_adaptive.json path/to/your_params.json \
    --deals 1000 --seeds cmp1 cmp2 cmp3
```
- The first file is the incumbent (v2). Each other file gets per-opponent results, mean, worst, and head-to-head vs v2 with a 95% CI.
- For your model **vs the blended one directly**:
  ```bash
  python train/selfplay.py --params path/to/your_params.json --opp models/grace_20261107_2122/params_adaptive.json --deals 3000 --seed h2h
  ```
- Limitation: only one net can be loaded per process (`bot/q_weights.npz`). If your params also use a Q-net with *different* weights, use 5B.

### 5B. Your model is different code (another agent.py/strategy, ML model, etc.)
Compare as real bots over HTTP, exactly like the tournament. Run the blended bot from its **own worktree** so the two codebases never mix:

```bash
git worktree add ../blended spoc_db
cp models/grace_20261107_2122/q_weights.npz              ../blended/bot/q_weights.npz
cp models/grace_20261107_2122/params_adaptive.json       ../blended/bot/params.json
```
Start servers, each in its own terminal (Mac/Linux shown; on Windows PowerShell use `$env:PORT=5008; python agent.py`):
```bash
cd ../blended && PORT=5008 python agent.py                  # blended bot
cd <your repo> && PORT=5010 python agent.py                 # your bot
PORT=5009 python baseline-bots/baseline_random.py           # optional opponents, any free ports:
PORT=5013 python baseline-bots/baseline-add-1.py            # add1..add4 = baseline-add-N.py (stronger bots)
```
Matches (1000 deals = 2000 hands = one tournament match; run ≥3 seeds for close results):
```bash
python judge_engine.py --agent1 http://127.0.0.1:5008 --agent2 http://127.0.0.1:5010 --seed 101
python judge_engine.py --agent1 http://127.0.0.1:5008 --agent2 http://127.0.0.1:5010 --seed 102
python judge_engine.py --agent1 http://127.0.0.1:5008 --agent2 http://127.0.0.1:5010 --seed 103
```
Also run **each** bot against the same opponents and seeds (random, always_pot, calling_station, always_half, add1–add4), so you compare like with like.

**Mandatory pitfalls (they produced wrong results here before):**
- Use `127.0.0.1`, **not** `localhost` (on Windows the judge couldn't reach the bots via `localhost`).
- **Stale servers:** if a port is busy, the new server exits with `error while attempting to bind`, and the judge silently plays the **old** bot. Before starting, kill leftovers (Mac/Linux `pkill -f "agent.py|baseline"`; Windows: see `RUNNING.md` §2) and check each server log for `bind` errors.
- Restart both bots after changing any params or weights (they're loaded at startup).

## 6. How the net was trained (for reproduction)
- **Data:** `train/gen_q.py`. v2 plays with 10% exploration (a random legal action) vs the 14-opponent pool plus **random-style opponents** (our engine with randomly drawn params). Each decision row has 39 features, the action taken, and the label "bb won from this decision on" (= hand net + chips already put in, so FOLD = 0).
  - Grace: `train/grace_gen.slurm`, 8 tasks × 12 cores × 480 matches × 1000 deals → **23.67M rows**, 64 chunks, 0 errors.
- **Training:** `train/train_q.py`. MLP 39 → 128 → 128 → 4, Huber loss on the taken action only, FOLD rows skipped, AdamW lr 1e-3.
  - Holdout = whole files (`part7_*`), so no hand is in both sets.
  - Grace T4: 20 epochs, batch 4096. Holdout loss **89.4 vs 115.7** for predict-the-average (−23%). Curve flat after ~13 epochs.
- **Pending:** a leave-the-Nit-out model (`train/grace_nonit.slurm`) to test generalisation to an unseen opponent.

## 7. Acceptance rule (team convention)
A version replaces the shipped bot only if, on **fresh seeds**:
1. Head-to-head vs the incumbent has its 95% CI entirely above 0.
2. Its pool mean is not lower.
3. Its worst opponent result is not lower.
4. Then an HTTP-judge check on fresh servers: 0 errors, no forfeits, thinking well under 8 s per half (hard limit; aim for ≤ 5 s).

Tournament scoring is win 1 / chop 0.5 / loss 0, decided by whether the 95% CI of bb/deal is entirely above or below 0. **A tie beats a loss; margins only break ties.**

## 8. Rules while working on this
- **Do not commit `bot/q_weights.npz`** or change `bot/params.json` on a shared branch until the model is accepted (section 7). The shipped bot must stay v2 until then.
- **Never `git push --force`.** It overwrote 4 commits on `spoc_db` once. If a push is rejected: `git pull origin <branch>`, then push.
- Training-only settings that break competition limits go in `train/local/` (git-ignored) via `--local`; never ship them.
- Update `context/notes.md` in the same push as any code or result change (team rule).
- Hard limits (tournament): 8 s thinking per half, 1 s per request, 5 errors in a half = forfeit = match loss, server up within 15 s, 1 CPU, 4 GB, no network.
