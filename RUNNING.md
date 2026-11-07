# Running, simulating and tuning the bot (Mac + Windows)

Bot: **Patrick Jane**, team **Straight Flush**. The strategy is in `bot/`, every tunable number is in `bot/params.json`, and the training tools are in `train/`.
Every command below runs from the repo root. Where Mac and Windows differ, both are shown.

---

## 1. One-time setup

**Needs:** Python 3.10 or newer, and Git. Docker is only needed for the pre-submission build check (section 7).

| | Mac (Terminal) | Windows (PowerShell) |
|---|---|---|
| Install Python | `brew install python@3.11` | python.org installer, tick "Add to PATH" |
| Clone | `git clone https://github.com/ritiknaik/datathon.git && cd datathon` | same |
| Create venv | `python3 -m venv .venv` | `python -m venv .venv` |
| Activate venv | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` |
| Install deps | `pip install -r requirements-dev.txt` | same |

- The repo is private for now. If `git clone` asks for a password, use `gh auth login` or a GitHub personal access token.
- Windows: if activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
- **Activate the venv in every new terminal.** Once it is active, `python` means the venv's Python on both OSes.

Check that everything works:
```
python -m pytest -q          # expect: 14 passed
```

---

## 2. Play a real match through the judge (same as the tournament)

Open **three terminals**, activate the venv in each, then:

```
# Terminal 1: our bot (port 5008)
python agent.py

# Terminal 2: an opponent (port 5009); pick one
python baseline-bots/baseline_random.py
python baseline-bots/baseline_always_pot.py
python baseline-bots/baseline_always_half.py
python baseline-bots/baseline_calling_station.py

# Terminal 3: the judge
python judge_engine.py --agent1 http://127.0.0.1:5008 --agent2 http://127.0.0.1:5009 --seed 7
```

> Always use `127.0.0.1`, not `localhost`. On Windows the judge failed to reach the bots through `localhost`, and an unreachable bot forfeits.

### Judge options
| Option | Effect |
|---|---|
| `--deals 200` | Quick run (default is 1000 deals per half) |
| `--seed 42` | Different cards. The same seed always deals the same cards |
| `--json out.json --logs` | Save the full result and every hand to a file |

### Reading the report
| Line | Meaning |
|---|---|
| `agent1: +9.649 bb/deal` | Our average win per deal, in big blinds |
| `Confidence interval [+8.0, +11.3]` | Entirely above 0 = **win**, entirely below 0 = loss, includes 0 = chop (draw) |
| `thinking 3.14s of 8s ... errors 0` | Time bank used per half. **Warning signs:** more than 6 s, any errors, `FORFEITED`, `TIME BANK EMPTY` |
| `RESULT: AGENT1_WIN` | Final verdict |

Terminal 1 shows the bot's live log. If the strategy ever crashes, its traceback prints there (the bot falls back to check/call instead of being charged an error).

### Our bot against another version of itself
Start a second copy on port 5010, then point the judge's `--agent2` at it:

| Mac | Windows |
|---|---|
| `PORT=5010 python agent.py` | `$env:PORT=5010; python agent.py` |

```
python judge_engine.py --agent1 http://127.0.0.1:5008 --agent2 http://127.0.0.1:5010
```
The bot only reads `bot/params.json` when it starts, so restart it after editing that file.

### Tournament-exact mode (fresh servers each half)
```
python judge_engine.py --agent1 http://127.0.0.1:5008 --agent2 http://127.0.0.1:5009 --half 1 --seed 7 --json h1.json
# restart both bots (Ctrl+C, run again)
python judge_engine.py --agent1 http://127.0.0.1:5008 --agent2 http://127.0.0.1:5009 --half 2 --seed 7 --json h2.json
python judge_engine.py combine h1.json h2.json
```

### Stop the servers
Press Ctrl+C in each terminal. If a port says "already in use":

| Mac | Windows |
|---|---|
| `lsof -ti :5008 \| xargs kill` | `Get-NetTCPConnection -LocalPort 5008 -State Listen \| % { Stop-Process -Id $_.OwningProcess -Force }` |

---

## 3. Fast simulation (no servers, about 10× quicker)

This plays a full duplicate match in-process. Use it for trying ideas.
```
python train/selfplay.py --opp random --deals 500          # opponents: random | call | half | pot
python train/selfplay.py --opp pot --deals 1000 --seed abc
python train/selfplay.py --opp bot/params_old.json         # vs another saved params file
```
Output: `bb/deal 11.556  CI [8.995, 14.118]` (our result, and its 95% interval).

**Workflow for testing a change:**
1. Copy `bot/params.json` to `bot/params_old.json`.
2. Edit `bot/params.json`.
3. Run `python train/selfplay.py --opp bot/params_old.json --deals 1000`.
4. Keep the change only if the CI is entirely above 0, and it still beats all 4 baselines.

### What the parameters do (`bot/params.json`)
| Key | Effect |
|---|---|
| `preflop` | Weights for preflop hand strength (fitted by `train/fit_preflop.py`, don't hand-edit) |
| `temperature` | Randomness. Higher is less predictable, but has more variance and leads to more chops |
| `bet_discount` | How much to respect an opponent's bet (scaled down against aggressive opponents) |
| `called_penalty` | How much weaker our equity is assumed to be when a raise gets called |
| `fe_scale` | Per-street multiplier on the opponent's fold-to-raise rate (bluffing) |
| `agg_prior`, `fold_prior`, `prior_weight` | Starting guesses about the opponent before data arrives |
| `time_budget` | Seconds of thinking per half that the bot allows itself (the tournament limit is 8) |
| `mc_n`, `mc_min`, `sample_cost` | Monte Carlo sample count, and when it drops to the fast fallback |

### Real-life scenarios (`bot/scenarios.json`)
These are hand-written rules that override the bot's choice. The first matching rule wins. The `when` keys are all optional: `street`, `facing` (whether we face a bet), `is_button`, `eq_min`, `eq_max` (our equity, 0–1).
```json
[
  {"when": {"street": "RIVER", "facing": true, "eq_max": 0.25}, "probs": {"FOLD": 0.9, "CHECK_CALL": 0.1}},
  {"when": {"street": "FLOP", "is_button": true, "eq_min": 0.8}, "probs": {"RAISE_POT": 0.7, "CHECK_CALL": 0.3}}
]
```
Action names: `FOLD`, `CHECK_CALL`, `RAISE_HALF`, `RAISE_POT`. Illegal actions are skipped automatically.

---

## 4. Inspecting individual hands

After a judge run with `--json out.json --logs`, list the 10 biggest pots. Our bot is seat 0 in half 1 and seat 1 in half 2; the script handles that.
```
python -c "import json; d=json.load(open('out.json')); L=sorted(d['logs'],key=lambda h:-h['pot'])[:10]; [print(h['half'],h['deal_index'],'pot',h['pot'],'our net',h['nets'][h['half']-1],h['hole_cards'],h['board_revealed'],[a['action'] for a in h['actions']]) for h in L]"
```
Other useful fields in `out.json`: `paired_bb` (our result per deal), `ci`, and `agent_stats` (time and errors per half).

---

## 5. Re-fit preflop strength

This simulates random hands, then fits the `preflop` weights in `bot/params.json`:
```
python train/fit_preflop.py --hands 3000 --samples 200     # about 45 s on a laptop
```

---

## 6. Parameter tuning

### Locally
```
python train/tune.py --iters 20 --deals 300 --seed 1 --out runs/1.json
python train/tune.py --pick runs        # copy the best run into bot/params.json
```
It prints `iteration  score  best` per attempt. Run several seeds in parallel terminals for more coverage.

### On the Grace cluster (Slurm)
```
ssh <netid>@grace.hprc.tamu.edu
git clone https://github.com/ritiknaik/datathon.git && cd datathon
module spider Python                          # find a Python 3.10+ module; put its name in train/grace.slurm
mkdir -p runs && sbatch train/grace.slurm     # 48 parallel searches
squeue -u $USER                               # job status
tail -f runs/log_0.txt                        # live progress of one search
python train/tune.py --pick runs              # after the jobs finish: install the best params
```
Then commit the new `bot/params.json`, and **re-run section 2 against all 4 baselines** before pushing.

---

## 7. Before submitting

1. `python -m pytest -q` passes.
2. Full 1000-deal judge matches against all 4 baselines, with a few seeds: **0 errors, under 6 s thinking per half, all wins**.
3. Docker build check, with Docker Desktop running:

   | Mac | Windows |
   |---|---|
   | `./tools/check_build.sh` | Run in Git Bash or WSL: `./tools/check_build.sh` |

   It must end with `ALL CHECKS PASSED`.
4. Push, make the repo **public**, then get the full commit ID with `git rev-parse HEAD` and submit it in the form.
