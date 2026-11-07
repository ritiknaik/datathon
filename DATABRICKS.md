# Databricks runbook: find our leaks with data

**Goal:**
1. Log every decision our bot makes against the 8 test opponents.
2. Load the logs into a Delta table and ask Genie where we lose chips.
3. Bring the findings back as `bot/params.json` or `bot/scenarios.json` changes, and validate them locally.

Everything here is **training-only**. The shipped bot never reads Databricks: it has no internet in the tournament.

---

## Step 0: Pull the latest code
1. Workspace → your repo Git folder (`datathon`) → switch to branch **`spoc`** → **Pull**.
2. Check that `train/gen_logs.py` and `train/opps/` exist. If not, the pull didn't get the latest commit.

> Create your notebooks in your **user folder**, not inside the repo folder, so they don't show up as repo changes.

## Step 1: Notebook setup (Python cell)
```python
import os, subprocess, sys
REPO = "/Workspace/Users/<your-email>/datathon"     # <- path of the Git folder (right-click > Copy path)
CATALOG, SCHEMA, VOLUME = "workspace", "default", "poker"   # change if your catalog/schema differ
DATA = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/decisions_v1"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(os.path.exists(f"{REPO}/train/gen_logs.py"), os.cpu_count(), "cores")
```

## Step 2: Generate the data
Our bot plays every opponent for `--deals` deals (1 deal = 2 hands). The output is one CSV per opponent.
```python
r = subprocess.run([sys.executable, f"{REPO}/train/gen_logs.py", "--deals", "1000", "--out", DATA],
                   capture_output=True, text=True)
print(r.stdout, r.stderr[-3000:])
```
- **Start with 1000 deals** (~7k rows per opponent) to check that everything works. Then rerun with `3000` or more into a new folder (`decisions_v2`, ...).
- Speed: about 5 minutes for 3000 deals on 8 cores. Free-tier compute is smaller, so expect longer.
- If it's too slow, generate locally or on Grace with the same command, then upload the CSVs to the Volume (Catalog → Volume → Upload).

## Step 3: Build the Delta table (Python cell)
```python
from pyspark.sql import functions as F
df = (spark.read.option("header", True).option("inferSchema", True).csv(DATA)
      .withColumn("eq_bucket", F.floor(F.col("eq") * 10) / 10)          # 0.0, 0.1, ... 0.9
      .withColumn("facing_bet", (F.col("to_call") > 0).cast("int")))
df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{CATALOG}.{SCHEMA}.poker_decisions")
display(df.groupBy("opp").agg(F.count("*").alias("decisions"), F.avg("net_bb").alias("avg_hand_bb")))
```

### Columns
| Column | Meaning |
|---|---|
| `opp` | Opponent: random, call, half, pot, past_best, adaptor, hyper_pass, hyper_agg |
| `hand`, `street`, `is_button` | Hand number in the half, betting round, whether we act last after the flop |
| `eq`, `eq_bucket` | Our estimated chance of winning vs a random hand (0–1) |
| `pot`, `to_call`, `pot_odds`, `facing_bet` | Money situation. `pot_odds` = the share of the final pot we must put in to call |
| `raises_this_street`, `opp_last`, `opp_raises_hand` | What the opponent did this hand |
| `opp_agg`, `opp_fold` | Our running estimate of the opponent's raise rate and fold-to-raise rate |
| `action` | What we did: FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT |
| `net_bb`, `showdown`, `won` | Hand outcome: bb won or lost, whether cards were shown, whether we won |

> `net_bb` is the whole hand's result, repeated on each of its decisions. Compare averages between groups of decisions rather than adding `net_bb` up.

## Step 4: Genie space
**New → Genie space**, then add the table `poker_decisions`.

**Instructions** (paste into the space's General instructions):
> This table logs every decision our poker bot made. net_bb is the bb won/lost in the whole hand the decision belongs to (positive = good). eq is our estimated win chance vs a random hand. Compare average net_bb between groups with at least 200 rows; ignore smaller groups. A "leak" is a group of decisions with clearly negative average net_bb that a different action could avoid.

**Questions to ask:**
1. Average net_bb and count by opp, street and action. Which combinations are most negative?
2. When facing_bet = 1 on the RIVER and we chose CHECK_CALL, show average net_bb by eq_bucket and opp.
3. For action in (RAISE_HALF, RAISE_POT) with eq < 0.4 (bluffs), show the win rate without showdown (won = 1 and showdown = 0) by street and opp.
4. Where do we FOLD with eq ≥ 0.5? Count and average pot by opp and opp_last.
5. Compare decisions against hyper_pass and call: in which street × eq_bucket is our average net_bb lowest?
6. Is is_button = 1 better than is_button = 0 per opp?

## Step 5: Genie Code (Databricks Assistant): fit what an opponent's bet really means
Paste into a new notebook cell's Assistant:
> Using table workspace.default.poker_decisions, filter showdown = 1. Per street, fit a logistic regression predicting `won` from: eq, facing_bet, opp_raises_hand, pot_odds, opp_agg, and one-hot opp_last. Use scikit-learn, 80/20 split, and report test AUC and a calibration table (predicted vs actual win rate in 10 bins). Then print the coefficients per street as one JSON dict. Keep the code short.

**What we want from it:**
- How much our real win chance drops below `eq` when the opponent has bet or raised. Today `bet_discount` (0.2) and `called_penalty` (0.12) are single guesses.
- The coefficients JSON. Post it in the team chat, and we'll wire it into `bot/strategy.py` as a few numbers in `params.json`.

## Step 6: Bring findings back (local machine)
1. Turn each leak into either a param change or a rule in `bot/scenarios.json` (format in `RUNNING.md` §3).
2. Validate against the previous version over 3 matches:
   ```
   python train/selfplay.py --opp train/opps/past_best.json --deals 3000
   ```
   Keep the change only if the CI is above 0. Then re-check all 8 opponents and the judge (`RUNNING.md` §2, §7).
3. Commit with a `context/notes.md` update (team rule).
4. For the next round, regenerate the data with the new params (`decisions_v2`) and repeat.
