# Open Questions and Conflicts

Add an entry when context conflicts or a decision is unresolved. Remove it once the developer resolves it.

- Submission deadline: the guide still shows a placeholder ("[[Date and time]]").
- Time budget: the PDF says 2–4 ms per decision, the README says 4–7 ms. We plan for the stricter one.
- Format of the real-life scenario data, to be confirmed against the `bot/scenarios.json` rule format.
- Team work split: not decided yet.
- Before submission: the repo must be public, Docker `tools/check_build.sh` must pass, and team registration must be done.
- Grace: confirm the account, and the Python 3.10+ module name for `train/grace.slurm`.
- numpy at runtime: the bot now imports numpy (in the base image per the README). Confirm with `tools/check_build.sh` once Docker is running.
- Tuning trade-off: v3 is tuned for the worst of the pool (strong baselines); it earns less vs random than v1 did. Decide what we expect the tournament field to look like.
- Before making the repo public: `databricks/equity_model_pipeline.ipynb` contains a Databricks username/email path (ritiknaik23@tamu.edu) in its outputs. Strip it or remove the notebook if that is not wanted.
