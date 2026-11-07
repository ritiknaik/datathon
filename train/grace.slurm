#!/bin/bash
# Parallel param search on Grace: one random-search chain per array task.
#   sbatch train/grace.slurm        then   python train/tune.py --pick runs
#SBATCH --job-name=pj-tune
#SBATCH --array=0-47
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=runs/log_%a.txt

module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true  # adjust to Grace's module list
cd "$SLURM_SUBMIT_DIR"
python train/tune.py --iters 40 --deals 400 --seed "$SLURM_ARRAY_TASK_ID" --out "runs/$SLURM_ARRAY_TASK_ID.json"
