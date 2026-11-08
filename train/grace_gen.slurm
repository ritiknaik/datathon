#!/bin/bash
# Q-net training data on Grace (CPU, short partition). One .npz per array task in $SCRATCH/pokerdata/q/.
# Before the first submit:  mkdir -p $SCRATCH/pokerdata/q $SCRATCH/pokerdata/logs
# Submit from the repo:     sbatch train/grace_gen.slurm
# Cost: about 8 tasks x 12 cores x ~1 h = ~100 SUs. Expected output: ~20M decisions.
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=poker_gen
#SBATCH --time=01:50:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --partition=short
#SBATCH --array=0-7
#SBATCH --output=/scratch/user/%u/pokerdata/logs/gen.%A_%a.out

module restore poker                      # saved in setup step 1 (GCCcore + Python/3.11.3)
source $SCRATCH/pokervenv/bin/activate
cd $SCRATCH/datathon
hostname; python --version
python train/gen_q.py --matches 480 --deals 1000 --explore 0.1 --workers $SLURM_CPUS_PER_TASK \
    --seed $SLURM_ARRAY_TASK_ID --out $SCRATCH/pokerdata/q/part$SLURM_ARRAY_TASK_ID.npz
