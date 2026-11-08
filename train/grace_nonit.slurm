#!/bin/bash
# Model 2 only: retrain with the Nit (add3) left out, to test generalisation to an unseen opponent.
# Reuses the existing data. Submit from the repo:  sbatch train/grace_nonit.slurm
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=grace_nonit
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/user/%u/pokerdata/logs/grace_nonit.%j.out

module restore poker
source $SCRATCH/pokervenv/bin/activate
cd $SCRATCH/datathon
D=$SCRATCH/pokerdata
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
echo "data files: $(ls $D/q/*.npz | wc -l), holdout files: $(ls $D/q/part7_*.npz | wc -l)"
python train/train_q.py --data "$D/q/*.npz" --val $D/q/part7_*.npz --hidden 128 --epochs 12 --batch 16384 \
    --exclude add3 --out $D/q_weights_no_nit.npz
