#!/bin/bash
# Train the Q-net on Grace (1 GPU). Run after grace_gen.slurm has finished.
# Submit from the repo:  sbatch train/grace_train.slurm
# Writes $SCRATCH/pokerdata/q_weights.npz (all opponents) and q_weights_no_nit.npz (Nit left out, to test generalisation).
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=poker_train
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --output=/scratch/user/%u/pokerdata/logs/train.%j.out

module restore poker
source $SCRATCH/pokervenv/bin/activate
cd $SCRATCH/datathon
nvidia-smi --query-gpu=name,memory.total --format=csv
D=$SCRATCH/pokerdata
python train/train_q.py --data "$D/q/*.npz" --val $D/q/part7.npz --hidden 128 --epochs 20 --out $D/q_weights.npz
python train/train_q.py --data "$D/q/*.npz" --val $D/q/part7.npz --hidden 128 --epochs 20 --exclude add3 \
    --out $D/q_weights_no_nit.npz
