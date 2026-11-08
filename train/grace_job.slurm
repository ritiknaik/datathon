#!/bin/bash
# Train the Q-net on Grace (GPU). Resources are sized to get scheduled fast and waste nothing:
#   any GPU type (gpu:1 -> A100, T4 or RTX, whichever frees first), 4 cores, 32 GB, 45 min.
#   Expected use: ~20M rows (~3 GB) loaded in a few minutes, 2 trainings x 20 epochs in ~10-20 min.
#
# Queue data generation AND this job with one command (training starts automatically when all data is done):
#   cd $SCRATCH/datathon
#   sbatch --dependency=afterok:$(sbatch --parsable train/grace_gen.slurm) train/grace_job.slurm
# Or, if the data already exists:  sbatch train/grace_job.slurm
#
# Output: $SCRATCH/pokerdata/q_weights.npz (all opponents) and q_weights_no_nit.npz (Nit left out = generalisation test)
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=grace_job
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/user/%u/pokerdata/logs/grace_job.%j.out

module restore poker
source $SCRATCH/pokervenv/bin/activate
cd $SCRATCH/datathon
D=$SCRATCH/pokerdata
ls $D/q/part7_*.npz >/dev/null 2>&1 || { echo "no data in $D/q (run train/grace_gen.slurm first)"; exit 1; }
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
echo "data files: $(ls $D/q/*.npz | wc -l)"
python train/train_q.py --data "$D/q/*.npz" --val $D/q/part7_*.npz --hidden 128 --epochs 20 --out $D/q_weights.npz
python train/train_q.py --data "$D/q/*.npz" --val $D/q/part7_*.npz --hidden 128 --epochs 20 --exclude add3 \
    --out $D/q_weights_no_nit.npz
