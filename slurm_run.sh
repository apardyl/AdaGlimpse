#!/bin/bash -l

#SBATCH -J wtlnv2
#SBATCH -N 1
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=48:00:00
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=4
#SBATCH -A plgactiveve-gpu-a100
#SBATCH --gres=gpu:4
#SBATCH --output="train.out"
#SBATCH -C memfs

# 90 seconds before training ends
#SBATCH --signal=SIGUSR1@90

cd $SLURM_SUBMIT_DIR

source $SCRATCH/conda/etc/profile.d/conda.sh

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

conda activate wtln3

srun python train.py ImageNet1k ClassificationRlMAE --data-dir $SCRATCH/imagenet --mem-fs --wandb --train-batch-size 128 --eval-batch-size 128 --rl-batch-size 128  --num-workers 8 --parallel-games 0 --num-glimpses 14 --pretrained-mae-path elastic_base.pth --replay-buffer-size 20000 --glimpse-grid-size 2
