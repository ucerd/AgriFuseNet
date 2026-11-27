#!/bin/bash
#SBATCH --job-name=agri_swap
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre/nabeel/kamran/agrifusenet_revision/revision/results/slurm/swap_%j.out
export ONLY='swap_dinov2_s_leaf\+tiles' CUDA_VISIBLE_DEVICES='' LOKY_MAX_CPU_COUNT=8 OMP_NUM_THREADS=8
cd /mnt/lustre/nabeel/kamran/agrifusenet_revision/revision/src
/mnt/lustre/nabeel/kamran/llm-env-py311/bin/python -W ignore experiments.py swap dinov2_s
echo finished $(date -Is)
