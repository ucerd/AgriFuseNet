#!/bin/bash
#SBATCH --job-name=organs_eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/lustre/nabeel/kamran/agrifusenet_revision/revision/results/slurm/organs_eval_%j.out

# Multi-organ feasibility evaluation (CPU). Usage: sbatch --dependency=afterok:<extract job> organs_eval.sbatch
ROOT=/mnt/lustre/nabeel/kamran/agrifusenet_revision
PY=/mnt/lustre/nabeel/kamran/llm-env-py311/bin/python
export HF_HOME=/mnt/lustre/nabeel/kamran/hf_cache TORCH_HOME=/mnt/lustre/nabeel/kamran/torch_cache XDG_CACHE_HOME=/mnt/lustre/nabeel/kamran/xdg_cache HF_HUB_OFFLINE=1
echo "organs_eval on $(hostname) $(date -Is)"
cd "$ROOT/revision/src"
$PY -W ignore organs_eval.py 0,1,2; rc=$?
echo "finished $(date -Is) exit=$rc"
