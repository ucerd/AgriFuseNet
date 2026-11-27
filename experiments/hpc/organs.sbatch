#!/bin/bash
#SBATCH --job-name=organs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/lustre/nabeel/kamran/agrifusenet_revision/revision/results/slurm/organs_%j.out

# Organ-category feature extraction (GPU). Usage: sbatch --export=ALL,NAUG=4 organs.sbatch
ROOT=/mnt/lustre/nabeel/kamran/agrifusenet_revision
PY=/mnt/lustre/nabeel/kamran/llm-env-py311/bin/python
export HF_HOME=/mnt/lustre/nabeel/kamran/hf_cache TORCH_HOME=/mnt/lustre/nabeel/kamran/torch_cache XDG_CACHE_HOME=/mnt/lustre/nabeel/kamran/xdg_cache HF_HUB_OFFLINE=1
echo "organs naug=$NAUG on $(hostname) $(date -Is)"
cd "$ROOT/revision/src"
$PY -W ignore organs_extract.py "$NAUG"; rc=$?
echo "finished $(date -Is) exit=$rc"
