#!/bin/bash
#SBATCH --job-name=organs_obj
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/lustre/nabeel/kamran/agrifusenet_revision/revision/results/slurm/organs_obj_%j.out

# Organ-only (SAM 3 masked) features, then the leakage-free per-organ and fusion evaluation.
ROOT=/mnt/lustre/nabeel/kamran/agrifusenet_revision
PY=/mnt/lustre/nabeel/kamran/llm-env-py311/bin/python
export HF_HOME=/mnt/lustre/nabeel/kamran/hf_cache TORCH_HOME=/mnt/lustre/nabeel/kamran/torch_cache XDG_CACHE_HOME=/mnt/lustre/nabeel/kamran/xdg_cache HF_HUB_OFFLINE=1
echo "organs_obj on $(hostname) $(date -Is)"
cd "$ROOT/revision/src"
$PY -W ignore organs_obj_extract.py 2 && \
$PY -W ignore organs_eval.py 0,1,2 --feat organs_obj_dinov2_s.npz --views obj,bgonly --aug aug_obj --tag obj; rc=$?
echo "finished $(date -Is) exit=$rc"
