#!/bin/bash
#SBATCH --job-name=ofusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/lustre/nabeel/kamran/agrifusenet_revision/revision/results/slurm/ofusion_%j.out

# Organ-set attention fusion vs late/early fusion. Usage: sbatch --export=ALL,SEEDS=0,PROTO=grouped ofusion.sbatch
ROOT=/mnt/lustre/nabeel/kamran/agrifusenet_revision
PY=/mnt/lustre/nabeel/kamran/llm-env-py311/bin/python
export HF_HOME=/mnt/lustre/nabeel/kamran/hf_cache TORCH_HOME=/mnt/lustre/nabeel/kamran/torch_cache XDG_CACHE_HOME=/mnt/lustre/nabeel/kamran/xdg_cache HF_HUB_OFFLINE=1
echo "ofusion seeds=$SEEDS proto=$PROTO on $(hostname) $(date -Is)"
cd "$ROOT/revision/src"
$PY -W ignore organ_fusion.py "$SEEDS" --protocol "$PROTO"; rc=$?
echo "finished $(date -Is) exit=$rc"
