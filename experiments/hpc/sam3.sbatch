#!/bin/bash
#SBATCH --job-name=sam3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/lustre/nabeel/kamran/agrifusenet_revision/revision/results/slurm/sam3_%j.out

# SAM 3 organ masks in the dedicated sam3-env. Usage: sbatch --export=ALL,ARGS="crossSection --sample 6" sam3.sbatch
ROOT=/mnt/lustre/nabeel/kamran/agrifusenet_revision
PY=/mnt/lustre/nabeel/kamran/sam3-env/bin/python
export HF_HOME=/mnt/lustre/nabeel/kamran/hf_cache TORCH_HOME=/mnt/lustre/nabeel/kamran/torch_cache XDG_CACHE_HOME=/mnt/lustre/nabeel/kamran/xdg_cache HF_HUB_OFFLINE=1
echo "sam3 $ARGS on $(hostname) $(date -Is)"
cd "$ROOT/revision/src"
eval "$PY -W ignore sam3_masks.py $ARGS"; rc=$?
echo "finished $(date -Is) exit=$rc"
