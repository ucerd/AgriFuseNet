#!/bin/bash
#SBATCH --job-name=ofusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/lustre/nabeel/kamran/agrifusenet_revision/revision/results/slurm/ofa_%A_%a.out

# One organ-fusion configuration per array task (arguments in hpc/ofusion_jobs.txt).
ROOT=/mnt/lustre/nabeel/kamran/agrifusenet_revision
PY=/mnt/lustre/nabeel/kamran/llm-env-py311/bin/python
export HF_HOME=/mnt/lustre/nabeel/kamran/hf_cache HF_HUB_OFFLINE=1
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/revision/hpc/ofusion_jobs.txt")
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $ARGS $(date -Is)"
cd "$ROOT/revision/src"
$PY -W ignore organ_fusion.py $ARGS; rc=$?
echo "finished $(date -Is) exit=$rc"
