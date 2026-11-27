#!/bin/bash
#SBATCH --job-name=agrifuse
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
# The fusion network is tiny and launch-bound on a GPU (<10% utilisation), so configurations run in parallel
# on CPU cores instead; this also keeps the numerics identical in kind to the laptop runs.
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/lustre/nabeel/kamran/agrifusenet_revision/revision/results/slurm/%A_%a.out

# One experiment configuration per array task; results are cached per configuration in results/runs/.
ROOT=/mnt/lustre/nabeel/kamran/agrifusenet_revision
PY=/mnt/lustre/nabeel/kamran/llm-env-py311/bin/python
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/revision/hpc/jobs.txt")
SET=$(echo "$LINE" | cut -d' ' -f1)
export ONLY=$(echo "$LINE" | cut -d' ' -f2-)
export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): set=$SET only=$ONLY  $(date -Is)"
cd "$ROOT/revision/src"
$PY -W ignore experiments.py "$SET" dinov2_s
echo "finished $(date -Is) exit=$?"
