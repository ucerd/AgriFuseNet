#!/usr/bin/env bash
# Complete pipeline: segmentation, features, leakage groups and the main evaluations.
# Usage: SUGARCANE_PK_ROOT=/path/to/Sugarcane_Varieties_Dataset bash run_pipeline.sh [OUT_DIR]
# A CUDA GPU is required for Segment Anything 3 and recommended for feature extraction and organ fusion.
set -euo pipefail
OUT=${1:-outputs}
: "${SUGARCANE_PK_ROOT:?set SUGARCANE_PK_ROOT to the Sugarcane-PK dataset root}"
A="--data-root $SUGARCANE_PK_ROOT --out-dir $OUT"

# 1. Segmentation
python scripts/segment_colour.py $A --category leafwidth
python scripts/segment_sam3.py $A --categories leafwidth leavePattern buddataset crossSection

# 2. Features
for B in dinov2_s resnet50; do
  python scripts/extract_leaf_features.py $A --backbone $B
done
python scripts/extract_descriptors.py $A
python scripts/extract_organ_features.py $A frames
python scripts/extract_organ_features.py $A crops

# 3. Leakage groups, capture periods and background boards
python scripts/build_groups.py $A

# 4. Leaf blade: random versus grouped splits, background probe, encoder
python scripts/evaluate_leaf.py $A --inputs full --model svm --protocol random
python scripts/evaluate_leaf.py $A --inputs full --model svm --protocol grouped
python scripts/evaluate_leaf.py $A --inputs background_cells --model svm
python scripts/evaluate_leaf.py $A --inputs leaf tiles descriptors --model encoder

# 5. Background swap
python scripts/background_swap.py $A make
python scripts/background_swap.py $A features --backbone dinov2_s
python scripts/background_swap.py $A descriptors
python scripts/evaluate_leaf.py $A --inputs full --model svm --test-on swapped --seeds 0,1
python scripts/evaluate_leaf.py $A --inputs leaf tiles descriptors --model encoder --test-on swapped --seeds 0,1

# 6. Single-organ views and multi-organ fusion
python scripts/evaluate_organ_views.py $A
python scripts/evaluate_organ_fusion.py $A --protocol grouped --seeds 0,1,2,3,4
python scripts/evaluate_organ_fusion.py $A --protocol session

# 7. Inference cost
python scripts/benchmark.py $A leaf --backbone dinov2_s
python scripts/benchmark.py $A organs
