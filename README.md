# AgriFuseNet

Background-free multi-organ feature fusion for image-based sugarcane variety identification, with
leakage-free and session-aware evaluation.

The code isolates every organ (leaf blade, leaf top, bud, stalk cross-section) from its acquisition background,
describes each organ with frozen deep features (and, for the leaf blade, with a dedicated encoder), and combines
the organs of a specimen with a permutation-invariant organ-set network. It also provides the evaluation
protocols and the diagnostics used to measure how much of a classifier's accuracy comes from the background or
from the capture session rather than from the plant.

This repository accompanies the article *AgriFuseNet: Background-Free Multi-Organ Feature Fusion for Sugarcane
Variety Identification under Leakage- and Session-Aware Evaluation* by Tassadaq Hussain and Kamran Younis
(Centre for AI and Big Data, Namal University, Mianwali, Pakistan).

## Method overview

| Stage | Module | Description |
|---|---|---|
| Curation | `agrifusenet/data.py` | Files whose content (SHA-256) occurs under more than one variety are removed. |
| Leakage groups | `agrifusenet/data.py` | Union-find over byte-identical files, near-identical images (cosine similarity >= 0.985 of DINOv2 embeddings) and contiguous capture-order blocks of 10 images within each variety. |
| Capture periods | `agrifusenet/data.py` | The capture sequence of every variety is divided into three contiguous periods. |
| Colour segmentation | `agrifusenet/masks.py` | Hue/saturation gate, morphology and GrabCut refinement against the reference boards. |
| SAM 3 segmentation | `agrifusenet/sam3.py` | Text-prompted Segment Anything 3; union of all instances above the score threshold, one retry at a lower threshold, colour mask as the last resort. |
| Views | `agrifusenet/views.py` | Masked organ, background only, organ-free background cells, corner mosaic, rectified blade tiles, organ crop. |
| Botanical descriptors | `agrifusenet/descriptors.py` | 164-D colour, vein-aligned texture (GLCM, LBP, Gabor), venation spectrum and morphometric descriptors computed on blade pixels only. |
| Leaf-blade encoder | `agrifusenet/leaf_encoder.py` | Channel-wise attention fusion of the masked-blade embedding, the blade-tile embedding and the botanical descriptors, with modality dropout; 128-D code classified by an SVM. |
| Organ-set fusion | `agrifusenet/organ_fusion.py` | Organ tokens with organ-type embeddings, masked mean (or attention) pooling over the organs present, organ dropout; late- and early-fusion baselines. |
| Evaluation | `agrifusenet/evaluation.py` | Repeated stratified group k-fold, session-level and cross-board protocols, metrics, corrected resampled t-test. |

## Data

The code is written for the public **Sugarcane-PK v1.0** dataset (Zenodo, DOI 10.5281/zenodo.18619387). Download
and extract it; the dataset root is the folder that contains `metadata/samples.csv` and one folder per organ
category (`leafwidth`, `leavePattern`, `buddataset`, `crossSection`, ...). No images are included here.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Segment Anything 3 is loaded from the Hugging Face Hub (`facebook/sam3`); accept the model licence on the Hub and
log in (`huggingface-cli login`) before the first run. A CUDA GPU is required for SAM 3 and recommended for
feature extraction and organ-set training; all other steps run on a CPU.

## Running the pipeline

```bash
export SUGARCANE_PK_ROOT=/path/to/Sugarcane_Varieties_Dataset
bash run_pipeline.sh outputs
```

`run_pipeline.sh` lists every step in order. Each script also runs on its own and prints its options with
`--help`. All files are written below `--out-dir` (default `outputs/`):

```
outputs/
  masks/colour/...            colour masks of the leaf blade (mirrors the dataset tree)
  masks/sam3/...              SAM 3 masks of the four organs
  features/                   leaf_<backbone>.npz, leaf_descriptors.parquet, leaf_baseline34.parquet,
                              frames_dinov2_s.npz, organ_crops_dinov2_s.npz, swap_*.npz / .parquet
  groups/<category>.csv       leakage groups, capture periods, background boards (leaf blade)
  background_swap/            counterfactual composites and their donor images
  results/                    evaluation outputs (JSON / CSV)
```

## Scripts

| Script | Purpose |
|---|---|
| `segment_colour.py` | Colour masks of one category. |
| `segment_sam3.py` | SAM 3 masks; `--rank-prompts` ranks candidate concept prompts on a calibration subset. |
| `extract_leaf_features.py` | Frozen backbone features of all leaf-blade views and of augmented copies (`dinov2_s`, `resnet50`, `convnext_t`, `effnet_b0`, `mobilenetv3`). |
| `extract_descriptors.py` | 164-D botanical descriptors and the 34-D whole-image baseline descriptor. |
| `extract_organ_features.py` | DINOv2 features of full frames and corner mosaics (`frames`) and of SAM 3 organ crops and background-only frames (`crops`). |
| `build_groups.py` | Leakage groups, capture periods and background boards. |
| `evaluate_leaf.py` | Any leaf-blade representation and classifier under the grouped, duplicate-only, random, session or cross-board protocol, optionally tested on background-swapped images. |
| `evaluate_organ_views.py` | Balanced accuracy of the frame, corner, background and organ-crop views of each organ. |
| `evaluate_organ_fusion.py` | Organ-set fusion, late and early fusion on identical specimens for every subset of organs. |
| `background_swap.py` | Counterfactual composites (blade pasted onto another variety's board) and their features. |
| `benchmark.py` | Inference cost of the leaf-blade encoder (CPU) and of the four-organ system (GPU). |
| `compare_runs.py` | Corrected resampled t-test between two leaf-blade runs. |

### Examples

```bash
A="--data-root $SUGARCANE_PK_ROOT --out-dir outputs"

# leaf-blade encoder and variants
python scripts/evaluate_leaf.py $A --inputs leaf tiles descriptors --model encoder --name encoder
python scripts/evaluate_leaf.py $A --inputs leaf tiles descriptors --model encoder --no-modality-dropout
python scripts/evaluate_leaf.py $A --inputs leaf descriptors --model scalar-gate
python scripts/evaluate_leaf.py $A --inputs leaf tiles descriptors --model encoder --head knn
python scripts/evaluate_leaf.py $A --inputs leaf tiles descriptors --model encoder --descriptor-groups col glcm lbp gab ven

# baselines and shortcut probes
python scripts/evaluate_leaf.py $A --inputs full --model svm --name full_image
python scripts/evaluate_leaf.py $A --inputs full baseline34 --model svm --reduce pca128
python scripts/evaluate_leaf.py $A --inputs background_cells --model svm
python scripts/evaluate_leaf.py $A --inputs leaf tiles descriptors --model encoder --protocol boards

python scripts/compare_runs.py outputs/results/leaf/encoder.json outputs/results/leaf/full_image.json

# organ-set fusion variants
python scripts/evaluate_organ_fusion.py $A --protocol session --organ-dropout 0
python scripts/evaluate_organ_fusion.py $A --protocol grouped --seeds 0,1,2,3,4 --leaf-encoder sam3-crop
python scripts/evaluate_organ_fusion.py $A --protocol grouped --seeds 0,1,2,3,4 --pairing period
```

## Evaluation protocols

* **Grouped**: 5-fold stratified group cross-validation over leakage groups, repeated with five seeds.
* **Session-level**: leave one capture period out (three splits); tests generalisation to unseen sessions.
* **Cross-board** (leaf blade): train on one background board design, test on the other.
* **Random**: ordinary stratified 5-fold cross-validation, which ignores leakage groups; for comparison only.

Augmented copies are generated for training images only, and hyper-parameters are tuned inside the training
part of each split. Multi-organ specimens are composed from images of one variety, one image per organ; training
specimens are drawn from training images and test specimens from test images, with identical test specimens
for all methods.

## Reproducibility notes

* All random choices are seeded (splits, augmentation, specimen draws, network initialisation).
* Results on a GPU and on a CPU can differ slightly because of non-deterministic floating-point reductions.
* Colour segmentation and feature extraction iterate over images in a fixed order (sorted relative paths). GrabCut
  draws from OpenCV's global random-number generator and the augmentation stream is seeded once per run, so masks
  and features should be computed for the full category in one run, as the scripts do.

## Citation

If you use this code, please cite the software release (metadata in `CITATION.cff`) and the dataset:

* Hussain, T., Younis, K. (2026). AgriFuseNet: background-free multi-organ feature fusion for sugarcane variety
  identification (version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.24151375
* Hussain, T. (2026). Sugarcane-PK v1.0: A Scale-Referenced Dataset and MATLAB Implementation for Precision
  Phenotyping of Pakistani Sugarcane Varieties [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.18619387

## Licence

Code: MIT (see `LICENSE`). The dataset and the pretrained models (DINOv2, ResNet50, ConvNeXt, EfficientNet,
MobileNetV3, SAM 3) are subject to their own licences.
