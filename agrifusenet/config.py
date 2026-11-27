"""Constants of the method and the directory layout of the generated files."""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path

# Variety labels in class-index order.
VARIETIES = ["CP77-400", "CPF-237", "CPF-250", "CPF-251", "CPF-253"]

# Organ categories (folder names of the Sugarcane-PK dataset) used by the multi-organ model.
ORGANS = ["leafwidth", "leavePattern", "buddataset", "crossSection"]
ORGAN_NAMES = {"leafwidth": "leaf blade", "leavePattern": "leaf top", "buddataset": "bud",
               "crossSection": "stalk cross-section"}

# Concept prompts for Segment Anything 3; all instances of any prompt are united.
SAM3_PROMPTS = {"leafwidth": ["long narrow leaf", "grass blade", "green leaf"],
                "leavePattern": ["plant", "sugarcane leaf", "grass blade"],
                "buddataset": ["sugarcane stalk"],
                "crossSection": ["food", "slice"]}
SAM3_THRESHOLD = 0.5          # instance score threshold
SAM3_RETRY_THRESHOLD = 0.3    # used when no instance passes SAM3_THRESHOLD

# Leakage groups and capture periods.
NEAR_DUPLICATE_SIMILARITY = 0.985   # cosine similarity of DINOv2 embeddings
CAPTURE_BLOCK = 10                  # contiguous capture-order block size within a variety
N_PERIODS = 3                       # capture periods per variety for session-level evaluation

# Number of augmented copies per training image.
N_AUG_LEAF = 4
N_AUG_ORGAN = 2

# Embedding used to find near-identical images: the masked blade for the leaf blade (so that shots of one
# blade are grouped irrespective of the board), the full photograph for the other organs.
GROUP_EMBEDDING = {"leafwidth": "leaf", "leavePattern": "frame", "buddataset": "frame", "crossSection": "frame"}


@dataclass
class Paths:
    """Locations of the dataset and of every file produced by the scripts."""
    data_root: Path
    out_dir: Path

    def __post_init__(self):
        self.data_root, self.out_dir = Path(self.data_root), Path(self.out_dir)

    @property
    def metadata(self):
        return self.data_root / "metadata" / "samples.csv"

    def masks(self, kind):
        """kind: 'colour' or 'sam3'."""
        return self.out_dir / "masks" / kind

    @property
    def features(self):
        return self._dir("features")

    @property
    def groups(self):
        return self._dir("groups")

    @property
    def swap(self):
        return self._dir("background_swap")

    @property
    def results(self):
        return self._dir("results")

    def _dir(self, name):
        p = self.out_dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p


def add_path_args(parser: argparse.ArgumentParser):
    parser.add_argument("--data-root", default=os.environ.get("SUGARCANE_PK_ROOT"),
                        help="root of Sugarcane-PK (contains metadata/samples.csv); default $SUGARCANE_PK_ROOT")
    parser.add_argument("--out-dir", default=os.environ.get("AGRIFUSENET_OUT", "outputs"),
                        help="directory for masks, features, groups and results (default: outputs)")
    return parser


def paths_from_args(args) -> Paths:
    if not args.data_root:
        raise SystemExit("--data-root (or $SUGARCANE_PK_ROOT) must point to the Sugarcane-PK dataset")
    return Paths(args.data_root, args.out_dir)
