"""Text-prompted organ segmentation with Segment Anything 3 (facebook/sam3, Hugging Face Transformers).

All instances of any of the organ's concept prompts whose score exceeds the threshold are united. If no
instance is found, the threshold is lowered once; if there is still none, the colour mask is used.
"""
import numpy as np
import torch

from .config import SAM3_PROMPTS, SAM3_THRESHOLD, SAM3_RETRY_THRESHOLD
from .masks import colour_mask

MODEL_ID = "facebook/sam3"


def load_model(device=None):
    from transformers import Sam3Model, Sam3Processor
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Sam3Model.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()
    return model, Sam3Processor.from_pretrained(MODEL_ID)


@torch.no_grad()
def segment(model, processor, image, prompts, threshold=SAM3_THRESHOLD):
    """Union of all instance masks of the prompts. Returns (mask, best score, number of instances)."""
    W, H = image.size
    union, best, n = np.zeros((H, W), bool), 0.0, 0
    for p in prompts:
        inputs = processor(images=image, text=p, return_tensors="pt").to(model.device)
        inputs = {k: (v.to(model.dtype) if torch.is_floating_point(v) else v) for k, v in inputs.items()}
        out = model(**inputs)
        r = processor.post_process_instance_segmentation(out, threshold=threshold, mask_threshold=0.5,
                                                         target_sizes=inputs["original_sizes"].tolist())[0]
        for m, s in zip(r["masks"], r["scores"]):
            union |= m.cpu().numpy().astype(bool)
            best = max(best, float(s))
            n += 1
    return union, best, n


def segment_organ(model, processor, image, organ, prompts=None):
    """Organ mask with the fallback sequence; returns (mask, score, instances, source)."""
    prompts = prompts or SAM3_PROMPTS[organ]
    m, s, n = segment(model, processor, image, prompts)
    source = "sam3"
    if not m.any():
        m, s, n = segment(model, processor, image, prompts, threshold=SAM3_RETRY_THRESHOLD)
        source = "sam3_retry"
    if not m.any():
        m, source = colour_mask(np.asarray(image), organ) > 0, "colour"
    return m, s, n, source
