#!/bin/bash
# Dedicated environment for SAM 3. Only the newer transformers stack is installed here; torch/numpy/cv2 are
# reused read-only from llm-env-py311 through a .pth entry that comes AFTER this env's own site-packages.
set -e
K=/mnt/lustre/nabeel/kamran
ENV=$K/sam3-env
export HF_HOME=$K/hf_cache PIP_CACHE_DIR=$K/pip_cache TMPDIR=$K/tmp
pkill -u nabeel -f "sam3-env/bin/pip" || true
rm -rf $ENV $K/tmp/*
$K/python-3.11.9/bin/python3.11 -m venv $ENV
SP=$($ENV/bin/python -c "import site; print(site.getsitepackages()[0])")
LLM_SP=$K/llm-env-py311/lib/python3.11/site-packages
echo "$LLM_SP" > $SP/zz_llm_env_fallback.pth
# pip sees llm-env's packages through the .pth, so only missing or too-old dependencies land in this env
$ENV/bin/python -m pip install -q --upgrade-strategy only-if-needed "transformers==5.22.1" huggingface_hub tokenizers safetensors accelerate
$ENV/bin/python -m pip list 2>/dev/null | grep -i -E "^(torch|numpy|transformers|huggingface)" || true
$ENV/bin/python - <<'PY'
import torch, transformers, huggingface_hub, numpy
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| transformers", transformers.__version__, "| hub", huggingface_hub.__version__, "| numpy", numpy.__version__)
from transformers import Sam3Model, Sam3Processor
print("Sam3 classes OK")
PY
$K/llm-env-py311/bin/python -c "import transformers; print('llm-env transformers unchanged:', transformers.__version__)"
echo BUILD_DONE
