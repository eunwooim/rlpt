#!/usr/bin/env bash
# Merge a Qwen2.5-VL LoRA adapter into a full Hugging Face checkpoint.
#
# Usage:
#   bash train/merge_lora.sh <lora_adapter_dir> <output_hf_dir> [base_model]
#
# Example:
#   bash train/merge_lora.sh \
#     /mnt/data1/eunwooim/rlpt/outputs/exp-name/hf_global_step_285/lora_adapter \
#     /mnt/data1/eunwooim/rlpt/outputs/exp-name/hf_global_step_285_lora_merged \
#     Qwen/Qwen2.5-VL-3B-Instruct

set -euo pipefail
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <lora_adapter_dir> <output_hf_dir> [base_model]"
    exit 1
fi

ADAPTER_DIR="$1"
OUTPUT_DIR="$2"
BASE_MODEL="${3:-Qwen/Qwen2.5-VL-3B-Instruct}"

ADAPTER_DIR="$(readlink -f "${ADAPTER_DIR}")"
mkdir -p "$(dirname "${OUTPUT_DIR}")"
OUTPUT_DIR="$(readlink -m "${OUTPUT_DIR}")"

PATCHED_ADAPTER_DIR="${ADAPTER_DIR%/}_patched_for_merge"

if [ ! -d "${ADAPTER_DIR}" ]; then
    echo "Adapter directory does not exist: ${ADAPTER_DIR}"
    exit 1
fi

if [ ! -f "${ADAPTER_DIR}/adapter_config.json" ]; then
    echo "Missing adapter_config.json in ${ADAPTER_DIR}"
    find "${ADAPTER_DIR}" -maxdepth 2 -type f | sort || true
    exit 1
fi

if [ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" ]; then
    echo "Missing adapter_model.safetensors in ${ADAPTER_DIR}"
    find "${ADAPTER_DIR}" -maxdepth 2 -type f | sort || true
    exit 1
fi

echo "============================================================"
echo "Merging Qwen2.5-VL LoRA"
echo "============================================================"
echo "BASE_MODEL          = ${BASE_MODEL}"
echo "ADAPTER_DIR         = ${ADAPTER_DIR}"
echo "PATCHED_ADAPTER_DIR = ${PATCHED_ADAPTER_DIR}"
echo "OUTPUT_DIR          = ${OUTPUT_DIR}"
echo "============================================================"

rm -rf "${PATCHED_ADAPTER_DIR}"
cp -r "${ADAPTER_DIR}" "${PATCHED_ADAPTER_DIR}"

# Patch veRL-exported adapter_config.json.
# veRL target_modules=all-linear can export target names that vanilla PEFT
# interprets too broadly. We restrict to valid leaf Linear module names.
PATCHED_ADAPTER_DIR="${PATCHED_ADAPTER_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

from safetensors.torch import load_file

adapter = Path(os.environ["PATCHED_ADAPTER_DIR"])
cfg_path = adapter / "adapter_config.json"
sd_path = adapter / "adapter_model.safetensors"

print("patching adapter:", adapter)

cfg = json.loads(cfg_path.read_text())
sd = load_file(sd_path)

allowed = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv",
    "proj",
}

seen = set()

for key in sd.keys():
    if ".lora_A." in key:
        prefix = key.split(".lora_A.")[0]
        name = prefix.split(".")[-1]
        if name in allowed:
            seen.add(name)
    elif ".lora_B." in key:
        prefix = key.split(".lora_B.")[0]
        name = prefix.split(".")[-1]
        if name in allowed:
            seen.add(name)

target_modules = sorted(seen)

if not target_modules:
    print("First 50 adapter keys:")
    for key in list(sd.keys())[:50]:
        print(key)
    raise RuntimeError("Could not infer valid target_modules from adapter weights")

bad_numeric = {str(i) for i in range(100)}
bad = sorted(set(target_modules) & bad_numeric)
if bad:
    raise RuntimeError(f"Invalid numeric target_modules inferred: {bad}")

print("Using target_modules:")
for name in target_modules:
    print(" ", name)

cfg["target_modules"] = target_modules
cfg.pop("auto_mapping", None)

cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
print("Wrote patched adapter config:", cfg_path)
PY

rm -rf "${OUTPUT_DIR}"

BASE_MODEL="${BASE_MODEL}" \
PATCHED_ADAPTER_DIR="${PATCHED_ADAPTER_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
python - <<'PY'
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

base_model = os.environ["BASE_MODEL"]
adapter_dir = os.environ["PATCHED_ADAPTER_DIR"]
output_dir = os.environ["OUTPUT_DIR"]

print("Loading base model:", base_model)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter:", adapter_dir)

model = PeftModel.from_pretrained(
    model,
    adapter_dir,
    is_trainable=False,
)

print("Merging LoRA weights")
model = model.merge_and_unload()

print("Saving merged model:", output_dir)
model.save_pretrained(
    output_dir,
    safe_serialization=True,
    max_shard_size="4GB",
)

print("Saving processor/tokenizer")
processor = AutoProcessor.from_pretrained(
    base_model,
    trust_remote_code=True,
)
processor.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
)
tokenizer.save_pretrained(output_dir)

out = Path(output_dir)

required = [
    "config.json",
    "tokenizer.json",
    "preprocessor_config.json",
]

for name in required:
    path = out / name
    print(name, "OK" if path.exists() else "MISSING")

has_index = (out / "model.safetensors.index.json").exists()
has_single = (out / "model.safetensors").exists()

print("model weights:", "OK" if has_index or has_single else "MISSING")

if not (has_index or has_single):
    raise RuntimeError("Merged model weights were not saved correctly")

print("Done:", output_dir)
PY

echo "============================================================"
echo "Merged model written to:"
echo "${OUTPUT_DIR}"
echo
echo "Evaluate with:"
echo "CUDA_VISIBLE_DEVICES=5 python eval/scienceqa.py --model_path ${OUTPUT_DIR}"
echo
echo "Use for GRPO with:"
echo "MODEL_PATH=${OUTPUT_DIR}"
echo "USE_PEFT=0"
echo "============================================================"