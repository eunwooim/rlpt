#!/usr/bin/env bash
# SFT | ScienceQA | Qwen2.5-VL-3B-Instruct | veRL | FSDP | LoRA

set -euo pipefail
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <nproc_per_node> <save_path> [extra verl overrides...]"
    exit 1
fi

nproc_per_node="$1"
save_path="$2"
shift 2

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/data1/eunwooim/rlpt/src}
DATA_DIR=${DATA_DIR:-/mnt/data1/eunwooim/rlpt/data/scienceqa_sft}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/val.parquet}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}

HF_HOME=${HF_HOME:-/mnt/shared/shared_hf_home}
export HF_HOME
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ============================================================
# Data / batching
# ============================================================

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
MAX_LENGTH=${MAX_LENGTH:-4096}

USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-65536}

# Used only if USE_DYNAMIC_BSZ=False.
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-16}

PAD_MODE=${PAD_MODE:-no_padding}
TRUNCATION=${TRUNCATION:-error}
NUM_WORKERS=${NUM_WORKERS:-2}

# ============================================================
# Optimizer
# ============================================================

LR=${LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
BETAS=${BETAS:-"[0.9,0.95]"}
CLIP_GRAD=${CLIP_GRAD:-1.0}

LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
LR_WARMUP_STEPS_RATIO=${LR_WARMUP_STEPS_RATIO:-0.03}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:--1}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}

# ============================================================
# Engine
# ============================================================

BACKEND=${BACKEND:-fsdp}
FSDP_STRATEGY=${FSDP_STRATEGY:-fsdp2}
FSDP_SIZE=${FSDP_SIZE:--1}
SP_SIZE=${SP_SIZE:-1}

ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-False}

# ============================================================
# LoRA
# ============================================================

USE_PEFT=${USE_PEFT:-0}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_TARGETS=${LORA_TARGETS:-all-linear}

# ============================================================
# Runtime
# ============================================================

SEED=${SEED:-42}
FULL_DETERMINISM=${FULL_DETERMINISM:-False}

PROJECT_NAME=${PROJECT_NAME:-rlpt-scienceqa-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25vl3b-scienceqa-sft-lora${LORA_RANK}}

SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-50}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-3}
RESUME_MODE=${RESUME_MODE:-disable}

export WANDB_DIR=${WANDB_DIR:-${save_path}/wandb}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export TORCH_MP_SHARING_STRATEGY=${TORCH_MP_SHARING_STRATEGY:-file_system}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export TRANSFORMERS_NO_ADVISORY_WARNINGS=${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}
export PYTHONHASHSEED="${SEED}"

ulimit -n 1048576 || ulimit -n 65535 || true

# ============================================================
# Checks
# ============================================================

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Missing TRAIN_FILE: ${TRAIN_FILE}"
    exit 1
fi

if [ ! -f "${VAL_FILE}" ]; then
    echo "Missing VAL_FILE: ${VAL_FILE}"
    exit 1
fi

mkdir -p "${save_path}" "${WANDB_DIR}"
cd "${PROJECT_ROOT}"

cat <<EOF
============================================================
ScienceQA Qwen2.5-VL SFT
============================================================
PROJECT_ROOT              = ${PROJECT_ROOT}
TRAIN_FILE                = ${TRAIN_FILE}
VAL_FILE                  = ${VAL_FILE}
save_path                 = ${save_path}
MODEL_PATH                = ${MODEL_PATH}
nproc_per_node            = ${nproc_per_node}

TRAIN_BATCH_SIZE          = ${TRAIN_BATCH_SIZE}
MAX_LENGTH                = ${MAX_LENGTH}
USE_DYNAMIC_BSZ           = ${USE_DYNAMIC_BSZ}
MAX_TOKEN_LEN_PER_GPU     = ${MAX_TOKEN_LEN_PER_GPU}
MICRO_BATCH_SIZE_PER_GPU  = ${MICRO_BATCH_SIZE_PER_GPU}
PAD_MODE                  = ${PAD_MODE}
TRUNCATION                = ${TRUNCATION}

LR                        = ${LR}
WEIGHT_DECAY              = ${WEIGHT_DECAY}
BETAS                     = ${BETAS}
CLIP_GRAD                 = ${CLIP_GRAD}
TOTAL_EPOCHS              = ${TOTAL_EPOCHS}

BACKEND                   = ${BACKEND}
FSDP_STRATEGY             = ${FSDP_STRATEGY}
FSDP_SIZE                 = ${FSDP_SIZE}
SP_SIZE                   = ${SP_SIZE}

ATTN_IMPL                 = ${ATTN_IMPL}
USE_REMOVE_PADDING        = ${USE_REMOVE_PADDING}
USE_TORCH_COMPILE         = ${USE_TORCH_COMPILE}

USE_PEFT                  = ${USE_PEFT}
LORA_RANK                 = ${LORA_RANK}
LORA_ALPHA                = ${LORA_ALPHA}
LORA_TARGETS              = ${LORA_TARGETS}

PROJECT_NAME              = ${PROJECT_NAME}
EXPERIMENT_NAME           = ${EXPERIMENT_NAME}
============================================================
EOF

# ============================================================
# Metadata
# ============================================================

{
    echo "date=$(date)"
    echo "script=$0"
    echo "nproc_per_node=${nproc_per_node}"
    echo "save_path=${save_path}"
    echo "extra_args=$*"
    echo
    env | sort
} > "${save_path}/run_env.txt"

python - <<'PY' > "${save_path}/env_versions.txt"
import sys
print("python:", sys.version)

mods = [
    "torch",
    "transformers",
    "datasets",
    "wandb",
    "verl",
    "qwen_vl_utils",
    "peft",
    "accelerate",
    "flash_attn",
]

for name in mods:
    try:
        m = __import__(name)
        print(f"{name}:", getattr(m, "__version__", "unknown"), getattr(m, "__file__", ""))
    except Exception as e:
        print(f"{name}: IMPORT_ERROR {repr(e)}")

try:
    import torch
    print("torch:", torch.__version__)
    print("torch.cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu_{i}:", torch.cuda.get_device_name(i))
except Exception as e:
    print("torch_cuda_report_error:", repr(e))
PY

# ============================================================
# Build args
# ============================================================

engine_args=(
    "engine=${BACKEND}"
    "optim=${BACKEND}"
    "engine.strategy=${FSDP_STRATEGY}"
    "engine.fsdp_size=${FSDP_SIZE}"
    "engine.ulysses_sequence_parallel_size=${SP_SIZE}"
    "engine.seed=${SEED}"
    "engine.full_determinism=${FULL_DETERMINISM}"
    "engine.use_torch_compile=${USE_TORCH_COMPILE}"
)

optim_args=(
    "optim.lr=${LR}"
    "optim.weight_decay=${WEIGHT_DECAY}"
    "optim.betas=${BETAS}"
    "optim.clip_grad=${CLIP_GRAD}"
    "optim.lr_scheduler_type=${LR_SCHEDULER_TYPE}"
    "optim.lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO}"
    "optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
    "optim.min_lr_ratio=${MIN_LR_RATIO}"
)

data_args=(
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${VAL_FILE}"
    "data.messages_key=messages"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_length=${MAX_LENGTH}"
    "data.pad_mode=${PAD_MODE}"
    "data.truncation=${TRUNCATION}"
    "data.num_workers=${NUM_WORKERS}"
    "data.use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
)

if [ "${USE_DYNAMIC_BSZ}" = "True" ]; then
    data_args+=(
        "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    )
else
    data_args+=(
        "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
    )
fi

model_args=(
    "model.path=${MODEL_PATH}"
    "model.tokenizer_path=${MODEL_PATH}"
    "model.trust_remote_code=True"
    "model.use_remove_padding=${USE_REMOVE_PADDING}"
    "++model.override_config.attn_implementation=${ATTN_IMPL}"
    "++model.override_config._attn_implementation=${ATTN_IMPL}"
)

peft_args=()
if [ "${USE_PEFT}" = "1" ]; then
    peft_args+=(
        "model.lora_rank=${LORA_RANK}"
        "model.lora_alpha=${LORA_ALPHA}"
        "model.target_modules=${LORA_TARGETS}"
    )
fi

trainer_args=(
    "trainer.n_gpus_per_node=${nproc_per_node}"
    "trainer.seed=${SEED}"
    "trainer.default_local_dir=${save_path}/checkpoints"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    'trainer.logger=["console","wandb"]'
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.max_ckpt_to_keep=${MAX_CKPT_TO_KEEP}"
    "trainer.resume_mode=${RESUME_MODE}"
    "hydra.run.dir=${save_path}/hydra"
    "hydra.output_subdir=.hydra"
)

checkpoint_args=(
    "checkpoint.save_contents=[model,optimizer,extra,hf_model]"
)

# ============================================================
# Launch
# ============================================================

torchrun --standalone --nnodes=1 --nproc-per-node="${nproc_per_node}" \
    -m verl.trainer.sft_trainer \
    "${data_args[@]}" \
    "${model_args[@]}" \
    "${engine_args[@]}" \
    "${optim_args[@]}" \
    "${trainer_args[@]}" \
    "${checkpoint_args[@]}" \
    "${peft_args[@]}" \
    "$@"

# CUDA_VISIBLE_DEVICES=4,5 \
# HF_HOME=/mnt/shared/shared_hf_home \
# MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
# DATA_DIR=/mnt/data1/eunwooim/rlpt/data/scienceqa_sft \
# TRAIN_BATCH_SIZE=256 \
# MAX_LENGTH=4096 \
# USE_DYNAMIC_BSZ=True \
# MAX_TOKEN_LEN_PER_GPU=65536 \
# PAD_MODE=no_padding \
# TRUNCATION=error \
# NUM_WORKERS=2 \
# ATTN_IMPL=flash_attention_2 \
# USE_REMOVE_PADDING=True \
# USE_TORCH_COMPILE=False \
# LR=1e-5 \
# WEIGHT_DECAY=0.01 \
# TOTAL_EPOCHS=15 \
# SAVE_FREQ=50 \
# TEST_FREQ=25 \
# USE_PEFT=1 \
# LORA_RANK=32 \
# LORA_ALPHA=64 \
# PROJECT_NAME=rlpt-scienceqa-sft \
# EXPERIMENT_NAME=qwen25vl3b-sft-baseline-lora32-ep15 \
# bash train/run_scienceqa_qwen25vl3b_sft_fsdp.sh \
#   2 /mnt/data1/eunwooim/rlpt/outputs/scienceqa_sft_baseline_lora32_ep15 \
#   2>&1 | tee /mnt/data1/eunwooim/rlpt/outputs/scienceqa_sft_baseline_lora32_ep15/train.log
