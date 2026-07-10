# Training Baselines

This directory contains training entry points and training-specific notes. Reusable launch automation lives in `src/scripts/`.

The current training milestone is baseline validation only:

1. SFT on a small deterministic subset.
2. RLVR / GRPO on the same subset, with only accuracy and format rewards.
3. Later method work, including `ours`, remains out of scope for this pass.

## Fixed First-Run Controls

Use the same controls for fair SFT-vs-RL comparisons:

```text
BASE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
DATA_RATIO=0.01
MAX_SEQ_LEN=4096
SEED=42
TOKEN_BUDGET=1000000
DATASET_SPLIT=train
```

The first baselines use the datasets separately:

```text
visualprm  -> OpenGVLab/VisualPRM400K-v1.1-Raw
llava_cot  -> Xkev/LLaVA-CoT-100k
```

`--data_ratio` is the deterministic scale control. Keep it identical across SFT and RLVR when comparing methods.

## Command Flow

All scripts default to dry-run mode:

```bash
DRY_RUN=1 bash src/scripts/run_first_baselines.bash
```

This prints the exact commands without launching training, downloading datasets, or starting GPU work.

Run one command template:

```bash
DRY_RUN=1 MODE=sft DATASET=visualprm bash src/scripts/run_baseline.bash
DRY_RUN=1 MODE=rlvr DATASET=visualprm bash src/scripts/run_baseline.bash
```

Run the first baseline matrix:

```bash
DRY_RUN=1 bash src/scripts/run_first_baselines.bash
```

Run a small sweep template:

```bash
DRY_RUN=1 \
DATA_RATIOS="0.01 0.02" \
SEEDS="42 43" \
LEARNING_RATES="1e-5 2e-5" \
bash src/scripts/sweep_baselines.bash
```

Run RLVR reward ablation templates:

```bash
DRY_RUN=1 bash src/scripts/ablate_baselines.bash
```

The Python entrypoints support config-only checks:

```bash
CONFIG_ONLY=1 DRY_RUN=0 FORCE=1 MODE=sft DATASET=visualprm EXPERIMENT_NAME=config_check_sft bash src/scripts/run_baseline.bash
CONFIG_ONLY=1 DRY_RUN=0 FORCE=1 MODE=rlvr DATASET=visualprm EXPERIMENT_NAME=config_check_rlvr bash src/scripts/run_baseline.bash
```

Switch to execution explicitly:

```bash
DRY_RUN=0 MODE=sft DATASET=visualprm bash src/scripts/run_baseline.bash
DRY_RUN=0 MODE=rlvr DATASET=visualprm bash src/scripts/run_baseline.bash
```

Common overrides:

```bash
DRY_RUN=0 MODE=sft DATASET=visualprm BATCH_SIZE=1 GRAD_ACCUM_STEPS=16 bash src/scripts/run_baseline.bash
DRY_RUN=0 MODE=rlvr DATASET=visualprm BATCH_SIZE=32 bash src/scripts/run_baseline.bash
```

Bound a real smoke run while still selecting the requested ratio:

```bash
CUDA_VISIBLE_DEVICES=6 REPORT_TO=wandb WANDB_MODE=offline DRY_RUN=0 FORCE=1 MODE=sft DATASET=visualprm DATA_RATIO=0.1 BATCH_SIZE=1 GRAD_ACCUM_STEPS=1 MAX_STEPS=1 EXPERIMENT_NAME=smoke_visualprm_0p1 bash src/scripts/run_baseline.bash
```

Debug data preparation without loading the model:

```bash
DRY_RUN=0 PREPARE_DATA_ONLY=1 MAX_TRAIN_SAMPLES=8 MODE=sft DATASET=visualprm bash src/scripts/run_baseline.bash
DRY_RUN=0 PREPARE_DATA_ONLY=1 MAX_TRAIN_SAMPLES=8 MODE=sft DATASET=llava_cot bash src/scripts/run_baseline.bash
```

Enable W&B plumbing without deciding custom metrics yet:

```bash
REPORT_TO=wandb WANDB_MODE=offline DRY_RUN=0 MODE=sft DATASET=visualprm bash src/scripts/run_baseline.bash
```

## Output Locations

Real runs write timestamped outputs:

```text
src/outputs/train/<experiment_name>/run_YYYYMMDD_HHMMSS/
  configs/
  logs/
  checkpoints/
  metrics/
```

The launch command is saved to:

```text
logs/command.txt
```

Run metadata is saved to:

```text
configs/run_metadata.env
```

Dry-runs print the same output paths but do not create the directories.

## Entry Point Contract

Implemented entrypoints:

```text
src/train/sft.py
src/train/rlvr_grpo.py
src/train/baseline_rewards.py
```

The bash templates pass these shared arguments to both SFT and RLVR:

```text
--base_model
--dataset_name
--dataset_split
--data_ratio
--max_seq_len
--seed
--token_budget
--learning_rate
--output_dir
```

VisualPRM also receives:

```text
--dataset_config default
```

RLVR additionally receives:

```text
--rl_algorithm grpo
--process_reward rlvr
--reward_components accuracy,format
--reward_ablation accuracy_format
```

Ablations are limited to:

```text
accuracy_only
format_only
accuracy_format
```

Do not pass `ours` in these baseline scripts.

### SFT Entry Point

`src/train/sft.py` uses Hugging Face `datasets`, `transformers`, and `Trainer`. It applies deterministic `data_ratio` selection, saves configs and metrics under the run directory, and can optionally use LoRA with `--use_lora`.

For `OpenGVLab/VisualPRM400K-v1.1-Raw`, the entrypoint reads `annotations.zip` directly instead of `datasets.load_dataset(...)`, because the Hub JSON schema currently omits an `analysis` column that appears in raw rows. The VisualPRM annotation count and deterministic row selection show `tqdm` progress bars. SFT resolves `images.zip` once and lazily opens selected images from the archive during collation, mapping annotation paths like `VisualPRM400K-v1.1-Raw/...` to archive paths under `images/...`. In `IMAGE_MODE=auto`, missing archive members are dropped and those examples fall back to text-only if no resolved image remains; use `IMAGE_MODE=strict` when auditing dataset integrity.

For `Xkev/LLaVA-CoT-100k`, the dataset ships images as very large split zip parts. By default `IMAGE_MODE=auto` drops relative image paths and runs text-only SFT rather than passing invalid paths like `ai2d/images/968.png` to the processor. To run multimodal LLaVA-CoT SFT, assemble/provide an image zip and pass:

```bash
IMAGE_ZIP=/path/to/llava_cot_images.zip IMAGE_MODE=strict DRY_RUN=0 MODE=sft DATASET=llava_cot bash src/scripts/run_baseline.bash
```

W&B is opt-in through `REPORT_TO=wandb`. By default `WANDB_MODE=offline` unless overridden.

### RLVR / GRPO Entry Point

`src/train/rlvr_grpo.py` prepares veRL parquet files with:

```text
data_source
prompt
ability
reward_model.ground_truth
extra_info
```

It points veRL at `src/train/baseline_rewards.py` through `custom_reward_function.path`. The custom reward implements only the baseline accuracy and format rewards. Use `--prepare_data_only` to write parquet and the veRL command without launching training.

## Lightweight Checks

Use only lightweight checks for this pass:

```bash
bash -n src/scripts/*.bash
python -m py_compile src/train/train_common.py src/train/baseline_rewards.py src/train/sft.py src/train/rlvr_grpo.py
python src/train/sft.py --help
python src/train/rlvr_grpo.py --help
DRY_RUN=1 bash src/scripts/run_first_baselines.bash
DRY_RUN=1 MODE=sft DATASET=visualprm bash src/scripts/run_baseline.bash
```
