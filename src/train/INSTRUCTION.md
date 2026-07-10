# Instructions: Training Scripts

## Goal

Training entry points under `src/train/` are:

1. `src/train/sft.py` for supervised fine-tuning (SFT)
2. `src/train/rlvr_grpo.py` for RLVR / GRPO-style training with process rewards
3. `src/train/baseline_rewards.py` for veRL-compatible baseline rule rewards

This folder is for training only. Do not mix training logic with reliability-test code.

Current priority: test baselines first. Add command templates and documentation before implementing new method rewards.

## Scope

- Use FlashAttention when available and compatible.
- Use veRL for RLVR / GRPO training.
- Keep SFT and RL implementations separate.
- Support multiple experiment configurations, ablations, and reproducibility runs.
- Add reusable bash automation under `src/scripts/` when it reduces manual repetition.
- For the first pass, do not implement `ours`.
- For baseline RLVR, use only accuracy and format rewards.
- Do not run long training jobs while adding templates or docs.

## Baseline-First Controls

The first SFT-vs-RLVR comparisons must keep these fixed:

```text
BASE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
DATA_RATIO=0.01
MAX_SEQ_LEN=4096
SEED=42
TOKEN_BUDGET=1000000
DATASET_SPLIT=train
```

Use the datasets separately for the first run:

```text
visualprm -> OpenGVLab/VisualPRM400K-v1.1-Raw
llava_cot -> Xkev/LLaVA-CoT-100k
```

Use `--data_ratio` to deterministically control training dataset scale. The SFT and RLVR commands in a fair comparison must use the same base model, dataset alias, split, `data_ratio`, sequence length, seed, and token budget.

For `OpenGVLab/VisualPRM400K-v1.1-Raw`, do not use the default Hugging Face JSON builder for training rows. It can fail because raw rows include an `analysis` column missing from the declared feature schema. Load `annotations.zip` directly, deterministically sample rows, and show `tqdm` progress for annotation counting and row selection. For SFT, resolve `images.zip` once and lazily open images from the zip during collation instead of extracting all selected images before training. In `IMAGE_MODE=auto`, drop missing VisualPRM archive members and continue; use `IMAGE_MODE=strict` to fail on missing image references.

For `Xkev/LLaVA-CoT-100k`, do not pass raw relative image paths directly to the processor. In `IMAGE_MODE=auto`, drop image paths and run text-only unless `IMAGE_ZIP` or `LLAVA_COT_IMAGE_ZIP` points to an assembled image archive. In `IMAGE_MODE=strict`, fail clearly if a referenced image is missing.

## Shared Conventions

All training scripts must:

- read repository-relative paths
- write timestamped outputs
- avoid overwriting existing runs unless `--force` is passed
- save config, logs, and metrics with each run
- be resumable when possible
- support a dry-run or config-only mode if practical
- pass fairness controls explicitly rather than relying on hidden defaults
- make output locations obvious before launch

## SFT

Maintain a dedicated SFT script. It should:

- train with supervised loss only
- ignore `--process_reward`
- support standard training hyperparameters
- support FlashAttention
- support checkpoint saving and evaluation hooks
- support `--config_only` for lightweight validation

SFT is not RL. Do not reuse RL-specific reward logic inside SFT.

Baseline command templates must not pass any process-reward flags to SFT.

## RLVR / GRPO

Maintain a separate RL script for process-reward training. It should:

- support veRL-based execution
- support FlashAttention
- accept `--process_reward`
- keep reward construction modular
- prepare veRL parquet files from deterministic dataset subsets
- support `--config_only` and `--prepare_data_only` for lightweight validation

The `--process_reward` flag must support at least:

- `nli_heuristic`
- `ours`
- `rlvr`
- `sft`
- `none`

If a reward type is not applicable to the selected training mode, fail clearly.

Baseline command templates should use:

```text
--process_reward rlvr
--reward_components accuracy,format
```

Do not enable `ours` in the baseline scripts.

`src/train/baseline_rewards.py` is limited to:

```text
accuracy
format
```

Do not add calibrated compatibility rewards or reasoning-segment reward modeling there until explicitly requested.

## Process Reward

Process reward selection must be explicit and configurable.

For RL runs:

- `nli_heuristic` means the hand-designed NLI score
- `ours` means the calibrated compatibility reward
- `rlvr` means the baseline RLVR reward setup
- `sft` may be used only as a reference baseline or diagnostic, not as a true process reward inside SFT training
- `none` means no process reward

The reward implementation should be modular so new reward functions can be added later.

## Experiment Automation

Place repeatable bash helpers in `src/scripts/`.

Use them for:

- running multiple configurations
- hyperparameter sweeps
- consistency checks
- reproducibility runs
- ablations

Scripts should:

- accept explicit run names
- write logs
- avoid overwriting outputs
- reuse config files when possible
- default to `DRY_RUN=1`
- print exact commands in dry-run mode
- avoid dataset downloads and GPU work during dry-run checks

Expected reusable helpers:

```text
src/scripts/_train_common.bash
src/scripts/run_baseline.bash
src/scripts/run_first_baselines.bash
src/scripts/sweep_baselines.bash
src/scripts/ablate_baselines.bash
```

Training entrypoints may be configured with:

```text
SFT_ENTRYPOINT=src/train/sft.py
RL_ENTRYPOINT=src/train/rlvr_grpo.py
```

The bash templates may pass through:

```text
CONFIG_ONLY=1
PREPARE_DATA_ONLY=1
MAX_TRAIN_SAMPLES=<int>
BATCH_SIZE=<int>
GRAD_ACCUM_STEPS=<int>
REPORT_TO=wandb
WANDB_MODE=offline|online
IMAGE_MODE=auto|text_only|strict
IMAGE_ZIP=<path>
```

W&B setup should remain opt-in and metadata-oriented until the experiment metrics are specified.

## Output Layout

Use timestamped outputs, for example:

```text
src/outputs/train/<experiment_name>/run_YYYYMMDD_HHMMSS/
  configs/
  logs/
  checkpoints/
  metrics/
```

Dry-runs may print these paths without creating them. Real runs should save the command and run metadata under `logs/` and `configs/`.

## Lightweight Validation

For script and documentation changes, run only lightweight checks:

```bash
bash -n src/scripts/*.bash
python -m py_compile src/train/train_common.py src/train/baseline_rewards.py src/train/sft.py src/train/rlvr_grpo.py
python src/train/sft.py --help
python src/train/rlvr_grpo.py --help
DRY_RUN=1 bash src/scripts/run_first_baselines.bash
DRY_RUN=1 MODE=sft DATASET=visualprm bash src/scripts/run_baseline.bash
```

Do not run `DRY_RUN=0`, start GPU jobs, or download datasets unless explicitly instructed.
