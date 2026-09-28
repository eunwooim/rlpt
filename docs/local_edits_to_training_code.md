# Local Edits to Supervisor's Training Code

Running log of every change we make to code from `origin/main` (commit `545a4d0`,
"sft rlvr implement") while carrying out the baseline runs on Sol. Each entry says what
was wrong, why the fix was needed, and how to reproduce/revert. Nothing here is
committed without explicit sign-off; the working-tree diff is the source of truth.

---

## Edit 1 — `src/train/sft.py`: bound image pixels in `VLMDataCollator._load_image`

**Date:** 2026-07-12 · **Run affected:** `visualprm_sft_first_baseline` (job 58943318)

**Symptom:** SFT crashed at step 43/245 with

```
ValueError: Mismatch in `image` token count between text and `input_ids`.
Got ids=[4081] and text=[6696]. Likely due to `truncation='max_length'`.
```

**Root cause:** the collator tokenizes with `truncation=True, max_length=4096`
(the documented `MAX_SEQ_LEN` control), but images are passed to the Qwen2.5-VL
processor at full resolution with no size cap anywhere in the SFT pipeline. The
processor spends one sequence token per 28×28-pixel block, so a large VisualPRM image
can expand to more tokens than the whole sequence budget (the step-43 sample needed
6,696 image tokens alone). Truncation then cuts *inside* the image-token block, the
processor's consistency check between `<|image_pad|>` counts in text vs input_ids
fails, and the Trainer dies mid-run. Any sufficiently large image in the subset
reproduces it.

**Fix:** added `MAX_IMAGE_PIXELS` (env `SFT_MAX_IMAGE_PIXELS`, default
`1280 * 28 * 28` ≈ 1.0 MP ≈ 1,280 image tokens) and a `_bound_pixels()` helper that
downscales an image only when it exceeds the budget, applied in both `_load_image`
branches (filesystem path and `zip://` member). Images under the cap are untouched.
With the cap, the image block always fits inside `max_length=4096` and truncation only
ever trims text tails, which the processor accepts.

**Why this fix and not another:**
- Raising `MAX_SEQ_LEN` would change a fixed fairness control and still not bound the
  worst-case image.
- Disabling truncation risks OOM and breaks the token-budget accounting.
- Skipping failing samples would silently change the deterministic subset that SFT and
  RLVR must share.
- Parity: veRL's own data pipeline already resizes images to a max-pixel budget, so the
  RLVR arm effectively has this cap; SFT was the only arm missing it.

**Status:** applied locally, uncommitted. Raise with supervisor for upstreaming.
**Verified:** rerun job 58943680 passed the previously-crashing step 43 and completed all
245/245 steps (final state COMPLETED, train_loss mean 3.76, ~2.3 plateau).

---

## Edit 2 — `src/train/train_common.py`: nlvr2 candidate in `resolve_zip_member`

**Date:** 2026-07-12 · **Runs affected:** `visualprm_rlvr_first_baseline` (job 58943319,
crashed), `visualprm_sft_first_baseline` (job 58943680, silent data loss)

**Symptom (RLVR):** data preparation crashed with

```
FileNotFoundError: VisualPRM image not found in images.zip: images/train-1002-2-img0.png
```

**Symptom (SFT):** same resolution failure, but `IMAGE_MODE=auto` silently dropped the
images instead — "Dropped 364 VisualPRM image paths" (6.3% of the 5,747 selected
images), so those rows trained text-only.

**Root cause:** nlvr2 annotation rows reference images in a short style
(`images/train-*.png`) while the archive nests them one level deeper
(`images/nlvr2/images/train-*.png`). `resolve_zip_member` handled the standard
`VisualPRM400K-v1.1-Raw/...` → `images/...` mapping but not the nlvr2 nesting. We had
previously verified (full 565,149-row scan, 2026-07-11) that adding the
`images/nlvr2/<ref>` candidate yields 100% resolution with zero misses.

**Fix:** one extra candidate in the VisualPRM branch of `resolve_zip_member`: when the
reference already starts with `images/`, also try `"images/nlvr2/" + image_name`.
Both the RLVR materialize path and the SFT lazy-attach path go through this function.

**Consequence:** the earlier SFT run (job 58943680) trained with 364 rows silently
text-only, so SFT was rerun after this fix for data parity with RLVR.

**Status:** applied locally, uncommitted. Raise with supervisor for upstreaming.
**Verified:** unit-checked against the exact failing reference plus the prefixed-nlvr2
and MAVIS styles (all resolve); SFT rerun (job 58944776) resolved 5,747/5,747 selected
images with zero drops; RLVR rerun (job 58944777) materialized all 5,747 images without
the FileNotFoundError.

---

## Edit 3 — `src/train/rlvr_grpo.py`: wrong Hydra key for gradient checkpointing

**Date:** 2026-07-12 · **Run affected:** `visualprm_rlvr_first_baseline` (job 58944777)

**Symptom:** veRL exited immediately after launch:

```
Could not override 'actor_rollout_ref.actor.gradient_checkpointing'.
Key 'gradient_checkpointing' is not in struct
```

**Root cause:** `build_verl_command` emitted
`actor_rollout_ref.actor.gradient_checkpointing=True`, but verl's config schema has no
such key under `actor` — the setting is `actor_rollout_ref.model.enable_gradient_checkpointing`
(verl 0.8.0, and the same in earlier versions). The RLVR command path had never been
executed for real (all templates default `DRY_RUN=1`), so this never surfaced upstream.

**Fix:** changed the override to
`actor_rollout_ref.model.enable_gradient_checkpointing={args.gradient_checkpointing}`
(`rlvr_grpo.py:225`). Then validated the ENTIRE override list offline with Hydra compose
(`python -m verl.trainer.main_ppo <all overrides> --cfg job`) — the full config now
composes with no invalid keys, so there are no further schema mismatches waiting.

**Status:** applied locally, uncommitted. Raise with supervisor for upstreaming.

---

## Edit 4 — `src/train/train_common.py`: keep all images in `materialize_visualprm_images`

**Date:** 2026-07-12 · **Run affected:** `visualprm_rlvr_first_baseline` (job 58945116)

**Symptom:** veRL training crashed in the dataloader:

```
AssertionError: image_offset 1 >= len(images) 1
  (verl/utils/dataset/rl_dataset.py:331, _build_messages)
```

**Root cause:** `materialize_visualprm_images` ended with
`row["image"] = materialized_images[0]; row.pop("images", None)` — discarding all but
the first image. nlvr2 rows carry TWO images and two `<image>` placeholders in the
prompt, so verl's message builder ran out of images at the second placeholder. (Before
Edit 2 these rows never got this far — their images failed to resolve — which is why
the bug only surfaced now.) The SFT-side `attach_visualprm_image_zip` already keeps all
images; only the RLVR materialize path truncated to one.

**Fix:** mirror the attach-path idiom — `row["image"] = first`, and when more than one,
`row["images"] = all` (train_common.py:365-372).

**Status:** applied locally, uncommitted. Raise with supervisor for upstreaming.

---

## Edit 5 — `src/train/train_common.py`: bound image pixels in `materialize_visualprm_images`

**Date:** 2026-07-12 · **Run affected:** `visualprm_rlvr_first_baseline` (job 58945418)

**Symptom:** veRL crashed during step-0 validation rollout generation:

```
ValueError: Multimodal prompt produced 2206 tokens, exceeding rollout.prompt_length=2048.
Truncating multimodal token sequences corrupts vision/audio feature alignment...
Reduce the multimodal input size (e.g. ``max_pixels``) or increase ``rollout.prompt_length``.
```

**Root cause:** the RLVR twin of Edit 1. Images are materialized to disk at full
resolution; Qwen2.5-VL spends one prompt token per 28×28-pixel block, so a large image
alone can exceed `max_prompt_length=2048`, and veRL (correctly) refuses to truncate
inside multimodal token blocks. verl 0.8 exposes no pixel-cap config in its dataset,
and raising `max_prompt_length` would break the `MAX_SEQ_LEN=4096` fairness control
(prompt 2048 + response 2048).

**Fix:** `write_bounded_image()` applied at extraction time in
`materialize_visualprm_images`: images over a pixel budget are downscaled before being
written; smaller images are copied byte-identical. Budget env `RLVR_MAX_IMAGE_PIXELS`,
default `640 * 28 * 28` (≈0.5 MP ≈ 640 image tokens), chosen so a two-image nlvr2 row
(1,280 tokens) still leaves ~768 tokens for question text within the 2048 prompt budget.

**Note for supervisor:** this makes the RLVR image cap (640 tokens) tighter than SFT's
(1,280 tokens, Edit 1) because RLVR's prompt budget is 2048 vs SFT's 4096 joint budget.
If image-content parity between arms matters for the comparison, consider setting
`SFT_MAX_IMAGE_PIXELS=501760` on the SFT side too and rerunning.

**Status:** applied locally, uncommitted. Raise with supervisor for upstreaming.

---

## Edit 6 — `src/train/train_common.py`: filtered-jsonl dataset source (opt-in via env)

**Date:** 2026-07-18 · **Runs affected:** `visualprm_{sft,rlvr}_filtered_tau085`

**Why:** the controlled filtered-vs-unfiltered comparison needs the training pool
restricted to the two-stage-filtered traces
(`data/visualprm_v11_filtered/{sft_train,rlvr_prompts}.jsonl`) without changing
anything else about the pipeline. The filtered files carry only
`question_orig`/`steps` — NOT the `question` prompt template or the full
`response` the unfiltered runs trained on — so feeding them directly would
change prompt/response text, not just data quality.

**Change:** added `FilteredVisualPRMJsonlSource` (subclass of
`VisualPRMZipSource`, so `scaled_dataset`'s seeded-sample branch applies
unchanged) plus `load_filtered_visualprm_jsonl_source`, and a 3-line opt-in
hook in `load_hf_dataset`: when env `RLPT_FILTERED_JSONL` is set (and the
dataset is VisualPRM), the filtered jsonl supplies the (source_file,
line_index) trace-id pool and every selected record is re-read IN FULL from
`data/visualprm_v11_raw/annos/` — records are byte-identical to the
unfiltered loading path (verified: 8/8 sampled records `==` raw). With the env
var unset, behavior is bit-for-bit the original (hook is a no-op), so existing
baselines are unaffected. `RLPT_RAW_ANNOS_ROOT` overrides the raw dump
location if ever needed.

**Status:** applied locally, uncommitted. Raise with supervisor for upstreaming.
**Verified:** unit test (pool sizes 100,194 / 73,704; seed-42 sample-order
parity with the zip-source formula; record equality vs raw) + 8-row
`PREPARE_DATA_ONLY` smoke on both arms.

---

## Env fix A — `VLLM_USE_V1=1` in our RLVR sbatch (no code change)

**Date:** 2026-07-12 · **Run affected:** `visualprm_rlvr_first_baseline` (job 58944986)

**Symptom:** veRL's rollout server died during startup:

```
ValueError: Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. This should not happen.
```

**Root cause:** verl 0.8.0's `vLLMHttpServer` is an async Ray actor, so it builds the
vLLM engine config off the main thread. vLLM 0.10's V1-eligibility oracle treats
"Engine in background thread" as experimental and silently falls back to V0 —
*unless* `VLLM_USE_V1=1` is set explicitly, in which case it proceeds with a warning.
verl then unconditionally instantiates the V1 `AsyncLLM` class, which errors on the
mismatch. verl's own `main_generation_server.py` sets `VLLM_USE_V1=1` for exactly this
reason; `main_ppo` does not.

**Fix:** `export VLLM_USE_V1=1` in `run_baseline_rlvr.sbatch` (Ray workers inherit it
from the pre-started raylet; verl's runtime_env does not override it). Supervisor's
code untouched.

---

## Run ledger (which jobs used which code state)

| Job | Arm | Code state | Outcome |
|---|---|---|---|
| 58943318 | SFT | pristine 545a4d0 | CRASHED step 43/245 (image-token overflow → Edit 1) |
| 58943319 | RLVR | pristine 545a4d0 | CRASHED in data prep (nlvr2 image paths → Edit 2) |
| 58943680 | SFT | + Edit 1 | COMPLETED 245/245, but 364/5,747 images silently dropped (superseded) |
| 58944776 | SFT | + Edits 1,2 | **COMPLETED — canonical SFT baseline.** 245/245 steps, 0 dropped images, train_loss mean 3.82 (8.24 → 2.23 by 35-step buckets), 636 s. Output: `src/outputs/train/visualprm_sft_first_baseline/run_20260712_045241` |
| 58944777 | RLVR | + Edits 1,2 | CRASHED at veRL launch (bad Hydra key → Edit 3); data prep (5,747 images + parquet) succeeded |
| 58944986 | RLVR | + Edits 1,2,3 | CRASHED at vLLM engine start (V0/V1 mismatch → Env fix A); config composed and workers launched |
| 58945116 | RLVR | + Edits 1,2,3 + Env fix A | CRASHED in dataloader (multi-image rows truncated to 1 → Edit 4); vLLM V1 engines started OK |
| 58945418 | RLVR | + Edits 1-4 + Env fix A | CRASHED in step-0 validation rollout (multimodal prompt 2206 > 2048 → Edit 5); dataloader passed |
| 58945708 | RLVR | + Edits 1-5 + Env fix A | **COMPLETED — canonical RLVR baseline.** 4/4 GRPO steps, 21:44 wall. Val reward 0.5376 → 0.5553. Train rewards quantized to {0.25, 0.75} (format term ≡ 0.5, accuracy ∈ {0,1} — the degenerate-accuracy signature). Entropy 0.80→0.67, KL loss 0.014→0.088. Checkpoint: `run_20260712_060302/checkpoints/global_step_4`. Post-completion teardown noise in log (DataLoader worker reaped, vLLM monitor-thread race) — harmless, after final metrics + checkpoint. |

---

## Edit 7 — scale-run plumbing + opt-in resume (2026-07-18, for the real DATA_RATIO=1 runs)

**Why:** the scale-up runs need (a) GRPO knobs that `run_baseline.bash` did not
plumb (`rollout_n`, `ppo_mini_batch_size`, `ppo_micro_batch_size_per_gpu`,
`save_freq`, `test_freq` — all already argparse args in `rlvr_grpo.py`),
(b) SFT `--save_steps` plumbing, and (c) resumability across wall-clock kills.

**Changes (no behaviour change unless the new env vars are set):**

1. `src/scripts/run_baseline.bash` — new optional env→CLI pass-throughs:
   RLVR branch: `ROLLOUT_N`, `PPO_MINI_BATCH_SIZE`, `PPO_MICRO_BATCH_SIZE_PER_GPU`,
   `SAVE_FREQ`, `TEST_FREQ`; SFT branch: `SAVE_STEPS`. Plus
   `OUTPUT_DIR_OVERRIDE` — pins the run directory instead of minting a fresh
   `run_<timestamp>`, so a resubmitted RLVR job lands in the same dir and verl's
   default `trainer.resume_mode=auto` picks up `latest_checkpointed_iteration.txt`
   (use together with `FORCE=1`; `ensure_output_layout(force=True)` never deletes).
2. `src/train/sft.py` — `RLPT_RESUME=1` env gate: calls
   `transformers.trainer_utils.get_last_checkpoint(checkpoints/)` and passes it to
   `trainer.train(resume_from_checkpoint=...)`. Unset → `None` → identical to before.

**Status:** uncommitted local edit, layered on Edits 1–6.

**Addendum:** also plumbed `VERL_EXTRA_OVERRIDES` (space-separated Hydra overrides →
repeated `--verl_extra_override`, an escape hatch `rlvr_grpo.py` already had). Used
for `trainer.max_actor_ckpt_to_keep=2` in the scale runs (44 GB per FSDP checkpoint).
