# RLPT — QCVSR GQA Dataset Project

Synthetic visual-reasoning dataset generation for RL post-training of multimodal models.
The schema is **QCVSR** (Query-Conditioned Visual Schema Reasoning): the model must emit a
typed evidence object (bbox pairs or count pairs) plus an answer, with a deterministic
verifier checking consistency.

---

## Repo layout

```
/scratch/sghos104/rlpt/
├── src/data_factory/
│   ├── gqa.py                  # Main generation pipeline (CLI: argparse)
│   ├── smoke_test.py           # Streaming validation (~10 records, no full download)
│   ├── generate_gqa.sbatch     # Full run SLURM job
│   └── generate_gqa_test.sbatch# Limited-image test SLURM job (max_images=2000)
├── tools/
│   ├── inspect_dataset.py      # 9 validity checks + visual sample grid
│   ├── validity_report.csv     # Per-record pass/fail (last run)
│   └── sample_grid.png         # 5x10 grid of records with bbox overlays
├── data/
│   ├── scene_graphs/           # GQA scene-graph JSON (uploaded manually)
│   ├── images/                 # Extracted JPGs (~72k, ~7 GB)
│   ├── hf_cache/               # HF datasets cache (~19 GB after first download)
│   ├── qcvsr_gqa_v1/           # Final dataset: train.jsonl + eval_*.jsonl
│   └── logs/                   # SLURM stdout/stderr
└── requirements.txt
```

The `src/verl/` submodule was removed from git but may be added back later for training.

---

## Environment

- **Cluster**: ASU Sol (`sol-login04`), login via `sghos104@sol.asu.edu` (Duo MFA).
- **Conda env**: `/scratch/sghos104/envs/rlpt-data/` — has `datasets`, `PIL`, `pyarrow`, etc.
- **Python binary**: `/scratch/sghos104/envs/rlpt-data/bin/python`
- **HF cache**: `HF_HOME=/scratch/sghos104/rlpt/data/hf_cache/` (set in sbatch + gqa.py)

### Cluster gotchas

- **`--partition=public`**, not `general`. The `general` partition is now private (privately
  owned nodes only); using it returns "Unspecified error" from sbatch. Cluster suggests
  `-p htc` for jobs under 4 hours (shorter queue wait).
- Login nodes are **not** for long runs — anything over a few minutes belongs in sbatch.

---

## Data pipeline (`gqa.py`)

**Zero generative AI.** Pure rule-based assembly from two inputs:

1. **GQA images** (`lmms-lab/GQA`, config `train_balanced_images`) — real photographs.
2. **GQA scene graphs** (uploaded manually; ~74,942 images annotated) — JSON with object
   names, bounding boxes, attributes per image.

For each image, emit candidates for six schemas. **Disambiguation rule:** a question is
only emitted when its constraints resolve to a single determinate answer the verifier can
reproduce. Schemas that localize *one* object require a **unique referent** (a class with
exactly one instance); count/compositional schemas *use* multiple instances as the signal.

| Schema | Operations | Multi-instance | Verifier examples |
|---|---|---|---|
| `bbox_pair_spatial_compare` | `left`, `right`, `higher`, `lower` | unique referents only | `compare_x_center_min/max`, `compare_y_center_min/max` |
| `bbox_pair_size_compare` | `larger`, `smaller` | unique referents only | `compare_bbox_area_max/min` |
| `count_pair_compare` | `more_than_yesno`, `difference` | required (counts two classes) | `compare_count_greater_than`, `count_difference` |
| `attr_select_spatial` | `color_left/right/higher/lower` | **requires 2 instances** (diff colors); spatial clause disambiguates | `select_x_center_min_color`, … (selected node must be the geometric extreme) |
| `count_relation` | `count_relation` | requires ≥2 subjects related to a *unique* ref | `count_related_subjects` (count == #subject nodes) |
| `relation_target` | `relation_target` | unique subject, one matching relation | `relation_target_name` (target node name == answer) |

Compositional examples: *"What color is the cup on the left?"* (2 cups, differing colors),
*"How many chairs are near the desk?"* → 2, *"What is the boy holding?"* → bat.

Bounding boxes are **xyxy normalized to 0–1000**.

### Splits (deterministic via `seed=42`)

- `train` (80% of images): ops `{left, right, larger, more_than_yesno, color_left,
  color_right, count_relation, relation_target}`. Relation schemas use the **SEEN**
  predicate sets (`COUNT_RELATIONS_SEEN`, `TARGET_RELATIONS_SEEN`).
- `eval_id` (7%): same ops/predicates as train, paraphrased templates from `TRAIN_TEMPLATES`.
- `eval_ood_template` (7%): same ops/predicates, distinctly worded `OOD_TEMPLATES`.
- `eval_ood_operation` (~6%): held-out ops `{higher, lower, smaller, difference,
  color_higher, color_lower}` **plus** the **UNSEEN** relation predicate sets for
  `count_relation` / `relation_target` (predicate holdout handled inside their generators).

### Key constants in `gqa.py`

- `TARGET_QUOTAS` — per-op record cap (currently 1500 for train ops, 10% for evals → ≥50 each).
  `count_relation` is the rarest type (strict unique-ref + ≥2-subject gate) and typically
  under-fills its cap.
- `MAX_RECORDS_PER_IMAGE_PER_OP = 2` — diversity cap; without it, a single object-dense
  image floods the quota (early run produced 4000 train records from only 68 images).
- `STUFF_OBJECTS` — name blacklist (sky, ground, wall, ...) to skip non-thing classes.
- `COLOR_ATTRIBUTES` — color vocabulary for `attr_select_spatial`.
- `COUNT_RELATIONS_SEEN/UNSEEN`, `TARGET_RELATIONS_SEEN/UNSEEN` — relation predicates split
  into train-visible vs OOD-operation-held-out sets.
- `IRREGULAR_PLURALS` / `INVARIANT_PLURALS` — used by `pluralize()` for count templates;
  without this, count questions read "Are there more mans than..." which is ungrammatical.
  (Sibilant endings `ss/x/z/ch/sh` → `+es`, so "wine glass" → "wine glasses".)
- `extract_objects` drops boxes that collapse to zero w/h after rounding to 0–1000.

### Record schema (one JSONL line)

```json
{
  "question_id": "qcvsr_gqa_v1_train_00000000",
  "data_source": "qcvsr_gqa_<schema>",
  "prompt": [{"role": "user", "content": "<image>\nQuestion: ...\nReturn exactly:\n<schema>...</schema>\n<evidence>{...}</evidence>\n<answer>...</answer>"}],
  "images": ["/scratch/sghos104/rlpt/data/images/<id>.jpg"],
  "reward_model": {
    "style": "rule",
    "ground_truth": {
      "target_schema": "...",
      "target_evidence": {"nodes": [...], "operation": "..."},
      "target_answer": "...",
      "verifier": "...",
      "reward_types": {...}
    }
  },
  "ability": "query_conditioned_visual_schema_reasoning",
  "extra_info": {"image_id", "question", "split", "operation", ...}
}
```

---

## How to run

```bash
# Quick local validation (streaming, ~10 records, no large downloads)
/scratch/sghos104/envs/rlpt-data/bin/python src/data_factory/smoke_test.py

# Smaller cluster test (2000 images, separate output dir, warms HF cache)
sbatch src/data_factory/generate_gqa_test.sbatch

# Full generation
sbatch src/data_factory/generate_gqa.sbatch

# Inspect a generated jsonl (writes CSV + sample_grid.png)
/scratch/sghos104/envs/rlpt-data/bin/python tools/inspect_dataset.py \
    --input data/qcvsr_gqa_v1/train.jsonl
```

`gqa.py` accepts `--max-images`, `--out-dir`, `--image-out-dir`, `--seed`,
`--checkpoint-every`.

---

## Validity checks (`tools/inspect_dataset.py`)

| Check | Purpose |
|---|---|
| `image_exists` | JPG present at the path |
| `image_loadable` | PIL can open + verify (catches truncation) |
| `bbox_in_range` | Coords in `[0, 1000]` |
| `bbox_valid_order` | `x1 < x2` and `y1 < y2` |
| `evidence_node_count` | Pairwise schemas: 2 `nodes`; `count_relation`: 1 reference + ≥2 subjects |
| `evidence_schema_match` | Schema name matches node fields/roles (`bbox`/`count`, `selected_id`, `subject`/`target`) |
| `answer_consistency` | Re-runs the verifier against evidence, confirms target_answer — **most important** |
| `pluralization` | Question text contains no naive bad plurals (`mans`, `feets`, ...) |
| `required_fields` | Top-level keys all present |

---

## Image-description RL reward (current direction, 2026-06)

The QCVSR dataset above is one track. The active research track: RL on a model's **free-text
description of an image**, scored against GQA ground-truth annotations via **BERT-based
bipartite graph matching**. Goal: test whether a multimodal model gets better at *analysing
images* when rewarded on the content of what it describes.

**Scene graphs and the caption→graph parser are DROPPED** (2026-06). The model no longer emits
a graph, and **no parser sits in the reward loop**. This is the key win: removing the parser
eliminates the biggest reward-hacking confound — RL can no longer raise reward by learning
*parser-friendly phrasing* instead of actually seeing better. ("Graph" below refers to the
bipartite *matching* structure, not a scene-graph representation.)

### Pipeline shape (decided)

- **Model output = a natural-language description** of the image (free text). No graph, no
  typed schema, no parser.
- **GT = GQA oracle annotations** (objects, attributes, relations, boxes), linearized to text
  facts (`"black cat"`, `"cat wearing hat"`) — no model needed, GT is already structured.
- **Reward = soft bipartite match** between description units and the GT fact set.

### Reward = BERT-based bipartite graph matching

Soft optimal-assignment matching (generalizes SPICE exact-match and greedy BERTScore;
robust to "man"≈"person", "couch"≈"sofa"):

1. **Two sets** — predicted: the description decomposed into phrases/clauses (or scored
   token-level à la BERTScore); GT: the linearized annotation facts.
2. **Similarity matrix** `S[i][j]` from BERT / SBERT embeddings (SBERT-cosine preferred at RL
   scale for throughput).
3. **Bipartite matching** — Hungarian (`scipy.optimize.linear_sum_assignment` on `-S`) for a
   one-to-one assignment between description units and GT facts, optional threshold `τ`.
   One-to-one (not greedy) blocks the "say the same thing five ways" exploit.
4. **Soft precision-dominant P/R/F1** from matched similarity mass `M`: `P=M/|pred|`,
   `R=M/|GT|`, `Fβ` with **β<1**; bounded recall as guardrail. Fine-grained (attribute/
   relation) credit only for **unique referents**; unannotated GT slot = *no-signal, not wrong*.

### Multimodality in the match (optional grounding)

Blend text similarity with image grounding when aligning a description span to a GT object:
`S[i][j] = w1·text(BERT/SBERT) + w2·IoU(box) + w3·CLIP(GT-crop, description-span)`

- **Box IoU / center distance** — instance disambiguation ("the cat on the left" → the right
  GT cat), reusing the QCVSR unique-referent idea geometrically.
- **Region-CLIP** (multimodal BERTScore ≈ region CLIPScore) — embed the GT box crop with CLIP
  image encoder, the description span with CLIP text encoder; grounds reward in pixels, the
  strongest defense against rephrasing exploits.
- **Geometric predicate checks** for spatial relations (`left of`, `above`) via existing
  `compare_x_center`/`compare_y_center` verifiers.
- **Scaling**: GT-side embeddings are fixed → precompute/cache; only the description side runs
  per step.

### Status of the parser track (legacy)

The flan-t5 caption→graph parser is **no longer in the reward loop**. Tools
`tools/caption_to_graph.py`, `tools/eval_parser_factual.py`, `tools/eval_parser_on_vg.py`,
`src/data_factory/bench_parser.sbatch` are **legacy** (built for the dropped scene-graph path).
The parser may still be reused to bootstrap SFT-format targets if needed. The supervisor's
earlier parser **scaling stress test is moot** under this direction unless the parser is reused
for SFT bootstrapping — confirm before investing in it.

### ScienceQA implementation (current, 2026-06)

Dataset pivoted to **ScienceQA** (`derek-thomas/ScienceQA`, cached in `data/hf_cache`):
12,726 train / 4,241 val / 4,241 test. **Only ~49% have an image**; the usable
image+solution subset is **5,678 / 1,922 / 1,836**. Each item = question + `choices` +
`answer` (int index) + free-text `lecture`/`solution`. Base model to finetune:
**Qwen2.5-VL-3B-Instruct** (committed; 3B for iteration speed).

Pipeline (model emits `<think>` reasoning + `<answer>` choice; reward = composite):

- `tools/clause_split.py` — **deterministic** clause splitter (spaCy dep-parse, NO LLM in
  the loop; splits boundaries, not truth). Turns free text → near-atomic claim chunks for
  both prediction and GT `solution`. Splits on advcl/ccomp/relcl/parataxis + subject-bearing
  conj; re-attaches relative-clause antecedents; `clause_deps` is the granularity knob. Only
  hardcoded *words* are `_LEAD_STRIP` (cosmetic edge-trim).
- `tools/graph_match_reward.py` — the reward.
  - `GraphMatchReward` — soft bipartite match: clause_split both sides → SBERT
    (`all-MiniLM-L6-v2`) cosine matrix → Hungarian one-to-one (`scipy`) → precision-dominant
    Fβ (β=0.5, τ=0.15). GT embeddings cached.
  - `CompositeReward` — `R = w_f·format + w_m·match + w_a·answer − w_p·pun` (1/2/5/1),
    STAR-R1-shaped (arxiv 2505.15804). `format`=valid `<think>`/`<answer>`; `answer`=hard MCQ
    exact-match (fixes SBERT's entity-swap/negation blindness — a fluent-but-wrong answer
    scored 0.41≈correct 0.63 with match-only; composite separates to 1.8 vs 7.3); `pun`=−1 per
    reasoning clause matching no GT fact (hallucination).
  - `compute_score(data_source, solution_str, ground_truth, extra_info)` — **verl drop-in**
    (cached singleton). `ground_truth = {answer, solution, choices}`.
- `src/data_factory/scienceqa_to_verl.py` (+ `.sbatch`) — ScienceQA → verl parquet:
  writes images to `data/scienceqa_images/`, builds the `<image>` prompt, packs
  `reward_model.ground_truth = {answer, solution, choices}`. Output `data/scienceqa_verl/`.
  Default keeps image+solution rows; model prompted to answer with the choice **TEXT** (not a
  letter — ScienceQA letters are positionally ambiguous; `answer_correct` matches text).
- `src/train/grpo_scienceqa_qwen2_5vl.sbatch` — GRPO launch (verl, vllm rollout, `rollout.n=5`
  = GRPO group = "attempts"; KL coef 0.001; `custom_reward_function`→`compute_score`).
- `src/train/setup_train_env.sh` + `requirements-train.txt` — build a **separate** `rlpt-train`
  env. The training stack is NOT installed in `rlpt-data` (no verl/vllm/ray/flash-attn;
  transformers 4.44 < 4.49 needed for Qwen2.5-VL). Training env needs the RL stack AND the
  reward deps (SBERT/spacy/scipy) since the reward runs in verl workers.

### Training env + GRPO bring-up (rlpt-train, 2026-06-06)

Built `/scratch/sghos104/envs/rlpt-train` via `src/train/build_rlpt_train.sbatch` (htc, 1 GPU).
**Working stack: verl 0.8.0 · vllm 0.10.1.1 · torch 2.7.1+cu126 · transformers 4.56.2** (NOT
"latest": the first latest-everything build hung verl's Ray workers). Cluster = ASU Sol, BeeGFS
`/scratch`, RHEL8 (glibc 2.28), nodes have internet; OOD VSCode shell = 1 CPU/no-GPU (don't run
there — use `sbatch`/`srun`; no GPU-login needed).

Smoke-test loop (`grpo_scienceqa_smoke.sbatch`, 64-row `smoke.parquet`, 1 GPU, fails in 1-2 min).
**Bring-up fix chain (all encoded in the sbatches):**
1. Ray hang (BeeGFS + 48 prestart workers) → `ray_kwargs.ray_init.num_cpus=12/16`.
2. Config → set `actor.ppo_micro_batch_size_per_gpu` + `rollout`/`ref.log_prob_micro_batch_size_per_gpu`.
3. `ROCR_VISIBLE_DEVICES` clash → `unset ROCR_VISIBLE_DEVICES` (Sol sets the AMD var).
4. verl 0.8.0 dropped sync rollout; async needs vllm≥0.9 → upgraded vllm 0.8.5→0.10.1.1 (V1 engine).
5. `VLLM_USE_V1=1` required for the async server.
6. flash-attn glibc: torch-2.7 wheels need glibc 2.32 (RHEL8 has 2.28) → **no flash-attn**;
   use SDPA: `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` +
   `use_remove_padding=False` (vLLM uses its own kernels for rollout).
7. rope conflict → pin `transformers==4.56.2` (5.x writes Qwen2.5-VL rope vllm can't parse).
8. images as path strings rejected → `scienceqa_to_verl.py` now embeds `{bytes,path}` dicts.
9. SBERT load in verl's reward worker → "cannot copy out of meta tensor" (verl leaves the
   reward worker in a meta/fast-init torch state, even on its own thread). **Fix: run SBERT
   in a SUBPROCESS** (`tools/sbert_embed_server.py`; `graph_match_reward.encode` talks to it
   over a pipe) — a fresh interpreter has clean torch state.
10. Training path needs `flash_attn.bert_padding` (verl's log-prob packing, used
    unconditionally) AND vLLM's ViT needs `flash_attn_varlen_func`. Real flash-attn won't load
    (glibc). **Fix: a shim `flash_attn` on PYTHONPATH** (`src/train/flash_attn_shim/`): real
    pure-torch `bert_padding` + SDPA-backed `flash_attn_interface` (numerically verified) +
    a dist-info so the version is seen. Set on PYTHONPATH in the train sbatches.
11. Host-RAM OOM (FSDP+vLLM+Ray+reward subproc) → `--mem` 80→200G (smoke) / 240G (full).
12. GPU OOM at the train→rollout `wake_up` handoff (FSDP + vLLM on one 80GB A100) → FSDP CPU
    offload: `actor.fsdp_config.param_offload=True` + `optimizer_offload=True` +
    `ref.fsdp_config.param_offload=True`, and `rollout.gpu_memory_utilization=0.4`.

**SMOKE PASSED 2026-06-08: full GRPO step runs end-to-end** (`step:1`: actor/pg_loss≈0.30,
grad_norm≈3.7, critic/score mean −1.28 / max 7.2 / min −28.6 = reward discriminates,
update_actor 41s, update_weights 2.3s, ~70/80GB GPU, 131/200GB CPU, 180 tok/s). vLLM
generation → compute_score → GRPO advantages → FSDP update → weight-sync all confirmed.
Always `PYTHONUNBUFFERED=1`; debug a live job with `srun --overlap --jobid=J py-spy dump --pid <pid>`.

**FULL 2×A100 RUN TRAINING 2026-06-08.** First full run OOM'd on HOST RAM at step 21 — but it
was NOT a leak: the footprint is a stable ~207GB at start → +50GB one-time (optimizer states
materialize on CPU via offload) → then only ~0.5GB/step; the 240GB `--mem` was just below the
~270GB steady state. Fix: `--mem=400G` (comfortable margin; 480G over-requested and queued
hours longer) + `save_freq=20` (early resumable checkpoints). verl already avoids accumulating
generations (`free_cache_engine=true`, `rollout_data_dir=null`, `log_val_generations=0`).
Relaunched run trains stably: **reward −2.0 → ~+5.9 then plateaus near the reward ceiling**
(format+answer learned; matches the smoke trend), cpu_mem ~273GB at step 22 (well under 400),
grad_norm ~3–6 (occasional transient spike to ~15, self-corrects), checkpoints every 20 steps.
8h time limit ⇒ ~1 epoch (~65–70 steps). NOTE: `critic/score` is dominated by the hard answer
term (w=5); to assess the *reasoning/image-analysis* signal, track the bipartite `match`
component separately. Job id persisted at `data/logs/last_full_jobid.txt`.

**RUN COMPLETED + RESULTS (2026-06-08).** The full run finished all **2 epochs / 176 steps**
(6h56m, clean exit; checkpoints every 20 at `checkpoints/rlpt-scienceqa/grpo-qwen2_5vl-3b/
global_step_*`, final = `global_step_176`). `compute_score` now returns a DICT so verl logs
`acc`/`match`/`format` separately (verl uses `result["score"]` as the reward, logs the rest).

**Held-out TEST (1,836 rows, `eval_test_qwen2_5vl.sbatch`, val_only on global_step_176):**
**acc 90.3% · format 100% · reasoning-match F 0.78 · hallucinated-clauses 0.09/resp · reward 6.98.**
**Base Qwen2.5-VL-3B is already 79% zero-shot on ScienceQA (supervisor-measured), so RL gave
+11.3 pts accuracy (79% → 90.3%).** The "−0.29 composite baseline" is NOT the base model being
bad — it's 79% accurate — it's that the untrained model doesn't emit our required `<think>/
<answer>` format, so the harness can't parse its answer. RL's value = +11 pts accuracy AND
better reasoning faithfulness, on an already-competent model.

**Checkpoint sweep (`sweep_checkpoints.sbatch`, 256-row test subset, steps 40→176):** the
DECOMPOSITION is the real finding — format saturates immediately (1.00 by step 40), accuracy
is mostly there early (~0.85 → ~0.90, noisy), but the **bipartite reasoning `match` climbs
steadily and monotonically 0.682 → 0.766** across the run. So RL kept refining *reasoning
faithfulness* (alignment to GT solutions) after format/answers were largely solved — exactly
the signal the match term was designed to capture. Results TSV: `data/logs/match_sweep_results.tsv`.

**Curve anchored (2026-06-09, `data/logs/match_sweep_anchors.tsv`):** base (untrained) on the
same subset = **match 0.371** · acc 0.645 · format 0.480; step 20 = match 0.625 · acc 0.824 ·
format 1.00. Full trajectory: **0.37 → 0.62 (step 20) → 0.68 (40) → 0.77 (176)** — RL roughly
doubled reasoning match; format saturates by step 20. Base acc 0.645 (vs supervisor's 79%) is
the strict-harness format effect, not a contradiction. Base-match caveat: no `<think>` block →
matcher scores the whole (often terse) output, so 0.37 partly reflects brevity. Ops notes:
original step-20 row died on an HF 429 (fix: `HF_HUB_OFFLINE=1`, all data/models cached);
2×A100 allocations launch-failed cluster-wide on 2026-06-09 (REQUEUE_HOLD, unreleasable) →
**always `#SBATCH --no-requeue`**, evals run fine on 1×A100; FSDP checkpoints resume only at
their saved world size (2) → for 1-GPU eval, merge shards to HF first (`verl.model_merger`,
CPU-only; see `sweep_step20_merged.sbatch`; merged step-20 model at
`models/qwen2_5vl-3b-scienceqa-grpo-step20`).

### Validation plan: mimicry vs real reasoning (NEXT STEPS, 2026-06-09)

The match climb (0.37→0.77) is partly circular (we optimized it) and SBERT cosine could
reward *stylistic mimicry* of ScienceQA solutions rather than better image analysis. Design
principle for ruling that out: mimicry is **style-specific + image-independent**; real
reasoning is **content-specific + image-dependent**. Ordered by decisiveness per GPU-hour:

1. **Shuffled-GT control** (no training): score each checkpoint's `<think>` traces against
   GT solutions of *other* same-topic questions (style-matched, content-mismatched). Report
   the **gap** own-GT − shuffled-GT across checkpoints; a widening gap refutes mimicry.
   Needs one generation pass per checkpoint on the 256-row subset (merge remaining FSDP
   checkpoints to HF on CPU first, as done for steps 20/176).
2. **Style-insensitive re-scoring** (no training, reuses pass-1 generations): replace SBERT
   cosine with NLI entailment (DeBERTa-MNLI: "GT solution entails clause?"). If the climb
   survives NLI, the gain is propositional content, not phrasing. Also re-score with a
   different embedding model — a gain visible only under the reward's own encoder = encoder
   hacking.
3. **Image-dependence probe** (no training): eval base + step-176 on the 256 rows with
   swapped images. Real image analysis collapses under swap; mimicry/text-prior survives.
   Trained model should degrade MORE than base if RL increased image-grounding.
4. **Transfer eval** (no training): base vs step-176 on a differently-styled benchmark
   (A-OKVQA / MMMU-mini). Style mimicry can't transfer; visual-reasoning gains can.
5. **`w_match=0` ablation** (one training run — the causal keystone): identical GRPO
   config, reward = format+answer only, same seed/steps; run tests 1–4 on its checkpoints.
   Licenses the claim "the bipartite term *helped*". Stretch arm: `w_answer=0` (match-only)
   — if match alone lifts MCQ accuracy, strongest evidence the term trains real reasoning.

Publishable claim needs three legs: content-not-style (1,2) · image-grounded (3,4) ·
caused-by-match-term (5). Tests 1–3 share one generation pass per checkpoint (~a day on
1×A100); the ablation is the only real compute cost (~7h on 2×A100).

**VALIDATION RESULTS — tests 1–4 ALL PASS (2026-06-10).** Infra: all 9 checkpoints merged
to HF (`merge_all_checkpoints.sbatch`, 12 min CPU); shared greedy-vLLM generation pass
`tools/gen_eval_outputs.py` (offline vLLM needs `VLLM_WORKER_MULTIPROC_METHOD=spawn` —
V1 EngineCore forks outside Ray and hits "Cannot re-initialize CUDA in forked subprocess");
scoring `tools/validation_scoring.py`; TSVs in `data/logs/validation_results/`
(`validation_{main,swap,aokvqa}.tsv`); raw generations in `outputs/validation/`.

1. **Shuffled-GT (PASS)**: own-GT match climbs 0.377 (base) → 0.627 (20) → 0.760 (176),
   monotone; same-topic shuffled-GT flat 0.15 → 0.22–0.23 from step 20 on. Gap widens
   0.22 → 0.53. Style mimicry would lift both; it didn't.
2. **Style-insensitive re-scoring (PASS)**: NLI entailment (DeBERTa-v3 MNLI) own-GT
   0.21 → 0.52 with shuffled-GT flat ~0.07 — the gain is propositional content. Alternate
   encoder (all-mpnet-base-v2) tracks the reward encoder ~1:1 (0.375→0.759) — no encoder
   hacking. Offline scorer also reproduces the verl anchors (0.377/0.627 vs 0.371/0.625).
3. **Image-swap probe (PASS, moderate)**: swapping images collapses accuracy at every
   checkpoint (~−0.21 absolute: base 0.62→0.41, s176 0.875→0.668) — answers are image-
   dependent, not text-prior. Reasoning-match swap-sensitivity GROWS with training
   (Δmatch −0.048 base → −0.064 s20 → −0.079 s176): RL increased image-grounding of the
   reasoning. Swapped s176 acc 0.668 ≫ chance reflects text-answerable items, not leakage.
4. **A-OKVQA transfer (PASS)**: differently-styled benchmark, zero training on it:
   base 0.430 → s176 **0.840** accuracy (format 0.38→1.00; base number carries the usual
   strict-harness deflation). Match-vs-rationales 0.39→0.47. Gains transfer; style can't.
5. **w_match=0 ablation (PASS — match term is CAUSAL, 2026-06-11)**: reward weights
   env-overridable in `graph_match_reward.py` (`RLPT_W_{FORMAT,MATCH,ANSWER,PUN}`);
   ablation set **both** `RLPT_W_MATCH=0` and `RLPT_W_PUN=0` (pun is matcher-derived).
   Chain (train 55105027, 7h12m → merge 55110261 → gen+score 55110262) completed clean;
   results in `data/logs/validation_results_nomatch/`, same 256-row subset/harness as the
   main arm. **Ablation own-GT match: 0.377 → 0.461 (step 20) → flat/declining 0.42–0.46
   → 0.424 (176), vs main arm 0.377 → 0.627 → 0.760.** The small step-20 bump (~+0.08)
   quantifies the answer-only confound; everything beyond it (~0.30 of the main arm's
   +0.38 gain) is caused by the match term. **NLI entailment never moves in the ablation**
   (flat 0.17–0.22 vs main 0.21→0.52) — without the match reward, propositional alignment
   to GT doesn't improve at all. Image-swap match-sensitivity also stays at base level
   (Δmatch −0.036 at step 176 vs base −0.049; main arm grew to −0.079) — the increased
   image-grounding of reasoning is also match-term-caused. **Honest caveat: MCQ accuracy
   does NOT need the match term** — ablation acc 0.93–0.94 late ≥ main arm ~0.88–0.90;
   the term buys reasoning faithfulness + grounding, not accuracy (the `w_answer=0`
   stretch arm would test whether match alone can lift accuracy). Glob gotcha (fixed):
   the sbatch's "LAST checkpoint" pick was lexicographic, so the ablation's swap/A-OKVQA
   passes first ran on step **80**; job 55250678 reran them step-matched at 176 —
   ablation-176 A-OKVQA acc **0.844** ≈ main-176 0.840 (in-distribution-style transfer
   accuracy is entirely the answer term; only MMK12 below separates the arms).

Base-model rows in these tables use GREEDY decoding — base format 0.38 here vs 0.48
sampled earlier; both deflate base acc vs the supervisor's 79% lenient-parse number.
Comparisons within a column (across checkpoints) are the meaningful signal.

**MMK12 hard-benchmark result — match term DOES buy accuracy OOD (2026-06-12).** The
ScienceQA "no accuracy difference" caveat was a ceiling effect, not a property of the
match term. On `FanqingM/MMK12` test (MM-Eureka; 2,000 K12 exam MCQs, 500 each
math/physics/chem/bio, single image, letter answers; chosen over Geo3k = free-form
numeric grading and ViRL-39K = train-split-only), 1,024-row balanced sample, zero
training on it:

| model | acc | format |
|---|---|---|
| base | 0.311 | 0.18 |
| ablation-176 (answer-only) | 0.397 | 0.99 |
| main-176 (with bipartite match) | **0.439** | 0.99 |

**+4.1 pts from the match term at identical format; paired McNemar exact p = 0.021**
(discordant 179 main-only-correct vs 137 ablation-only-correct). So: where reasoning is
the bottleneck (base 31% ≈ near chance vs ScienceQA's 62%), training the reasoning
channel converts to answer accuracy. Full claim set now: content-not-style (1,2) ·
image-grounded (3,4) · match-term-causal for faithfulness (5) · **and OOD accuracy
gains (MMK12)**. Infra: `src/data_factory/mmk12_to_eval.py` (parses "A. ..." choice
lines out of the question, stores gold CHOICE TEXT so `answer_correct` takes letter or
text; 6/2000 rows unparseable) → `data/mmk12_eval/mmk12_test1024.parquet`;
`src/train/gen_hard_evals.sbatch` (job 55250678: also reran ablation-176 A-OKVQA+swap
to fix the step-80 glob wart); scorer `tools/score_mcq_eval.py` →
`data/logs/validation_results/validation_mmk12.tsv`. Bug fixed in `answer_correct`:
`"③".isdigit()` is True but `int("③")` raises → guard with `p.isascii()` (models emit
circled digits on K12 items; scoring-only fix, prior runs would have crashed not
mis-scored).

**Ops TODO (next login):** investigate **interactive jobs** on Sol (`salloc` / `srun --pty`
/ Sol's `interactive` wrapper) as an alternative to fire-and-wait sbatch for debugging —
gets a shell on a GPU node for iterating on eval scripts without queue round-trips.

### Legacy tooling (scene-graph track, kept for reference / SFT bootstrap)

- `tools/eval_sgg_vs_gt.py` — `prf`/`by_type` matching helpers.
- `tools/vg_query_to_graph.py` — GT fact sets from annotations.
- `tools/fetch_vg_images.py` + `src/data_factory/fetch_vg_images.sbatch` — VG image fetch.

---

## History / decisions

- **First full run** met quotas exactly but only used 68 unique train images (~59
  records/image) because greedy quota fills from object-dense scenes first. Fixed with
  `MAX_RECORDS_PER_IMAGE_PER_OP = 2` and quota bump from 1000 → 2500 per op.
- **Diversity-capped run** produced 10k train (1699 unique images) + 1k each for 3 eval
  splits in ~5 min with warm cache.
- **Pluralization fix**: count templates originally used naive `{a}s` suffix → produced
  "mans", "womans", "feets", etc. (294/10000 = 2.9% of train records). Patched by adding
  `pluralize()` and switching templates to `{a_plural}` / `{b_plural}` placeholders.
- Default model citizen path for generated data: `data/qcvsr_gqa_v1/`. Test runs go to
  `data/qcvsr_gqa_v1_test/` to avoid collision.
- **Multi-instance ambiguity fix + compositional schemas (2026-05-30)**: ~33% of the old
  spatial/size records referenced a class with multiple instances ("which is left, the cup
  or the plate" with 3 cups → ill-posed). An interim union-bbox aggregation didn't truly
  resolve it (union of scattered instances overlaps). Resolution: spatial/size now require
  **unique referents** (single-instance classes, single `bbox` evidence, original
  `compare_*` verifiers). Multi-instance objects are instead routed to count and the three
  new **compositional** schemas (`attr_select_spatial`, `count_relation`, `relation_target`)
  where multiple instances are the signal and constraints yield a unique answer. Validated
  offline against scene graphs: 865k candidates, 0 logic-check failures.
- **Scene graphs + parser DROPPED (2026-06)**: brief intermediate plan was caption→frozen-parser
  →scene-graph matching, but the parser-in-loop reward-hacking confound made it unattractive.
  Final direction: model emits a **free-text description**, scored against GQA annotations via
  **BERT-based soft bipartite (Hungarian) matching** (semantic + optional multimodal box-IoU/
  region-CLIP costs, precision-dominant Fβ). **No parser in the reward loop** → confound gone.
  Parser tooling kept as legacy / possible SFT bootstrap only. See "Image-description RL reward"
  section above.
