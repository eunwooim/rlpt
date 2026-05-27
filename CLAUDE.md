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

For each image, iterate through object pairs and emit candidates for three schemas:

| Schema | Operations | Verifier examples |
|---|---|---|
| `bbox_pair_spatial_compare` | `left`, `right`, `higher`, `lower` | `compare_x_center_min/max`, `compare_y_center_min/max` |
| `bbox_pair_size_compare` | `larger`, `smaller` | `compare_bbox_area_max/min` |
| `count_pair_compare` | `more_than_yesno`, `difference` | `compare_count_greater_than`, `count_difference` |

Bounding boxes are **xyxy normalized to 0–1000**.

### Splits (deterministic via `seed=42`)

- `train` (80% of images): ops `{left, right, larger, more_than_yesno}`
- `eval_id` (7%): same ops as train, paraphrased templates from `TRAIN_TEMPLATES`
- `eval_ood_template` (7%): same ops, distinctly worded templates from `OOD_TEMPLATES`
- `eval_ood_operation` (~6%): held-out ops `{higher, lower, smaller, difference}`

### Key constants in `gqa.py`

- `TARGET_QUOTAS` — per-op record cap (currently 2500 for train, 10% for evals → 250 each).
- `MAX_RECORDS_PER_IMAGE_PER_OP = 2` — diversity cap; without it, a single object-dense
  image floods the quota (early run produced 4000 train records from only 68 images).
- `STUFF_OBJECTS` — name blacklist (sky, ground, wall, ...) to skip non-thing classes.
- `IRREGULAR_PLURALS` / `INVARIANT_PLURALS` — used by `pluralize()` for count templates;
  without this, count questions read "Are there more mans than..." which is ungrammatical.

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
| `evidence_two_nodes` | Exactly two `nodes` (pairwise comparison) |
| `evidence_schema_match` | Schema name matches node field types (`bbox` vs `count`) |
| `answer_consistency` | Re-runs the verifier against evidence, confirms target_answer — **most important** |
| `pluralization` | Question text contains no naive bad plurals (`mans`, `feets`, ...) |
| `required_fields` | Top-level keys all present |

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
