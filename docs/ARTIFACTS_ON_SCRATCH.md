# Experiment artifacts that live on /scratch (not in git)

Written 2026-09-28. The repository tracks code, sbatch launchers, docs, reports, status logs and small data manifests. Everything below is excluded by `.gitignore` because of size (GitHub rejects files over 100 MB and the total is ~3 TB); nothing was deleted. All paths are relative to `/scratch/sghos104/rlpt` on ASU Sol.

| path | size | what it is | how to regenerate |
|---|---|---|---|
| `grpo_arms/runs/` | 2.0 TB | every GRPO run: FSDP checkpoints, merged HF checkpoints (`hf_step_*`), per-step rollout dumps, verl logs. Runs used in the reports: `arm1_3b_matchv4_softgate` (Bipartite, epoch 1), `arm1_3b_match_e2` (epoch 2), `arm1_3b_rlvr_entropy` (RLVR, epoch 1), `arm1_3b_rlvr_e2` (epoch 2), `arm1_3b_ctrl_v4` (v4 A), earlier v2/v3 runs, smokes | the sbatch files under `grpo_arms/full/`, `grpo_arms/e2/`, `grpo_arms/v4/`, `grpo_arms/v3/` + `grpo_arms/launch_*.sh` |
| `grpo_arms/evals/*.jsonl`, `grpo_arms/evals/ood/` | 396 MB | greedy generations per evaluated checkpoint on vpb_test (2,456 q) and the seven OOD benchmarks; the numbers in `FULL_REPORT.md`, `E2_REPORT.md`, `OOD_REPORT.md` are computed from these | `grpo_arms/eval_arm1v2.sbatch`, `grpo_arms/ood/` |
| `grpo_arms/data/vpb_dev_images/`, `data/ood_eval/` | 30 MB / ~1.5 GB | bounded images for vpb_dev and the OOD benchmark sets | `grpo_arms/build_vpb_split.py`, `grpo_arms/ood/build_ood_sets.py` |
| `checkpoints/`, `models/` | 777 GB / 127 GB | SFT/RLVR baseline checkpoints and downloaded base models | `run_*_sft.sbatch`, `run_*_rlvr.sbatch`, `run_qwen_download.sbatch` |
| `src/outputs/` | (ignored already) | cold-start SFT `coldstart_vprm_unf_3b/run_v1/checkpoints/checkpoint-25` (= cs25), 2x2 baseline evals, reliability-suite scores | `run_coldstart_sft_vprm_unf.sbatch`, `run_eval_2x2.sbatch`, `run_reliable_*.sbatch` |
| `data/` | (ignored already) | VisualPRM400K v1 zip and v1.1-Raw, VisualProcessBench, MMK12, negation/paraphrase pair files (`data/visualprm400k/pairs.jsonl` 40,000 rows; `paraphrase_pairs.jsonl` 10,000 rows), HF cache | `run_visualprm_download.sbatch`, `run_vpb_download.sbatch`, `build_negation_pairs.py`, `build_paraphrase_pairs.py` |
| `reward_redesign/judge_bench/out/`, `reward_redesign/judge_bench/EQUATE/` | 124 MB | judge-benchmark shard outputs (raw NLI probabilities per pair) and the EQUATE download; `JUDGE_BENCH.md` is computed from these | `reward_redesign/judge_bench/jb_single.sbatch` (or `jb_launch.sh`) then `jb_aggregate.py` |
| `chunker/env/`, `chunker/runs/`, `chunker/data*/`, `chunker/release/` | 63 GB | the scoring/NLI conda env, DeBERTa chunker training runs, datasets and released chunker models | `chunker/` README and sbatch files |
| `envs/` | 8.9 GB | `rlpt-train` (verl/vLLM) and other Python envs | `docs/` env notes (`src/train/README.md`) |
| `canonical*.jsonl`, `chunk_canonical/` | 75 GB | canonical chunked VisualPRM traces (deliverable for Eun Woo) and shards | `chunk_canonical/` scripts |
| `chunk_eval/exp2_populations/`, `chunk_eval/singles/`, `chunk_eval/xdataset/**/*.jsonl` | 230 MB | chunk-evaluation populations and scores; `chunk_eval/EVAL_REPORT.md` is tracked | `chunk_eval/` scripts |
| `granularity/bands/`, `granularity/bands_probe/`, `granularity/nli_pairs.jsonl` | 1.5 GB | granularity-band generations and NLI pairs | `granularity/` scripts |
| `grpo_arms/logs/`, `*.log` | 30 MB | Slurm job logs | — |

Frozen inputs that ARE tracked because they are small and everything depends on them: `grpo_arms/data/train_subset.jsonl` (+ `.sha256`, `subset_manifest.json`), `grpo_arms/data/vpb_dev.jsonl`, `vpb_test.jsonl`, `vpb_eval.jsonl` (+ sha files), `reward_redesign/probe.jsonl` and its scored variants, `reward_redesign/judge_pairs_300.{jsonl,md}`.
