# Cross-dataset eval — acquisition record (2026-07-29)

**EVAL-ONLY data.** Nothing under `data/xdataset_eval/` is ever written into
any training directory. Downloaded by SLURM job 59934752 (rc=0); full
manifest with first-record examples: `acquisition.json`.

## PRM800K (OpenAI) — `data/xdataset_eval/prm800k/`

Source: `github.com/openai/prm800k`, `prm800k/data/phase2_{train,test}.jsonl`
(fetched via the raw redirect; LFS-pointer detection in
`download_xdataset.py` did not trigger — content served directly).

| file | rows | bytes | sha256 |
|---|---:|---:|---|
| phase2_train.jsonl | 97,782 | 456,135,365 | `1110237feeb51d1bc200cb37b8f965cfdc1036eac7d506094049366fe7dc1089` |
| phase2_test.jsonl | 2,762 | 12,240,719 | `6b172efa884ac8341a946dd82e06947c135b7254109fb3f7aa907c715d98aaad` |

Record = one labeler trajectory over one generator solution: `question`
(problem, ground truths, `pre_generated_steps`) + `label.steps[]`, each step
holding rated `completions[]`, optional `human_completion`, and
`chosen_completion` (index of the continuation actually taken). The gold
step sequence we evaluate = the chosen-completion path (human fallback);
see `convert_xdataset.py` for the exact filter funnel.

Example (abridged; full in acquisition.json): arithmetic-sequence problem,
13 `label.steps`, each with `completions:[{text, rating}]`, `chosen_completion`.

## ProcessBench (Qwen) — `data/xdataset_eval/processbench/`

Source: HF dataset `Qwen/ProcessBench` (snapshot_download, 6 files).

| file | rows | bytes | sha256 |
|---|---:|---:|---|
| gsm8k.json | 400 | 574,343 | `fb1c59dfd1e83e1c5a48dbe598db27e4bc0b8bc524a09445f2d4b69178f61349` |
| math.json | 1,000 | 1,938,778 | `60cb0d7a69cbec98d43cf50855b9b7e7294599c3c9a24bdd031abd18c3b6b136` |
| olympiadbench.json | 1,000 | 2,808,955 | `e9d801c9f3c03f949adbfa08e90cb21ad85c0b9224bc5e98326f8215c218aa55` |
| omnimath.json | 1,000 | 2,842,872 | `e8d7cf4512c82168dae2fe48b1718069a3002e7910af1d180a27ee0d9948ba47` |

Record = `{id, generator (e.g. Qwen2-7B-Instruct), problem, steps: [...],
final_answer_correct, label}` — `steps` is the pre-split solution used by
human error-annotators; we use it as the gold segmentation (vetted, not
authored — see REPORT.md caveat).

Example (abridged): `gsm8k-0`, flamingo counting problem, 6 steps.
