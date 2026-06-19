# Text Scorer Reliability Experiment

This package runs proof-of-concept reliability checks for text segment scorers used in reasoning-segment reward design. It normalizes benchmark captions into text pairs, saves every raw scorer value, and writes inspection tables.

## Datasets

- `sugarcrepe`: separation mode. Positive is the correct caption; negatives are hard negative captions. Subcategories are loaded from the official SugarCrepe JSON files and keep their original names: `replace_obj`, `replace_att`, `replace_rel`, `swap_obj`, `swap_att`, `add_obj`, `add_att`.
- `sugarcrepepp`: ranking mode. `caption` is the anchor, `caption2` is the positive, and `negative_caption` is the hard negative. Config/subcategory names are preserved from `Aman-J/SugarCrepe_pp`.
- `negbench`: separation mode. The loader checks for local NegBench CSVs for `MSR-VTT MCQ-Neg`, `HardNeg-Syn MCQ-Neg`, `VOC2007 MCQ-Neg`, `COCO-MCQ-Neg`, and `CheXpert control/negation task`. Missing files are recorded in `load_errors.jsonl`.

## Scorers

- `sbert`: cosine similarity from `sentence-transformers/all-mpnet-base-v2`, implemented with `transformers` mean pooling.
- `nli`: `microsoft/deberta-large-mnli`; saves directional entailment, contradiction, neutral probabilities plus `nli_score_coverage` and `nli_score_equiv`.
- `bertscore`: token-similarity precision, recall, and F1 from `microsoft/deberta-xlarge-mnli`, implemented with `transformers`.

## Outputs

- `cases_top3.jsonl`: normalized cases with `case_id`, `dataset`, `subcategory`, `mode`, `anchor`, `positive`, and `negatives`.
- `scores_top3.jsonl`: one scorer row per text pair, including raw score fields and metadata.
- `subcategory_counts.csv`: case and pair counts by dataset/subcategory/mode.
- `cases_preview.csv`: up to 5 examples per dataset/subcategory.
- `table_ranking_by_subcategory.csv`: ranking accuracy and margins for ranking cases.
- `table_separation_by_subcategory.csv`: false positive rates over tau grid `0.3` to `0.9`.
- `table_scorer_overall.csv`: dataset-level scorer summary.
- `load_errors.jsonl`: dataset/subcategory load failures.
- `scorer_errors.jsonl`: scorer load/run failures.

## Commands

Run commands from the repository root. The `microsoft/deberta-large-mnli` loader may print an unused `config` weight warning; this is expected for this checkpoint and is not a scorer failure. Check `scorer_errors.jsonl`; an empty file means all requested scorers ran.

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n rlpt python src/metrics/reliable/run_experiment.py \
  --datasets sugarcrepe,sugarcrepepp,negbench \
  --scorers sbert,nli \
  --output_dir src/outputs/reliable_smoke \
  --max_cases_per_subcategory 3 \
  --batch_size 4 \
  --device auto \
  --seed 42
```

Overnight run:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run -n rlpt python src/metrics/reliable/run_experiment.py \
  --datasets sugarcrepe,sugarcrepepp,negbench \
  --scorers sbert,nli,bertscore \
  --output_dir src/outputs/reliable \
  --max_cases_per_subcategory 500 \
  --batch_size 32 \
  --device auto \
  --seed 42
```

Aggregation is integrated into `run_experiment.py`. To regenerate tables from existing JSONL outputs:

```bash
conda run -n rlpt python src/metrics/reliable/aggregate.py --output_dir src/outputs/reliable
```

Hyperparameter search, tau selection, and calibration are intentionally deferred to a future script. This run saves raw values needed for that later step.

## Expected Missing Data

NegBench CSV files are not bundled with this repo. If they are absent, the run records one `FileNotFoundError` per requested NegBench subcategory in `load_errors.jsonl` and continues with SugarCrepe and SUGARCREPE++.

## Latest Overall Results

Run: `--max_cases_per_subcategory 500`, `--scorers sbert,nli,bertscore`. Completed with 0 load errors and 0 scorer errors. Outputs are under `src/outputs/reliable`.

### Case Counts

| dataset | subcategory | mode | num_cases | num_positive_pairs | num_negative_pairs |
|:--|:--|:--|--:|--:|--:|
| negbench | COCO-MCQ-Neg | separation | 500 | 0 | 1500 |
| negbench | CheXpert control/negation task | separation | 500 | 0 | 500 |
| negbench | HardNeg-Syn MCQ-Neg | separation | 500 | 0 | 1500 |
| negbench | MSR-VTT MCQ-Neg | separation | 500 | 0 | 1500 |
| negbench | VOC2007 MCQ-Neg | separation | 500 | 0 | 1500 |
| sugarcrepe | add_att | separation | 500 | 0 | 500 |
| sugarcrepe | add_obj | separation | 500 | 0 | 500 |
| sugarcrepe | replace_att | separation | 500 | 0 | 500 |
| sugarcrepe | replace_obj | separation | 500 | 0 | 500 |
| sugarcrepe | replace_rel | separation | 500 | 0 | 500 |
| sugarcrepe | swap_att | separation | 500 | 0 | 500 |
| sugarcrepe | swap_obj | separation | 245 | 0 | 245 |
| sugarcrepepp | replace_attribute | ranking | 500 | 500 | 500 |
| sugarcrepepp | replace_object | ranking | 500 | 500 | 500 |
| sugarcrepepp | replace_relation | ranking | 500 | 500 | 500 |
| sugarcrepepp | swap_atribute | ranking | 500 | 500 | 500 |
| sugarcrepepp | swap_object | ranking | 245 | 245 | 245 |

### Scorer Overall

| dataset | scorer | score_field | ranking_acc | separation_fpr_at_0_6 | mean_positive_score | mean_negative_score | mean_margin |
|:--|:--|:--|--:|--:|--:|--:|--:|
| negbench | bertscore | bertscore_f1 |  | 0.9968 |  | 0.8279 | -0.2279 |
| negbench | bertscore | bertscore_precision |  | 0.9968 |  | 0.8314 | -0.2314 |
| negbench | bertscore | bertscore_recall |  | 0.9954 |  | 0.8255 | -0.2255 |
| negbench | nli | C_ab |  | 0.9540 |  | 0.9491 | -0.3491 |
| negbench | nli | C_ba |  | 0.9468 |  | 0.9421 | -0.3421 |
| negbench | nli | E_ab |  | 0.0057 |  | 0.0112 | 0.5888 |
| negbench | nli | E_ba |  | 0.0063 |  | 0.0130 | 0.5870 |
| negbench | nli | nli_score_coverage |  | 0.0025 |  | 0.0045 | 0.5955 |
| negbench | nli | nli_score_equiv |  | 0.0014 |  | 0.0041 | 0.5959 |
| negbench | sbert | raw_score |  | 0.6385 |  | 0.6522 | -0.0522 |
| sugarcrepe | bertscore | bertscore_f1 |  | 1.0000 |  | 0.9475 | -0.3475 |
| sugarcrepe | bertscore | bertscore_precision |  | 1.0000 |  | 0.9423 | -0.3423 |
| sugarcrepe | bertscore | bertscore_recall |  | 1.0000 |  | 0.9530 | -0.3530 |
| sugarcrepe | nli | C_ab |  | 0.5257 |  | 0.5312 | 0.0688 |
| sugarcrepe | nli | C_ba |  | 0.5039 |  | 0.5095 | 0.0905 |
| sugarcrepe | nli | E_ab |  | 0.1186 |  | 0.1325 | 0.4675 |
| sugarcrepe | nli | E_ba |  | 0.4502 |  | 0.4589 | 0.1411 |
| sugarcrepe | nli | nli_score_coverage |  | 0.0998 |  | 0.1979 | 0.4021 |
| sugarcrepe | nli | nli_score_equiv |  | 0.0986 |  | 0.2603 | 0.3397 |
| sugarcrepe | sbert | raw_score |  | 0.9861 |  | 0.8846 | -0.2846 |
| sugarcrepepp | bertscore | bertscore_f1 | 0.1372 |  | 0.9053 | 0.9483 | -0.0430 |
| sugarcrepepp | bertscore | bertscore_precision | 0.1243 |  | 0.8977 | 0.9478 | -0.0501 |
| sugarcrepepp | bertscore | bertscore_recall | 0.1773 |  | 0.9133 | 0.9489 | -0.0356 |
| sugarcrepepp | nli | C_ab | 0.0192 |  | 0.0157 | 0.7599 | -0.7442 |
| sugarcrepepp | nli | C_ba | 0.0263 |  | 0.0128 | 0.7322 | -0.7194 |
| sugarcrepepp | nli | E_ab | 0.9724 |  | 0.9580 | 0.1841 | 0.7739 |
| sugarcrepepp | nli | E_ba | 0.9657 |  | 0.9516 | 0.2254 | 0.7261 |
| sugarcrepepp | nli | nli_score_coverage | 0.9657 |  | 0.9485 | 0.1534 | 0.7951 |
| sugarcrepepp | nli | nli_score_equiv | 0.9679 |  | 0.9469 | 0.1575 | 0.7894 |
| sugarcrepepp | sbert | raw_score | 0.6272 |  | 0.9305 | 0.8728 | 0.0578 |
