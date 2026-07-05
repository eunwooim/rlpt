# Reliability Experiment Commands

## Caption / Negation Reliability

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/metrics/reliable/run_hn_neg.py \
  --datasets sugarcrepe,sugarcrepepp,negbench \
  --scorers sbert,nli,bertscore \
  --output_dir src/outputs/reliable/caption_negation/run_YYYYMMDD_HHMMSS \
  --batch_size 32 \
  --device auto \
  --seed 42
```

```bash
python src/metrics/reliable/aggregate_hn_neg.py \
  --output_dir src/outputs/reliable/caption_negation/run_YYYYMMDD_HHMMSS
```

## VisualPRM Segment Ranking

Full run:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/metrics/reliable/run_visualprm_numeric.py \
  --output_dir src/outputs/reliable/visualprm_numeric/run_YYYYMMDD_HHMMSS \
  --dataset_name OpenGVLab/VisualPRM400K-v1.1-Raw \
  --dataset_config default \
  --dataset_split train \
  --dataset_streaming \
  --max_traces 10000 \
  --scorers sbert,nli,cross_nli,bge_reranker \
  --generator_model Qwen/Qwen3-32B \
  --tensor_parallel_size 4 \
  --generation_batch_size 64 \
  --batch_size 32 \
  --device auto \
  --seed 42
```

Generation only:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/metrics/reliable/run_visualprm_numeric.py \
  --output_dir src/outputs/reliable/visualprm_numeric/run_YYYYMMDD_HHMMSS \
  --dataset_name OpenGVLab/VisualPRM400K-v1.1-Raw \
  --dataset_config default \
  --dataset_split train \
  --dataset_streaming \
  --max_traces 10000 \
  --generate_only \
  --generator_model Qwen/Qwen3-32B \
  --tensor_parallel_size 4 \
  --generation_batch_size 64 \
  --seed 42
```

Scoring only:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/metrics/reliable/run_visualprm_numeric.py \
  --cases_jsonl src/outputs/reliable/visualprm_numeric/run_YYYYMMDD_HHMMSS/raw/numeric_cases.jsonl \
  --output_dir src/outputs/reliable/visualprm_numeric/run_YYYYMMDD_HHMMSS \
  --score_only \
  --scorers sbert,nli,cross_nli,bge_reranker \
  --batch_size 32 \
  --device auto \
  --seed 42
```

Regenerate VisualPRM tables:

```bash
python src/metrics/reliable/aggregate_visualprm_numeric.py \
  --output_dir src/outputs/reliable/visualprm_numeric/run_YYYYMMDD_HHMMSS
```
