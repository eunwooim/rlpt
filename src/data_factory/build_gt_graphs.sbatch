#!/bin/bash
#SBATCH --job-name=rlpt-gt-graphs
#SBATCH --partition=htc
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/sghos104/rlpt/data/logs/gt-graphs-%j.out
#SBATCH --error=/scratch/sghos104/rlpt/data/logs/gt-graphs-%j.err

# Build the consistency-checked GT scene graphs over ALL Visual Genome images.
# Each region -> a caption-faithful scene graph (objects/attributes/relations),
# deterministic text<->graph grounding, degenerate graphs dropped.

set -euo pipefail
mkdir -p /scratch/sghos104/rlpt/data/logs

module load mamba/latest
ENV_PREFIX=/scratch/sghos104/envs/rlpt-data
export NLTK_DATA=/scratch/sghos104/rlpt/data/nltk_data

cd /scratch/sghos104/rlpt

echo "[sbatch] starting $(date -Iseconds) on $(hostname)"
"$ENV_PREFIX/bin/python" -u tools/vg_query_to_graph.py \
    --attach-qa \
    --out data/visual_genome/qsg_full.jsonl

echo "[sbatch] done $(date -Iseconds)"
echo "[sbatch] GT graphs: $(wc -l < data/visual_genome/qsg_full.jsonl)"
