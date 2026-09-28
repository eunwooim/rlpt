#!/bin/bash
# Arm 1 launch — exact command from LAUNCH_PLAN.md (gate-2, user-approved 2026-08-22)
sbatch --job-name=arm1_3b_chunk \
  --export=ALL,ARM=1,MODEL=Qwen/Qwen2.5-VL-3B-Instruct,REWARD=match,SEG=chunker,TAU=0.45,OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_match_chunker \
  /scratch/sghos104/rlpt/grpo_arms/run_arm.sbatch
