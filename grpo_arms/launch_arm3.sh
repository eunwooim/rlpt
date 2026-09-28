#!/bin/bash
# Arm 3 launch — exact command from LAUNCH_PLAN.md (user-approved 2026-08-25:
# "continue with 3B with VisualPRM"; arms 4 and 6 skipped per same directive)
sbatch --job-name=arm3_3b_vprm \
  --export="ALL,ARM=3,MODEL=Qwen/Qwen2.5-VL-3B-Instruct,REWARD=vprm,SEG=vprm,OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm3_3b_vprm,GPUMEM=0.30,EXTRA=--verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288" \
  /scratch/sghos104/rlpt/grpo_arms/run_arm.sbatch
