# Launch plan (gated — do not submit past a gate without go-ahead)

TAU = 0.45 (gate-1 decision, see TAU_DECISION.md; 0.56 was the FPR<=1% alternative, rejected — split-cliff bias against marker arms).
Each arm: fixed OUTDIR; resubmitting the identical command resumes from the
latest checkpoint (verl resume_mode=auto + save_freq 20 + keep 2).

## Arm 1 (gate 2)
sbatch --job-name=arm1_3b_chunk --export=ALL,ARM=1,MODEL=Qwen/Qwen2.5-VL-3B-Instruct,REWARD=match,SEG=chunker,TAU=0.45,OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_match_chunker grpo_arms/run_arm.sbatch

## SCOPE CHANGE 2026-08-25 (user): arm 3 launched alongside arm 1 (job 62366932);
## arms 4 (7B match/chunker) and 6 (7B vprm) SKIPPED. Remaining gated: arms 2, 5.

## Arms 2-6 (gate 3; 4 and 6 skipped)
sbatch --job-name=arm2_3b_marker --export=ALL,ARM=2,MODEL=Qwen/Qwen2.5-VL-3B-Instruct,REWARD=match,SEG=marker,TAU=0.45,OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm2_3b_match_marker grpo_arms/run_arm.sbatch
sbatch --job-name=arm3_3b_vprm   --export=ALL,ARM=3,MODEL=Qwen/Qwen2.5-VL-3B-Instruct,REWARD=vprm,SEG=vprm,OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm3_3b_vprm,GPUMEM=0.40 grpo_arms/run_arm.sbatch
sbatch --job-name=arm4_7b_chunk  --export=ALL,ARM=4,MODEL=Qwen/Qwen2.5-VL-7B-Instruct,REWARD=match,SEG=chunker,TAU=0.45,OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm4_7b_match_chunker,GPUMEM=0.40 grpo_arms/run_arm.sbatch
sbatch --job-name=arm5_7b_marker --export=ALL,ARM=5,MODEL=Qwen/Qwen2.5-VL-7B-Instruct,REWARD=match,SEG=marker,TAU=0.45,OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm5_7b_match_marker,GPUMEM=0.40 grpo_arms/run_arm.sbatch
sbatch --job-name=arm6_7b_vprm   --export=ALL,ARM=6,MODEL=Qwen/Qwen2.5-VL-7B-Instruct,REWARD=vprm,SEG=vprm,OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm6_7b_vprm,GPUMEM=0.35 grpo_arms/run_arm.sbatch

## Shared config (train_arm.py defaults; identical across arms)
- frozen subset sha bd15b1f1..., 5,880 train / 120 verl-val (seed-0 split)
- batch 32 x rollout_n 5, mini 8, micro 1, lr 1e-5, KL loss on
- max_prompt 2048 / max_response 1024, image cap 640*28*28
- 183 steps = 1 epoch; save/test every 20
- 2xA100, 200G host mem, 48h wall

## Post-arm eval (per arm, after completion)
1. merge:  <rlpt-train python> -m verl.model_merger merge --backend fsdp \
     --local_dir <OUTDIR>/checkpoints/global_step_183/actor --target_dir <OUTDIR>/hf_final
2. generate: vpb_generate.py --model_path <OUTDIR>/hf_final --tag arm<k>   (1xA100)
3. score:    score_vpb.py
Baselines: vpb_generate.py on the two base models (tags base3b / base7b) once.
