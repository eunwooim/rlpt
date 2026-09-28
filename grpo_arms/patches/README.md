# patches/

## filter_groups_partial_batch.patch (2026-09-11)
Context: verl 0.8.0 as installed in /scratch/sghos104/envs/rlpt-train ships only `FilterGroupsConfig` (no implementation, no `recipe/dapo`).
The DAPO dynamic-sampling loop was therefore ported into `grpo_arms/v3_trainer.py` (FilterGroupsTrainer.fit). Upstream DAPO raises
`ValueError("Generated too many batches ...")` when `max_num_gen_batches` is exhausted before `train_batch_size` mixed groups are collected.
This patch is the diff between that strict behaviour and the shipped behaviour: log a warning and train on the mixed groups collected so
far (truncated to a multiple of `ppo_mini_batch_size` prompts; all of them if fewer than one mini-batch). Metrics `filter/partial`,
`filter/num_gen_batches`, `filter/num_prompts_trained` record when the fallback fired.

Reapply (if v3_trainer.py is ever regenerated from the strict version): `patch -p1 < grpo_arms/patches/filter_groups_partial_batch.patch`
(paths inside the patch are informational; apply against grpo_arms/v3_trainer.py).

## adaptive_entropy.diff (2026-09-18, run R)
Skywork-OR1-style adaptive entropy coefficient for verl 0.8.0, implemented as two NEW files (grpo_arms/adaptive_entropy.py, grpo_arms/entropy_main.py)
rather than an edit of the env: the driver-side trainer subclass measures `actor/entropy` in the old-log-prob pass, injects the live coefficient
into the batch meta_info, and updates `coeff <- clip(coeff + ENTROPY_LR·(ENTROPY_TARGET − entropy), 0, ENTROPY_COEFF_MAX)` after every optimizer
step; the worker-side `ppo_loss` replacement adds `−coeff·entropy_loss`. Gated by ENTROPY_MODE (default off = no behaviour change). Needs
`--entrypoint entropy_main` and `actor_rollout_ref.actor.calculate_entropy=True`. Logged per step: actor/entropy_coeff (used), actor/entropy_coeff_applied
(worker), actor/entropy_coeff_next; state file <checkpoints>/entropy_coeff_log.jsonl (restored on resume). Smoke: grpo_arms/full/smoke_entropy.sbatch.
