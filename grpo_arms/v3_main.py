#!/usr/bin/env python
"""v3_main.py — verl main_ppo entrypoint using FilterGroupsTrainer (DAPO-style filter_groups port, see v3_trainer.py).

Run exactly like `python -m verl.trainer.main_ppo <hydra overrides>`; train_arm.py selects it via --entrypoint v3_main
(grpo_arms/ must be on PYTHONPATH so the Ray TaskRunner worker can import v3_trainer).
"""
import os
import sys

import hydra
import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verl.trainer.main_ppo import auto_set_device, migrate_legacy_reward_impl, run_ppo  # noqa: E402

from v3_trainer import V3TaskRunner  # noqa: E402


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    # same pre-processing as verl.trainer.main_ppo.main: without migrate_legacy_reward_impl the custom_reward_function
    # never reaches config.reward.* and the reward loop falls back to default_compute_score (smoke 63052231 failed that way)
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(V3TaskRunner))


if __name__ == "__main__":
    main()
