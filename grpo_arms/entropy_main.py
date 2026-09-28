#!/usr/bin/env python
"""entropy_main.py — verl main_ppo entrypoint with the adaptive entropy coefficient (grpo_arms/adaptive_entropy.py), gated by ENTROPY_MODE.
ENTROPY_MODE=off -> byte-for-byte the same as `python -m verl.trainer.main_ppo`; ENTROPY_MODE=adaptive -> AdaptiveEntropyTaskRunner.
Run like `python -m verl.trainer.main_ppo <hydra overrides>` (train_arm.py --entrypoint entropy_main; grpo_arms on PYTHONPATH for Ray workers).
"""
import os
import sys

import hydra
import ray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adaptive_entropy  # noqa: E402  (install() runs at import when ENTROPY_MODE=adaptive)
from verl.trainer.main_ppo import auto_set_device, migrate_legacy_reward_impl, run_ppo  # noqa: E402


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    if adaptive_entropy.MODE == "adaptive":
        run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(adaptive_entropy.AdaptiveEntropyTaskRunner))
    else:
        run_ppo(config)


if __name__ == "__main__":
    main()
