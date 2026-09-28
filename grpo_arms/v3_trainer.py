#!/usr/bin/env python
"""v3_trainer.py — minimal port of DAPO-style `algorithm.filter_groups` into verl 0.8.0's RayPPOTrainer (2026-09-11).

The installed verl 0.8.0 ships `FilterGroupsConfig` (verl/trainer/config/algorithm.py, `AlgoConfig.filter_groups`,
default None) but NO implementation: neither `trainer/ppo/ray_trainer.py` nor any `recipe/` directory exists in the
package. This module subclasses RayPPOTrainer and re-implements `fit()` (a copy of the installed one, minus the REMAX
branch) with the DAPO dynamic-sampling loop inserted between reward computation and the rest of the step:

    for each dataloader batch:
        generate n rollouts / prompt, compute rewards
        if filter_groups.enable:
            keep only prompt groups whose `metric` (e.g. "acc") has std > 0 across the n rollouts ("mixed" groups)
            accumulate kept groups; if fewer than train_batch_size prompts and max_num_gen_batches not reached -> pull
            the next batch (no optimizer step); otherwise proceed with what we have
            PARTIAL-BATCH PATCH (this port, by design): when max_num_gen_batches is exceeded the upstream DAPO loop
            raises ValueError; here we log a WARNING and train on the mixed groups collected so far, truncated to a
            multiple of ppo_mini_batch_size prompts (or all of them if fewer than one mini-batch).
        train on the (possibly partial) batch — everything else identical to RayPPOTrainer.fit

Logged per step: filter/num_gen_batches, filter/num_prompts_generated, filter/num_prompts_mixed, filter/num_prompts_trained,
filter/mixed_frac, filter/partial (1 = fallback used). Console line: "[filter_groups] step=N gen_batches=.. mixed=../.. trained=.. partial=..".

Config: +algorithm.filter_groups.enable=true +algorithm.filter_groups.metric=acc +algorithm.filter_groups.max_num_gen_batches=3
(the `+` is needed because ppo_trainer.yaml has no filter_groups key). `metric` must be a key the reward function returns in
its dict (grpo_arms/arm_reward.py emits "acc"). Entry point: grpo_arms/v3_main.py (-> V3TaskRunner -> FilterGroupsTrainer).
"""
from __future__ import annotations

import os
import socket
import uuid
import warnings

import numpy as np
import torch
from tqdm import tqdm

import verl.trainer.main_ppo as mp
import verl.trainer.ppo.ray_trainer as rt
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.reward import extract_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip


class FilterGroupsTrainer(rt.RayPPOTrainer):
    """RayPPOTrainer + DAPO-style dynamic sampling (filter_groups) with a partial-batch fallback."""

    def _filter_cfg(self):
        fg = self.config.algorithm.get("filter_groups", None)
        if fg is None:
            return False, None, 0
        enable = bool(fg.get("enable", False))
        metric = fg.get("metric", None)
        max_gen = int(fg.get("max_num_gen_batches", 0) or 0)
        return enable, metric, max_gen

    @staticmethod
    def _mixed_mask(batch: DataProto, metric: str):
        """Boolean sample mask keeping prompt groups (uid) whose `metric` varies across rollouts; plus counts."""
        uids = batch.non_tensor_batch["uid"]
        vals = np.asarray(batch.non_tensor_batch[metric], dtype=np.float64)
        by_uid: dict[str, list[float]] = {}
        for u, v in zip(uids, vals):
            by_uid.setdefault(str(u), []).append(float(v))
        mixed = {u for u, vs in by_uid.items() if len(vs) > 1 and float(np.std(vs)) > 0.0}
        mask = np.array([str(u) in mixed for u in uids], dtype=bool)
        return mask, len(by_uid), len(mixed)

    def fit(self):
        """Copy of verl 0.8.0 RayPPOTrainer.fit with the filter_groups loop (see module docstring)."""
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        from omegaconf import OmegaConf
        from pprint import pprint

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        assert self.config.algorithm.adv_estimator != AdvantageEstimator.REMAX, "REMAX branch not ported in v3_trainer"
        fg_enable, fg_metric, fg_max_gen = self._filter_cfg()
        train_bs = int(self.config.data.train_batch_size)
        mini_bs = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        print(f"[filter_groups] enable={fg_enable} metric={fg_metric} max_num_gen_batches={fg_max_gen} "
              f"train_batch_size={train_bs} ppo_mini_batch_size={mini_bs} rollout_n={rollout_n}", flush=True)

        # accumulation state for filter_groups
        buffer: list[DataProto] = []
        num_gen_batches = 0
        num_prompts_generated = 0
        num_prompts_mixed = 0
        timing_raw: dict = {}
        metrics: dict = {}

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                if not buffer:
                    metrics = {}
                    timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        # NOTE: replicas are put to sleep only once this step proceeds to training (below); with
                        # filter_groups a further generation batch may follow, and generating on sleeping replicas
                        # raised "CUDA error: an illegal memory access" (smoke 63057803).
                        if curr_step_profile:
                            self.llm_server_manager.stop_profile()
                        gen_timing = gen_batch_output.meta_info.pop("timing", None)
                        if gen_timing:
                            for k, v in gen_timing.items():
                                timing_raw[k] = timing_raw.get(k, 0.0) + v if isinstance(v, (int, float)) else v
                    if "__do_sample__" in gen_batch_output.non_tensor_batch:
                        gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

                    batch = batch.repeat(repeat_times=rollout_n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = rt.compute_response_mask(batch)

                    # ---- reward first (needed by the filter) ----
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # ---- filter_groups: keep mixed groups, accumulate, maybe pull another batch ----
                    if fg_enable:
                        if fg_metric not in batch.non_tensor_batch:
                            raise KeyError(f"filter_groups.metric={fg_metric!r} not in reward extra info "
                                           f"{sorted(reward_extra_infos_dict.keys())}")
                        mask, n_groups, n_mixed = self._mixed_mask(batch, fg_metric)
                        num_gen_batches += 1
                        num_prompts_generated += n_groups
                        num_prompts_mixed += n_mixed
                        if n_mixed > 0:
                            buffer.append(batch.select_idxs(mask))
                        have = sum(len(b) for b in buffer) // rollout_n
                        print(f"[filter_groups] step={self.global_steps} gen_batch={num_gen_batches} "
                              f"mixed_in_this_batch={n_mixed}/{n_groups} accumulated_prompts={have}", flush=True)
                        if have < train_bs and (fg_max_gen <= 0 or num_gen_batches < fg_max_gen):
                            continue  # generate another batch; no optimizer step yet
                        partial = have < train_bs
                        if partial:
                            warnings.warn(
                                f"[filter_groups] max_num_gen_batches={fg_max_gen} reached with only {have} mixed prompts "
                                f"(< train_batch_size={train_bs}); proceeding with a PARTIAL batch instead of raising.",
                                stacklevel=1,
                            )
                        if not buffer:
                            print(f"[filter_groups] step={self.global_steps}: no mixed group in {num_gen_batches} gen batches "
                                  f"— skipping optimizer step", flush=True)
                            num_gen_batches = num_prompts_generated = num_prompts_mixed = 0
                            continue
                        merged = DataProto.concat(buffer)
                        merged.meta_info.update(buffer[0].meta_info)
                        # truncate to whole prompt groups: min(train_bs, largest multiple of mini_bs <= have); keep all if < mini_bs
                        keep_prompts = min(train_bs, (have // mini_bs) * mini_bs) if have >= mini_bs else have
                        uids_seq = list(dict.fromkeys(str(u) for u in merged.non_tensor_batch["uid"]))[:keep_prompts]
                        keep_set = set(uids_seq)
                        sel = np.array([str(u) in keep_set for u in merged.non_tensor_batch["uid"]], dtype=bool)
                        batch = merged.select_idxs(sel)
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)
                        metrics.update({
                            "filter/num_gen_batches": num_gen_batches,
                            "filter/num_prompts_generated": num_prompts_generated,
                            "filter/num_prompts_mixed": num_prompts_mixed,
                            "filter/num_prompts_trained": keep_prompts,
                            "filter/mixed_frac": num_prompts_mixed / max(1, num_prompts_generated),
                            "filter/partial": int(partial),
                        })
                        print(f"[filter_groups] step={self.global_steps} gen_batches={num_gen_batches} "
                              f"mixed={num_prompts_mixed}/{num_prompts_generated} trained={keep_prompts} partial={int(partial)}",
                              flush=True)
                        buffer = []
                        num_gen_batches = num_prompts_generated = num_prompts_mixed = 0

                    # free rollout weights + KV cache before the actor/ref passes (original fit does this right after gen)
                    self.checkpoint_manager.sleep_replicas()

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            metrics.update({"actor/entropy": entropy_agg.detach().item(), "perf/mfu/actor_infer": old_log_prob_mfu})
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        with marked_timer(str(rt.Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = rt.apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = rt.compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=rollout_n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                    if self.config.trainer.critic_warmup > self.global_steps:
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(rt.compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(rt.compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(rt.compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(rt.compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                metrics.update(
                    rt.compute_spec_decode_metrics(
                        batch.non_tensor_batch.get("spec_num_draft_tokens", None),
                        batch.non_tensor_batch.get("spec_num_accepted_tokens", None),
                        batch.non_tensor_batch.get("spec_num_verify_steps", None),
                    )
                )
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    self._shutdown_dump_executor()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)

        self._shutdown_dump_executor()


class V3TaskRunner(mp.TaskRunner):
    """main_ppo.TaskRunner with FilterGroupsTrainer instead of RayPPOTrainer (run() copied from verl 0.8.0)."""

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f"V3TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        mp.validate_config(
            config=config,
            use_reference_policy=mp.need_reference_policy(config),
            use_critic=mp.need_critic(config),
        )
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        resource_pool_manager = self.init_resource_pool_mgr(config)
        train_dataset = mp.create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True,
                                             max_samples=config.data.get("train_max_samples", -1))
        val_dataset = mp.create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False,
                                           max_samples=config.data.get("val_max_samples", -1))
        train_sampler = mp.create_rl_sampler(config.data, train_dataset)
        trainer = FilterGroupsTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()
