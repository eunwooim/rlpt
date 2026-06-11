#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_str(x):
    if isinstance(x, bool):
        return "True" if x else "False"
    return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    project_root = Path(cfg["project_root"])
    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["out_dir"])

    train_file = cfg.get("train_file") or str(data_dir / "train.parquet")
    val_file = cfg.get("val_file") or str(data_dir / "val.parquet")

    if not Path(train_file).exists():
        raise FileNotFoundError(f"Missing train file: {train_file}")
    if not Path(val_file).exists():
        raise FileNotFoundError(f"Missing val file: {val_file}")

    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = out_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "python",
            str(project_root / "train" / "snapshot_code.py"),
            "--src_root",
            str(project_root),
            "--out_dir",
            str(snapshot_dir),
        ],
        check=True,
    )
    (out_dir / "wandb").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_visible_devices"])
    env["HF_HOME"] = str(cfg["hf_home"])
    env["HUGGINGFACE_HUB_CACHE"] = str(Path(cfg["hf_home"]) / "hub")
    env["TRANSFORMERS_CACHE"] = str(Path(cfg["hf_home"]) / "hub")
    env["WANDB_DIR"] = str(out_dir / "wandb")
    env["HYDRA_FULL_ERROR"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    if cfg.get("suppress_transformers_warnings", True):
        env["TRANSFORMERS_VERBOSITY"] = "error"
        env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    overrides = [
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        "data.messages_key=messages",
        f"data.train_batch_size={cfg['train_batch_size']}",
        f"data.micro_batch_size_per_gpu={cfg['micro_batch_size_per_gpu']}",
        f"data.max_length={cfg['max_length']}",
        f"data.num_workers={cfg['num_workers']}",

        f"optim.lr={cfg['learning_rate']}",

        "engine=fsdp",
        "engine.use_torch_compile=False",
        f"trainer.n_gpus_per_node={cfg['num_gpus']}",

        f"model.path={cfg['model_id']}",
        f"model.tokenizer_path={cfg['model_id']}",
        "model.trust_remote_code=True",
        "++model.override_config.attn_implementation=eager",
        "++model.override_config._attn_implementation=eager",
        f"model.lora_rank={cfg['lora_rank']}",
        f"model.lora_alpha={cfg['lora_alpha']}",
        "model.target_modules=all-linear",
        "model.use_remove_padding=False",

        f"trainer.default_local_dir={out_dir / 'checkpoints'}",
        f"trainer.project_name={cfg['wandb_project']}",
        f"trainer.experiment_name={cfg['wandb_name']}",
        f"trainer.total_epochs={cfg['total_epochs']}",
        f"trainer.save_freq={cfg['save_freq']}",
        f"trainer.test_freq={cfg['test_freq']}",
        "trainer.logger=['console','wandb']",
        f"trainer.max_ckpt_to_keep={cfg['max_ckpt_to_keep']}",
        f"trainer.resume_mode={cfg['resume_mode']}",

        "checkpoint.save_contents=[model,optimizer,extra,hf_model]",
    ]

    overrides.extend(args.overrides)

    print("=" * 80)
    print("Launching veRL SFT")
    print("=" * 80)
    for k in [
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "WANDB_DIR",
        "TRANSFORMERS_VERBOSITY",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS",
    ]:
        print(f"{k}={env.get(k)}")

    print("\nHydra overrides:")
    for item in overrides:
        print(f"  {item}")
    print("=" * 80)

    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={cfg['num_gpus']}",
        "-m",
        "verl.trainer.sft_trainer",
        *overrides,
    ]

    subprocess.run(cmd, cwd=str(project_root), env=env, check=True)


if __name__ == "__main__":
    main()
