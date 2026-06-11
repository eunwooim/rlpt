#!/usr/bin/env python3

import argparse
import os
import subprocess
import tarfile
from pathlib import Path


def git_commit_hash(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def make_tarball(src_root: Path, out_path: Path) -> None:
    exclude_dirs = {
        ".git",
        "__pycache__",
        ".cache",
        "wandb",
        "outputs",
        "data",
        "env_logs",
    }

    include_suffixes = {
        ".py",
        ".bash",
        ".sh",
        ".yaml",
        ".yml",
        ".md",
        ".txt",
    }

    with tarfile.open(out_path, "w:gz") as tar:
        for path in src_root.rglob("*"):
            rel = path.relative_to(src_root)
            if any(part in exclude_dirs for part in rel.parts):
                continue
            if path.is_file() and path.suffix in include_suffixes:
                tar.add(path, arcname=str(rel))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = git_commit_hash(src_root)
    tar_path = out_dir / f"code_snapshot_{commit}.tar.gz"
    make_tarball(src_root, tar_path)

    meta_path = out_dir / "code_snapshot_meta.txt"
    meta_path.write_text(
        f"src_root={src_root}\ncommit={commit}\ntarball={tar_path}\n",
        encoding="utf-8",
    )

    print(tar_path)


if __name__ == "__main__":
    main()
