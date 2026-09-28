#!/usr/bin/env python
"""Acquire Qwen2.5-VL {3B,7B}-Instruct into the project HF cache and verify
every weight/config file against the SHA256 the Hub reports.

Same verification contract as src/process_reward/download_vpb.py: LFS-tracked
files (all safetensors shards) carry a content sha256 as their LFS oid, so
those get a true remote-vs-local identity check. Small non-LFS files (configs,
tokenizer json) expose only a git blob sha1 upstream — recomputed and compared,
recorded as PASS_GIT_SHA1 so nothing is claimed as sha256-verified that isn't.

Writes qwen_acquisition.json next to the repo checkouts. Exits non-zero on any
mismatch or missing file.
"""
import hashlib
import json
import os
import sys

REPOS = ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]
HF_HOME = "/scratch/sghos104/rlpt/data/hf_cache"
MANIFEST = "/scratch/sghos104/rlpt/models/qwen_acquisition.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def git_blob_sha1(path):
    size = os.path.getsize(path)
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def remote_digests(repo):
    from huggingface_hub import HfApi
    api = HfApi()
    out = {}
    for f in api.list_repo_tree(repo, recursive=True):
        if type(f).__name__ != "RepoFile":
            continue
        lfs = getattr(f, "lfs", None)
        sha = None
        if lfs is not None:
            sha = (getattr(lfs, "sha256", None)
                   or (lfs.get("sha256") if isinstance(lfs, dict) else None)
                   or getattr(lfs, "oid", None)
                   or (lfs.get("oid") if isinstance(lfs, dict) else None))
        out[f.path] = {"lfs_sha256": sha,
                       "blob_id": getattr(f, "blob_id", None),
                       "size": getattr(f, "size", None)}
    return out


def main():
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    manifest, failures = {}, 0

    for repo in REPOS:
        print(f"\n[qwen] ===== {repo} =====", flush=True)
        remote = remote_digests(repo)
        print(f"[qwen] remote files: {len(remote)}", flush=True)
        root = snapshot_download(repo)          # into HF_HOME cache
        print(f"[qwen] snapshot -> {root}", flush=True)

        rows, n_lfs, n_git, n_bad = [], 0, 0, 0
        for rel, meta in sorted(remote.items()):
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                rows.append({"path": rel, "status": "MISSING"})
                n_bad += 1
                continue
            got = sha256_of(path)
            row = {"path": rel, "bytes": os.path.getsize(path), "sha256": got,
                   "remote_size": meta["size"]}
            if meta["lfs_sha256"]:
                ok = got == meta["lfs_sha256"]
                row["remote_lfs_sha256"] = meta["lfs_sha256"]
                row["status"] = "PASS" if ok else "SHA256_MISMATCH"
                n_lfs += ok
                n_bad += (not ok)
            else:
                gs = git_blob_sha1(path)
                ok = (meta["blob_id"] is None) or (gs == meta["blob_id"])
                row["git_blob_sha1"] = gs
                row["remote_blob_id"] = meta["blob_id"]
                row["status"] = "PASS_GIT_SHA1" if ok else "GIT_SHA1_MISMATCH"
                row["note"] = "not LFS-tracked; Hub exposes no sha256"
                n_git += 1
                n_bad += (not ok)
            rows.append(row)

        total = sum(r.get("bytes", 0) for r in rows)
        manifest[repo] = {"local_dir": root, "files": rows,
                          "total_bytes": total,
                          "summary": {"lfs_sha256_verified": n_lfs,
                                      "non_lfs_git_sha1": n_git,
                                      "failures": n_bad}}
        failures += n_bad
        for r in rows:
            print(f"  {r.get('status','?'):<18} {r.get('bytes',0):>14,}  {r['path']}")
        print(f"[qwen] {repo}: {total/2**30:.2f} GiB | lfs sha256 ok={n_lfs} "
              f"git sha1 ok={n_git} failures={n_bad}", flush=True)

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[qwen] manifest -> {MANIFEST}")
    if failures:
        print(f"[qwen] FAILED ({failures} bad files)")
        return 1
    print("[qwen] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
