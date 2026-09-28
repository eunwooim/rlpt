#!/usr/bin/env python
"""Acquire OpenGVLab/VisualProcessBench into data/visualprocessbench/ and
verify every downloaded file against the SHA256 the Hub reports.

Verification: for LFS-tracked files the Hub stores the sha256 of the content
as the LFS oid, so the check is a real end-to-end integrity + identity test
(computed digest vs remote-declared digest). Small non-LFS files are not
LFS-tracked and the Hub only exposes a git blob sha1 for them; those are
recorded (sha256 computed locally, git sha1 recomputed and compared) but
flagged separately so nothing is claimed as verified that wasn't.

Writes vpb_acquisition.json + prints a PASS/FAIL table. Exits non-zero if
any LFS file mismatches.
"""
import hashlib
import json
import os
import sys

REPO = "OpenGVLab/VisualProcessBench"
OUTDIR = "/scratch/sghos104/rlpt/data/visualprocessbench"
MANIFEST = "/scratch/sghos104/rlpt/data/visualprocessbench/vpb_acquisition.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_blob_sha1(path):
    """git object id for a blob: sha1("blob <len>\\0" + content)."""
    size = os.path.getsize(path)
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def remote_digests():
    """path -> {"lfs_sha256": str|None, "blob_id": str|None, "size": int}"""
    from huggingface_hub import HfApi
    api = HfApi()
    out = {}
    for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True):
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
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"[vpb] listing {REPO} ...", flush=True)
    remote = remote_digests()
    print(f"[vpb] remote files: {len(remote)}", flush=True)

    local_root = snapshot_download(REPO, repo_type="dataset", local_dir=OUTDIR)
    print(f"[vpb] snapshot -> {local_root}", flush=True)

    rows, n_lfs_ok, n_lfs_bad, n_unverified = [], 0, 0, 0
    for rel, meta in sorted(remote.items()):
        path = os.path.join(local_root, rel)
        if not os.path.exists(path):
            rows.append({"path": rel, "status": "MISSING"})
            n_lfs_bad += 1
            continue
        got = sha256_of(path)
        size = os.path.getsize(path)
        row = {"path": rel, "bytes": size, "sha256": got,
               "remote_size": meta["size"]}
        if meta["lfs_sha256"]:
            ok = (got == meta["lfs_sha256"])
            row["remote_lfs_sha256"] = meta["lfs_sha256"]
            row["status"] = "PASS" if ok else "SHA256_MISMATCH"
            n_lfs_ok += ok
            n_lfs_bad += (not ok)
        else:
            gs = git_blob_sha1(path)
            ok = (meta["blob_id"] is None) or (gs == meta["blob_id"])
            row["git_blob_sha1"] = gs
            row["remote_blob_id"] = meta["blob_id"]
            row["status"] = ("PASS_GIT_SHA1" if ok else "GIT_SHA1_MISMATCH")
            row["note"] = "not LFS-tracked; Hub exposes no sha256"
            n_unverified += 1
            n_lfs_bad += (not ok)
        rows.append(row)

    # row counts for the jsonl/json data files
    for r in rows:
        p = os.path.join(local_root, r["path"])
        if r["path"].endswith((".jsonl", ".json")) and os.path.exists(p):
            try:
                if r["path"].endswith(".jsonl"):
                    with open(p) as f:
                        r["rows"] = sum(1 for line in f if line.strip())
                else:
                    with open(p) as f:
                        obj = json.load(f)
                    r["rows"] = len(obj) if isinstance(obj, list) else None
            except Exception as e:
                r["rows_error"] = str(e)

    manifest = {"repo": REPO, "local_dir": local_root, "files": rows,
                "summary": {"lfs_verified": n_lfs_ok, "failures": n_lfs_bad,
                            "non_lfs_recorded": n_unverified}}
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'status':<18} {'bytes':>13}  path")
    for r in rows:
        print(f"{r.get('status','?'):<18} {r.get('bytes',0):>13,}  {r['path']}"
              + (f"  rows={r['rows']:,}" if isinstance(r.get("rows"), int) else ""))
    print(f"\n[vpb] LFS sha256 verified: {n_lfs_ok} | non-LFS (git sha1): "
          f"{n_unverified} | failures: {n_lfs_bad}")
    print(f"[vpb] manifest -> {MANIFEST}")
    if n_lfs_bad:
        print("[vpb] FAILED")
        return 1
    print("[vpb] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
