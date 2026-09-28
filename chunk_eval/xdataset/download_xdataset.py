#!/usr/bin/env python
"""Acquire PRM800K (openai/prm800k GitHub, phase2 train/test jsonl) and
ProcessBench (Qwen/ProcessBench, HF dataset) into data/xdataset_eval/.

EVAL-ONLY data: nothing from here is ever written into a training directory.

For each downloaded file: SHA256, byte size, line/row count, and the first
record (pretty-printed, truncated) are recorded in acquisition.json.
PRM800K files on GitHub are stored via git-lfs — the raw URL may serve a
tiny LFS pointer file; we detect that and fall back to the media URL.
"""
import hashlib
import json
import os
import sys

OUTDIR = "/scratch/sghos104/rlpt/data/xdataset_eval"
PRM_FILES = ["phase2_train.jsonl", "phase2_test.jsonl"]
PRM_URLS = [
    "https://github.com/openai/prm800k/raw/main/prm800k/data/{f}",
    "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/{f}",
    "https://raw.githubusercontent.com/openai/prm800k/main/prm800k/data/{f}",
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def is_lfs_pointer(path):
    with open(path, "rb") as f:
        head = f.read(200)
    return head.startswith(b"version https://git-lfs")


def dl(url, dest):
    import requests
    print(f"[dl] {url} -> {dest}", flush=True)
    with requests.get(url, stream=True, timeout=120,
                      allow_redirects=True) as r:
        r.raise_for_status()
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            for blk in r.iter_content(1 << 20):
                f.write(blk)
        os.replace(tmp, dest)


def jsonl_summary(path, limit_example_chars=3000):
    n = 0
    first = None
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            if first is None:
                first = json.loads(line)
            n += 1
    ex = json.dumps(first, indent=1)[:limit_example_chars]
    return n, ex


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    manifest = {"prm800k": {}, "processbench": {}}

    # ---------------- PRM800K ----------------
    prm_dir = os.path.join(OUTDIR, "prm800k")
    os.makedirs(prm_dir, exist_ok=True)
    for fname in PRM_FILES:
        dest = os.path.join(prm_dir, fname)
        if not (os.path.exists(dest) and os.path.getsize(dest) > 1_000_000):
            ok = False
            for u in PRM_URLS:
                try:
                    dl(u.format(f=fname), dest)
                except Exception as e:
                    print(f"[dl] FAILED {u.format(f=fname)}: {e}", flush=True)
                    continue
                if is_lfs_pointer(dest):
                    print(f"[dl] {fname}: got LFS pointer, trying next URL",
                          flush=True)
                    continue
                ok = True
                break
            if not ok:
                raise SystemExit(f"[dl] could not fetch {fname}")
        n, ex = jsonl_summary(dest)
        manifest["prm800k"][fname] = {
            "path": dest, "bytes": os.path.getsize(dest),
            "sha256": sha256_of(dest), "rows": n, "first_record": ex}
        print(f"[prm800k] {fname}: rows={n:,} "
              f"bytes={os.path.getsize(dest):,}", flush=True)

    # ---------------- ProcessBench ----------------
    pb_dir = os.path.join(OUTDIR, "processbench")
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/ProcessBench", repo_type="dataset",
                      local_dir=pb_dir)
    data_files = []
    for root, _dirs, files in os.walk(pb_dir):
        for f in files:
            if f.endswith((".json", ".jsonl", ".parquet")):
                data_files.append(os.path.join(root, f))
    print(f"[processbench] data files: {sorted(data_files)}", flush=True)
    for path in sorted(data_files):
        rel = os.path.relpath(path, pb_dir)
        entry = {"path": path, "bytes": os.path.getsize(path),
                 "sha256": sha256_of(path)}
        if path.endswith(".jsonl"):
            n, ex = jsonl_summary(path)
            entry["rows"], entry["first_record"] = n, ex
        elif path.endswith(".json"):
            with open(path) as f:
                obj = json.load(f)
            if isinstance(obj, list):
                entry["rows"] = len(obj)
                entry["first_record"] = json.dumps(obj[0], indent=1)[:3000]
        elif path.endswith(".parquet"):
            import pyarrow.parquet as pq
            t = pq.read_table(path)
            entry["rows"] = t.num_rows
            entry["first_record"] = json.dumps(
                {k: t.column(k)[0].as_py() for k in t.column_names},
                indent=1, default=str)[:3000]
        manifest["processbench"][rel] = entry
        print(f"[processbench] {rel}: rows={entry.get('rows','?')} "
              f"bytes={entry['bytes']:,}", flush=True)

    out = "/scratch/sghos104/rlpt/chunk_eval/xdataset/acquisition.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[dl] manifest -> {out}")
    print("[dl] DONE")


if __name__ == "__main__":
    sys.exit(main())
