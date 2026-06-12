"""Download Visual Genome images that are not already on disk.

Uses the per-image `url` in image_data.json (which encodes VG_100K vs
VG_100K_2), skips images already present, downloads in parallel with retries,
and logs any failures. Idempotent: safe to re-run to fill gaps.

Usage:
    python tools/fetch_vg_images.py \
        --image-data data/visual_genome/raw/image_data.json \
        --out-dir data/images \
        --workers 24
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def download_one(rec, out_dir, retries=3, timeout=30):
    image_id = rec["image_id"]
    url = rec["url"]
    dest = out_dir / f"{image_id}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return ("skip", image_id)
    tmp = out_dir / f".{image_id}.jpg.part"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            if r.status_code == 200:
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
                return ("ok", image_id)
            elif r.status_code == 404:
                return ("404", image_id)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return ("fail", image_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-data", default="data/visual_genome/raw/image_data.json")
    ap.add_argument("--out-dir", default="data/images")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = json.load(open(args.image_data))
    have = {int(f[:-4]) for f in os.listdir(out_dir) if f.endswith(".jpg")}
    todo = [r for r in recs if r["image_id"] not in have]
    print(f"[vg-img] total VG images: {len(recs)} | already have: {len(have)} | "
          f"to download: {len(todo)}", flush=True)

    counts = {"ok": 0, "skip": 0, "404": 0, "fail": 0}
    failures = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, r, out_dir) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            status, image_id = fut.result()
            counts[status] += 1
            if status in ("fail", "404"):
                failures.append((image_id, status))
            if i % 1000 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-6)
                print(f"[vg-img] {i}/{len(todo)}  ok={counts['ok']} "
                      f"404={counts['404']} fail={counts['fail']}  "
                      f"{rate:.0f} img/s", flush=True)

    if failures:
        flog = out_dir.parent / "vg_image_failures.json"
        json.dump(failures, open(flog, "w"))
        print(f"[vg-img] wrote {len(failures)} failures -> {flog}", flush=True)

    print(f"[vg-img] DONE in {time.time()-t0:.0f}s | {counts}", flush=True)
    # Non-zero exit only on hard failures (404s are expected dead links).
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
