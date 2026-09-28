#!/usr/bin/env python
"""Stage 4: dedup Stage-2 survivors and emit SFT / RLVR training files.

Grouping key = (image-content-key, normalized question_orig), where the image
content key is (CRC32, file_size) of the translated images.zip member (from
zipfile infolist — nothing is extracted). List-typed image fields use the
tuple of member keys. Question normalization = whitespace-collapse + casefold
(`<image>` placeholders left in place; differing image content is already
separated by the image key).

One trace kept per group: highest min_score, then higher mean_score, then
fewest steps, then lowest (source_file, line_index).

rlvr_prompts.jsonl keeps one row per unique normalized question (chosen among
the deduped winners with the same tie-break) — the `steps` field is the
bipartite-match reference.

Streaming + two passes over the input (selection metadata only in memory,
never the full corpus). Output rows are written in input order, which is
(source_file, line_index)-sorted by construction — deterministic.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

FILTERED = Path("/scratch/sghos104/rlpt/data/visualprm_v11_filtered")
IMAGES_ZIP = Path("/scratch/sghos104/rlpt/data/visualprm_v11_raw/images.zip")
RAW_PREFIX = "VisualPRM400K-v1.1-Raw/"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage2_kept", type=Path,
                   default=FILTERED / "stage2_kept_tau0.85.jsonl")
    p.add_argument("--out_dir", type=Path, default=FILTERED)
    p.add_argument("--images_zip", type=Path, default=IMAGES_ZIP)
    p.add_argument("--drop_multi_image", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Exclude records whose image field is a list (nlvr2)")
    return p.parse_args()


def iter_jsonl(path: Path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def candidate_members(ref: str) -> list[str]:
    """Validated annotation-ref -> images.zip member mapping (same as
    resolve_zip_member in src/train/train_common.py, replicated so this
    script stays torch-free)."""
    cands = []
    if ref.startswith(RAW_PREFIX):
        cands.append("images/" + ref[len(RAW_PREFIX):])
    elif ref.startswith("images/"):
        cands.append(ref)
        cands.append("images/nlvr2/" + ref)  # nlvr2 short style nests deeper
    else:
        cands.append("images/" + ref)
    return cands


def normalize_question(q: str) -> str:
    return " ".join(str(q).split()).casefold()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.images_zip) as z:
        member_key = {i.filename: (i.CRC, i.file_size) for i in z.infolist()}

    def image_key(image) -> tuple | None:
        refs = image if isinstance(image, list) else [image]
        keys = []
        for ref in refs:
            for cand in candidate_members(ref):
                if cand in member_key:
                    keys.append(member_key[cand])
                    break
            else:
                return None  # unresolvable — caller decides
        return tuple(keys)

    # Pass 1: pick winners. Selection tuple sorts ascending = better.
    def rank(row: dict) -> tuple:
        return (-row["min_score"], -row["mean_score"], row["n_steps"],
                row["source_file"], row["line_index"])

    groups: dict[tuple, tuple] = {}           # group key -> (rank, trace id)
    n_in = n_multi_dropped = n_unresolved = 0
    for row in iter_jsonl(args.stage2_kept):
        n_in += 1
        if args.drop_multi_image and isinstance(row["image"], list):
            n_multi_dropped += 1
            continue
        ik = image_key(row["image"])
        if ik is None:
            n_unresolved += 1
            print(f"FATAL: unresolvable image {row['image']!r} at "
                  f"{row['source_file']}:{row['line_index']}", file=sys.stderr)
            return 2
        gk = (ik, normalize_question(row["question_orig"]))
        r = rank(row)
        if gk not in groups or r < groups[gk][0]:
            groups[gk] = (r, (row["source_file"], row["line_index"]))
    sft_ids = {tid for _, tid in groups.values()}

    # RLVR: unique normalized question among the SFT winners, same tie-break.
    q_groups: dict[str, tuple] = {}
    for (ik, nq), (r, tid) in groups.items():
        if nq not in q_groups or r < q_groups[nq][0]:
            q_groups[nq] = (r, tid)
    rlvr_ids = {tid for _, tid in q_groups.values()}

    # Pass 2: stream again, write winners in (deterministic) input order.
    emit_fields = ("image", "question_orig", "steps", "answer",
                   "source_file", "line_index")
    n_sft = n_rlvr = 0
    per_source_sft: dict[str, int] = {}
    per_source_rlvr: dict[str, int] = {}
    with open(args.out_dir / "sft_train.jsonl", "w") as fs, \
         open(args.out_dir / "rlvr_prompts.jsonl", "w") as fr:
        for row in iter_jsonl(args.stage2_kept):
            tid = (row["source_file"], row["line_index"])
            if tid not in sft_ids:
                continue
            out = json.dumps({k: row[k] for k in emit_fields}, ensure_ascii=False)
            fs.write(out + "\n")
            n_sft += 1
            per_source_sft[row["source_file"]] = \
                per_source_sft.get(row["source_file"], 0) + 1
            if tid in rlvr_ids:
                fr.write(out + "\n")
                n_rlvr += 1
                per_source_rlvr[row["source_file"]] = \
                    per_source_rlvr.get(row["source_file"], 0) + 1

    stats = {
        "input": str(args.stage2_kept),
        "input_rows": n_in,
        "drop_multi_image": args.drop_multi_image,
        "multi_image_dropped": n_multi_dropped,
        "question_normalization": "whitespace-collapse + casefold",
        "n_groups_image_question": len(groups),
        "n_unique_questions": len(q_groups),
        "sft_rows": n_sft,
        "rlvr_rows": n_rlvr,
        "per_source_sft": dict(sorted(per_source_sft.items())),
        "per_source_rlvr": dict(sorted(per_source_rlvr.items())),
    }
    with open(args.out_dir / "emission_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps({k: v for k, v in stats.items()
                      if not k.startswith("per_source")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
