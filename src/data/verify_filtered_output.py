#!/usr/bin/env python
"""Mandatory invariant checks over the filtered VisualPRM outputs.

1. Verbatim steps: every output trace's `steps` == the raw trace's body steps
   (raw looked up via source_file + line_index; per-step string equality).
   Checked for stage1_kept (both taus), stage2_kept (both taus), the band
   sample, sft_train.jsonl and rlvr_prompts.jsonl — via a single streaming
   merge-join over the raw annotations (all outputs are written in
   (source_file, line_index) order; nothing is loaded wholesale).
2. No duplicate (image-content-key, normalized question) in sft_train.jsonl.
3. Every output `image` ref resolves in images.zip via the path translation.
4. Row counts consistent: stage1 >= stage2 kept, stage2 kept+rejected ==
   stage1 kept, emitted sft <= stage2 kept, rlvr <= sft.
5. stage1 tau=0.85 stats match the known numbers (kept 338,685;
   answer-correct 503,885; total 565,149).

Exit code 0 iff every check passes; each check prints PASS/FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

FILTERED = Path("/scratch/sghos104/rlpt/data/visualprm_v11_filtered")
RAW = Path("/scratch/sghos104/rlpt/data/visualprm_v11_raw")
RAW_PREFIX = "VisualPRM400K-v1.1-Raw/"
KNOWN = {"total_rows": 565149, "answer_correct": 503885, "kept": 338685}


def iter_jsonl(path: Path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def candidate_members(ref: str) -> list[str]:
    cands = []
    if ref.startswith(RAW_PREFIX):
        cands.append("images/" + ref[len(RAW_PREFIX):])
    elif ref.startswith("images/"):
        cands.append(ref)
        cands.append("images/nlvr2/" + ref)
    else:
        cands.append("images/" + ref)
    return cands


def normalize_question(q: str) -> str:
    return " ".join(str(q).split()).casefold()


class Check:
    def __init__(self) -> None:
        self.failures = 0

    def report(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures += 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filtered_dir", type=Path, default=FILTERED)
    p.add_argument("--raw_dir", type=Path, default=RAW)
    p.add_argument("--taus", nargs="+", default=["0.85", "0.90"])
    args = p.parse_args()
    d = args.filtered_dir
    annos = args.raw_dir / "annos"
    ck = Check()

    # ---- Check 1: verbatim steps via one streaming merge-join over raw ----
    outputs = [d / "sft_train.jsonl", d / "rlvr_prompts.jsonl",
               d / "stage1_band_sample.jsonl"]
    for tau in args.taus:
        outputs += [d / f"stage1_kept_tau{tau}.jsonl", d / f"stage2_kept_tau{tau}.jsonl"]
    outputs = [o for o in outputs if o.exists()]
    streams = [{"path": o, "it": iter_jsonl(o), "head": None, "n": 0,
                "mismatch": 0, "missing": 0} for o in outputs]
    for s in streams:
        s["head"] = next(s["it"], None)

    raw_files = sorted((annos / "annotations").glob("*.jsonl"))
    for f in raw_files:
        source_file = f"annotations/{f.name}"
        with open(f) as fh:
            for line_index, line in enumerate(fh):
                tid = (source_file, line_index)
                hot = [s for s in streams if s["head"] is not None
                       and (s["head"]["source_file"], s["head"]["line_index"]) == tid]
                if not hot:
                    continue
                rec = json.loads(line)
                sws = rec.get("steps_with_score") or []
                body = [x["step"] for x in (sws[:-1] if len(sws) > 1 else sws)]
                for s in hot:
                    if s["head"]["steps"] != body:
                        s["mismatch"] += 1
                    s["n"] += 1
                    s["head"] = next(s["it"], None)
    for s in streams:
        # any rows left unconsumed reference (source_file, line_index) pairs
        # that don't exist in raw order — count them as missing
        while s["head"] is not None:
            s["missing"] += 1
            s["head"] = next(s["it"], None)
        ck.report(f"verbatim steps: {s['path'].name}",
                  s["mismatch"] == 0 and s["missing"] == 0,
                  f"{s['n']} rows checked, {s['mismatch']} mismatched, "
                  f"{s['missing']} not found in raw")

    # ---- Checks 2 + 3: dedup key uniqueness and image resolution ----
    with zipfile.ZipFile(args.raw_dir / "images.zip") as z:
        member_key = {i.filename: (i.CRC, i.file_size) for i in z.infolist()}

    def image_key(image):
        refs = image if isinstance(image, list) else [image]
        keys = []
        for ref in refs:
            for cand in candidate_members(ref):
                if cand in member_key:
                    keys.append(member_key[cand])
                    break
            else:
                return None
        return tuple(keys)

    seen: set = set()
    dupes = unresolved = n_rows = 0
    for row in iter_jsonl(d / "sft_train.jsonl"):
        n_rows += 1
        ik = image_key(row["image"])
        if ik is None:
            unresolved += 1
            continue
        gk = (ik, normalize_question(row["question_orig"]))
        if gk in seen:
            dupes += 1
        seen.add(gk)
    ck.report("no duplicate (image-key, question) in sft_train.jsonl",
              dupes == 0, f"{n_rows} rows, {dupes} duplicate keys")
    unresolved_rlvr = sum(1 for row in iter_jsonl(d / "rlvr_prompts.jsonl")
                          if image_key(row["image"]) is None)
    ck.report("all image refs resolve in images.zip",
              unresolved == 0 and unresolved_rlvr == 0,
              f"sft unresolved={unresolved}, rlvr unresolved={unresolved_rlvr}")

    # ---- Check 4: count consistency across stats files ----
    for tau in args.taus:
        s1 = json.loads((d / f"stage1_stats_tau{tau}.json").read_text())
        s2 = json.loads((d / f"stage2_stats_tau{tau}.json").read_text())
        ok = (s1["kept"] == s2["kept"] + s2["rejected"] and s2["kept"] <= s1["kept"])
        ck.report(f"counts: stage1 kept == stage2 kept+rejected (tau={tau})", ok,
                  f"stage1 {s1['kept']} vs stage2 {s2['kept']}+{s2['rejected']}")
    em = json.loads((d / "emission_stats.json").read_text())
    s2_85 = json.loads((d / "stage2_stats_tau0.85.json").read_text())
    ok = (em["sft_rows"] <= s2_85["kept"] and em["rlvr_rows"] <= em["sft_rows"]
          and em["sft_rows"] == em["n_groups_image_question"]
          and em["rlvr_rows"] == em["n_unique_questions"])
    ck.report("counts: stage2 kept >= sft rows >= rlvr rows (and == group counts)",
              ok, f"stage2 {s2_85['kept']} >= sft {em['sft_rows']} >= rlvr {em['rlvr_rows']}")

    # ---- Check 5: stage1 tau=0.85 known numbers ----
    s1_85 = json.loads((d / "stage1_stats_tau0.85.json").read_text())
    ok = all(s1_85[k] == v for k, v in KNOWN.items())
    ck.report("stage1 tau=0.85 matches known numbers", ok,
              ", ".join(f"{k}={s1_85[k]} (expect {v})" for k, v in KNOWN.items()))

    print(f"\n{'ALL CHECKS PASSED' if ck.failures == 0 else f'{ck.failures} CHECK(S) FAILED'}")
    return 0 if ck.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
