#!/usr/bin/env python
"""Step-0 profiling for the DeBERTa-v3-small chunk-boundary classifier.

Read-only. One streaming pass over canonical.jsonl to:
  * confirm the steps-per-record distribution,
  * stratified-reservoir-sample multi-step and single-step records,
  * count source_sample_id AND question-text multiplicity (leakage axes),
  * check schema integrity (step_labels alignment, empty fields),
then tokenize the sample with the DeBERTa-v3 tokenizer to report:
  * token-length distributions of joined trajectories and of individual steps
    (broken out for the 6+ bucket, which drives sliding-window volume),
  * sliding-window count estimate (window=512, stride=384),
  * fraction of non-final steps ending in sentence punctuation,
  * whether single-step texts are duplicated as steps of multi-step records.

Nothing is written except a JSON report + optional sampled JSONL for reuse.
"""
import argparse, json, os, random, re, hashlib, math
from collections import Counter, defaultdict

CANONICAL = "/scratch/sghos104/rlpt/canonical.jsonl"
MODEL = "microsoft/deberta-v3-small"
WINDOW, STRIDE = 512, 384
PUNCT_END = set(".!?;:")  # newline handled separately

STEPMARK = re.compile(r"(?im)^\s*step\s*\d+\s*[:.\)]")  # "Step 3:" style markers


def qhash(q):
    return hashlib.blake2b(q.encode("utf-8", "ignore"), digest_size=12).hexdigest()


def norm_text(s):
    return re.sub(r"\s+", " ", s).strip()


def windows_for_len(L):
    if L <= WINDOW:
        return 1
    return math.ceil((L - WINDOW) / (WINDOW - STRIDE)) + 1


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def dist_summary(vals):
    if not vals:
        return {}
    v = sorted(vals)
    return {
        "n": len(v), "min": v[0], "max": v[-1],
        "mean": round(sum(v) / len(v), 1),
        "p50": pct(v, .50), "p90": pct(v, .90),
        "p95": pct(v, .95), "p99": pct(v, .99),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=CANONICAL)
    ap.add_argument("--multi_sample", type=int, default=60000)
    ap.add_argument("--single_sample", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_lines", type=int, default=0, help="0 = full file; >0 for a quick dry run")
    ap.add_argument("--out_json", default="/scratch/sghos104/rlpt/chunker/profile_report.json")
    ap.add_argument("--out_sample", default="/scratch/sghos104/rlpt/chunker/profile_sample.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- streaming pass ----
    n_total = 0
    nstep_hist = Counter()
    ssid_counts = defaultdict(int)
    q_counts = defaultdict(int)
    steplabel_vals = Counter()
    n_bad_label_len = 0
    n_empty_images = 0
    n_empty_answer = 0
    n_empty_step = 0
    n_multi_with_marker = 0
    n_single_with_marker = 0

    multi_res, single_res = [], []      # reservoirs
    multi_seen = single_seen = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if args.max_lines and n_total >= args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_total += 1
            steps = r.get("steps") or []
            n = len(steps)
            bucket = n if n <= 5 else 6
            nstep_hist[bucket] += 1

            md = r.get("metadata") or {}
            ssid = md.get("source_sample_id")
            if ssid is not None:
                ssid_counts[ssid] += 1
            q = r.get("question") or ""
            q_counts[qhash(q)] += 1

            sl = r.get("step_labels") or []
            for v in sl:
                steplabel_vals[v] += 1
            if len(sl) != n:
                n_bad_label_len += 1
            if not (r.get("images")):
                n_empty_images += 1
            if not (r.get("answer")):
                n_empty_answer += 1
            if any((s is None or s == "") for s in steps):
                n_empty_step += 1

            has_marker = any(STEPMARK.search(s or "") for s in steps)
            if n >= 2:
                if has_marker:
                    n_multi_with_marker += 1
                # reservoir (multi)
                multi_seen += 1
                if len(multi_res) < args.multi_sample:
                    multi_res.append({"steps": steps, "q": q, "ssid": ssid})
                else:
                    j = rng.randint(0, multi_seen - 1)
                    if j < args.multi_sample:
                        multi_res[j] = {"steps": steps, "q": q, "ssid": ssid}
            else:
                if has_marker:
                    n_single_with_marker += 1
                single_seen += 1
                text = steps[0] if steps else ""
                if len(single_res) < args.single_sample:
                    single_res.append({"steps": steps, "q": q, "ssid": ssid, "text": text})
                else:
                    j = rng.randint(0, single_seen - 1)
                    if j < args.single_sample:
                        single_res[j] = {"steps": steps, "q": q, "ssid": ssid, "text": text}

            if n_total % 1_000_000 == 0:
                print(f"  ...scanned {n_total:,} lines", flush=True)

    print(f"[pass done] {n_total:,} records", flush=True)

    # ---- leakage summaries ----
    def multiplicity(counts):
        total = len(counts)
        dup_keys = sum(1 for v in counts.values() if v > 1)
        recs_in_dupes = sum(v for v in counts.values() if v > 1)
        mx = max(counts.values()) if counts else 0
        return {"unique_keys": total, "dup_keys": dup_keys,
                "records_in_dup_groups": recs_in_dupes, "max_multiplicity": mx}

    ssid_summary = multiplicity(ssid_counts)
    q_summary = multiplicity(q_counts)

    # ---- tokenize sample ----
    os.environ.setdefault("HF_HOME", "/scratch/sghos104/.hf_cache")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    def tlen(text):
        return len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])

    # multi-step joined trajectories
    joined_lens, joined_lens_by_bucket = [], defaultdict(list)
    step_lens = []
    total_windows = 0
    punct_counter = Counter()      # what non-final steps end with
    n_nonfinal = 0
    newline_end = 0
    for rec in multi_res:
        steps = rec["steps"]
        text = "\n".join(steps)
        L = tlen(text)
        joined_lens.append(L)
        b = len(steps) if len(steps) <= 5 else 6
        joined_lens_by_bucket[b].append(L)
        total_windows += windows_for_len(L)
        for s in steps:
            step_lens.append(tlen(s))
        for s in steps[:-1]:   # non-final steps
            n_nonfinal += 1
            raw = s.rstrip()
            if s.endswith("\n"):
                newline_end += 1
            last = raw[-1] if raw else ""
            if last in PUNCT_END:
                punct_counter[last] += 1
            else:
                punct_counter["<other>"] += 1

    single_lens = [tlen(r["text"]) for r in single_res]

    # ---- text-dup leakage (sample-scoped) ----
    single_norm = set(norm_text(r["text"]) for r in single_res if r["text"])
    dup_hits = 0
    for rec in multi_res:
        for s in rec["steps"]:
            if norm_text(s) in single_norm:
                dup_hits += 1
                break

    # ---- assemble report ----
    punct_end_total = sum(punct_counter[c] for c in PUNCT_END)
    rep = {
        "n_total": n_total,
        "nstep_hist": dict(sorted(nstep_hist.items())),
        "multi_total_seen": multi_seen,
        "single_total_seen": single_seen,
        "sample": {"multi": len(multi_res), "single": len(single_res)},
        "schema": {
            "step_labels_value_dist": dict(steplabel_vals),
            "records_with_label_len_mismatch": n_bad_label_len,
            "records_empty_images": n_empty_images,
            "records_empty_answer": n_empty_answer,
            "records_with_empty_step": n_empty_step,
            "multi_records_with_stepmarker": n_multi_with_marker,
            "single_records_with_stepmarker": n_single_with_marker,
        },
        "leakage": {
            "source_sample_id": ssid_summary,
            "question": q_summary,
            "single_step_text_dup_in_multi_sample": {
                "multi_records_hit": dup_hits,
                "of_multi_sample": len(multi_res),
            },
        },
        "tokenization": {
            "joined_multi_len": dist_summary(joined_lens),
            "joined_multi_frac_over_512": round(
                sum(1 for x in joined_lens if x > 512) / max(1, len(joined_lens)), 4),
            "joined_len_by_bucket": {str(k): dist_summary(v) for k, v in sorted(joined_lens_by_bucket.items())},
            "per_step_len": dist_summary(step_lens),
            "single_step_len": dist_summary(single_lens),
            "single_frac_over_512": round(
                sum(1 for x in single_lens if x > 512) / max(1, len(single_lens)), 4),
            "est_total_windows_multi_sample": total_windows,
            "avg_windows_per_multi": round(total_windows / max(1, len(multi_res)), 3),
        },
        "punctuation": {
            "n_nonfinal_steps": n_nonfinal,
            "ends_in_punct": dict(punct_counter),
            "frac_nonfinal_ending_in_sentence_punct": round(punct_end_total / max(1, n_nonfinal), 4),
            "frac_nonfinal_ending_in_newline_char": round(newline_end / max(1, n_nonfinal), 4),
        },
    }

    with open(args.out_json, "w") as f:
        json.dump(rep, f, indent=2)
    # persist the sample for reuse (steps only, keeps it small-ish)
    with open(args.out_sample, "w") as f:
        for r in multi_res:
            f.write(json.dumps({"steps": r["steps"], "q": r["q"], "ssid": r["ssid"], "kind": "multi"}) + "\n")
        for r in single_res:
            f.write(json.dumps({"steps": r["steps"], "q": r["q"], "ssid": r["ssid"], "kind": "single"}) + "\n")

    print("\n" + "=" * 70)
    print(json.dumps(rep, indent=2))
    print("=" * 70)
    print(f"[written] {args.out_json}\n[written] {args.out_sample}")


if __name__ == "__main__":
    main()
