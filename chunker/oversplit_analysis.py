#!/usr/bin/env python
"""Over-splitting analysis on VALIDATION at the SHIPPED threshold (0.25).

Over-splitting (spurious boundaries) is the chunker's main known weakness. This
quantifies it precisely and shows *where* it fires, reusing ONE forward pass per
trajectory (probs computed once, then the cheap DP + all metrics run off them --
no retraining, no re-inference).

Outputs (into --out_dir):
  oversplit_analysis.json   aggregate metrics, by language bucket
  oversplit_examples.md     30 worst 1-step + 30 worst multi-step offenders,
                            predicted cuts inline as  ...text⟦0.87⟧...  and gold
                            boundaries as  ...★, with lang / question-hash / tokens
  (also copied into --release_reports, and README limitations patched if --readme)

Metrics @ threshold, by language bucket (en / cjk / other) and ALL:
  1-step trajectories (gold = 0 boundaries, so every predicted cut is a FALSE split):
      false splits / trajectory   mean, histogram, % with >=1
      false splits / 1000 tokens
  multi-step trajectories:
      unmatched predicted cuts / trajectory (greedy match to gold, TOL=2)
      fragmentation ratio = predicted_segments / gold_segments
                            mean, % >1.5 (over-fragmented), % <0.7 (under)
"""
import argparse, json, os, shutil
from collections import Counter, defaultdict
import numpy as np
import chunker as ck
import chunker_common as cc

REPO = "/scratch/sghos104/rlpt/chunker"
TOL = 2
LANGS = ("en", "cjk", "other")
CONNECTIVES = {"so", "thus", "therefore", "hence", "then", "finally", "now",
               "next", "first", "second", "third"}


def decode(prob, cand, thr, min_t, max_t):
    """chunker.chunk()'s hard-then-soft min policy, reusing precomputed prob."""
    bnds, _, feasible = ck._dp_boundaries(prob, cand, min_t, max_t, thr, hard_min=True)
    if not feasible:
        bnds, _, _ = ck._dp_boundaries(prob, cand, min_t, max_t, thr, hard_min=False)
    return bnds


def match_gold(cuts, gold):
    """Greedy TOL-match of predicted char cuts to gold char positions.
    Returns (n_matched, matched_cut_indices set)."""
    used = set()
    matched = 0
    for g in gold:
        for ci, c in enumerate(cuts):
            if ci in used:
                continue
            if abs(c - (g + 1)) <= TOL:
                matched += 1
                used.add(ci)
                break
    return matched, used


def classify_cut(text, c):
    """Coarse context label for a false split whose chunk ends at char offset c
    (exclusive). Built from the closing char and the following word so the
    over-split can be attributed (list marker / connective / LaTeX closer / ...)."""
    last = text[c - 1] if c >= 1 else ""
    pre = text[max(0, c - 40):c]
    nxt = text[c:c + 48]
    pre_words = pre.split()
    prev_word = pre_words[-1].strip("".join(cc.CLOSER_CHARS)) if pre_words else ""
    nxt_stripped = nxt.lstrip()
    nxt_first = nxt_stripped.split()[0].lower().strip("".join(cc.CLOSER_CHARS)) if nxt_stripped else ""
    # does a list item begin right after?
    line_after = nxt if nxt.startswith("\n") else ("\n" + nxt)
    is_list_next = bool(cc.LIST_RX.match(line_after.lstrip("\n")[:48]) or cc.LIST_RX.match(nxt_stripped[:48]))

    if last in cc.CJK_PUNCT:
        return "CJK punctuation"
    if nxt_first in CONNECTIVES:
        return f"before connective '{nxt_first}'"
    if is_list_next:
        return "before list item (N. / -)"
    if last in cc.MATH_CLOSE:
        return f"LaTeX/math closer '{last}'"
    if last.isdigit():
        return "bare digit"
    if last == ":":
        return "colon ':'"
    if last in ";":
        return "semicolon ';'"
    if last in "!?":
        return f"'{last}'"
    if last == ".":
        # CLEVR-template sentences are short and end on these nouns/numbers
        clevr = {"objects", "spheres", "balls", "cubes", "cylinders", "things",
                 "sphere", "ball", "cube", "cylinder", "object"}
        if prev_word.lower() in clevr:
            return f"period after CLEVR noun ('{prev_word.lower()}')"
        return "sentence period '.'"
    return f"other (ends '{last}')"


def annotate(text, cuts, cut_probs, gold, cap=2000):
    """Insert  ⟦prob⟧  at each predicted cut char offset and  ★  after each gold
    boundary char, right-to-left so offsets stay valid."""
    marks = []
    for c, p in zip(cuts, cut_probs):
        marks.append((c, f"⟦{p:.2f}⟧"))
    for g in gold:
        marks.append((g + 1, "★"))
    # right-to-left; at equal positions place ★ before the cut marker
    marks.sort(key=lambda m: (m[0], 0 if m[1] == "★" else 1), reverse=True)
    s = text
    for pos, mk in marks:
        pos = max(0, min(pos, len(s)))
        s = s[:pos] + mk + s[pos:]
    s = s.replace("\n", "\\n ")
    if len(s) > cap:
        s = s[:cap] + f"  …[+{len(s) - cap} chars]"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",
                    default=os.path.join(REPO, "release/chunker-deberta-v3-small-v1/model"),
                    help="release model dir -> picks up the SHIPPED threshold (0.25)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override; default = model's chunker_meta.json best_thr")
    ap.add_argument("--val_traj", default=os.path.join(REPO, "data/val_traj.jsonl"))
    ap.add_argument("--out_dir", default=os.path.join(REPO, "runs/full_plain"))
    ap.add_argument("--release_reports",
                    default=os.path.join(REPO, "release/chunker-deberta-v3-small-v1/reports"))
    ap.add_argument("--readme",
                    default=os.path.join(REPO, "release/chunker-deberta-v3-small-v1/README.md"))
    ap.add_argument("--min_tokens", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=220)
    ap.add_argument("--n_examples", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap trajectories per kind")
    args = ap.parse_args()

    thr = ck.load(args.model_dir, threshold=args.threshold)
    print(f"[oversplit] shipped threshold = {thr}  model = {args.model_dir}", flush=True)

    rows = [json.loads(l) for l in open(args.val_traj)]
    single = [r for r in rows if r["kind"] == "single"]
    multi = [r for r in rows if r["kind"] == "multi"]
    if args.limit:
        single, multi = single[:args.limit], multi[:args.limit]
    print(f"[oversplit] {len(single):,} single / {len(multi):,} multi", flush=True)

    # ---- pass 1: 1-step (every predicted cut is a false split) ----
    single_recs = []
    ctx_counter = Counter()          # coarse context of every false split (both kinds)
    prevword_counter = Counter()     # exact preceding word, secondary view
    for i, r in enumerate(single):
        text = r["text"]
        ids, offs, prob = ck._token_probs(text)
        cand = np.array(cc.candidate_token_mask(text, offs), dtype=bool)
        bnds = decode(prob, cand, thr, args.min_tokens, args.max_tokens)
        cuts = [offs[b][1] for b in bnds]
        cprob = [float(prob[b]) for b in bnds]
        ntok = len(prob)
        for c in cuts:
            ctx_counter[classify_cut(text, c)] += 1
            pw = text[max(0, c - 40):c].split()
            if pw:
                prevword_counter[pw[-1].strip("".join(cc.CLOSER_CHARS)).lower()[:24]] += 1
        single_recs.append({"lang": r["lang"], "qh": r.get("qh"), "tokens": ntok,
                            "false_splits": len(cuts), "cuts": cuts, "cprob": cprob,
                            "text": text})
        if (i + 1) % 1000 == 0:
            print(f"  ...single {i+1:,}/{len(single):,}", flush=True)

    # ---- pass 2: multi-step (unmatched cuts = over-splits; fragmentation ratio) ----
    multi_recs = []
    for i, r in enumerate(multi):
        text = r["text"]
        gold, _ = cc.gold_boundary_char_positions(r["steps"])
        ids, offs, prob = ck._token_probs(text)
        cand = np.array(cc.candidate_token_mask(text, offs), dtype=bool)
        bnds = decode(prob, cand, thr, args.min_tokens, args.max_tokens)
        cuts = [offs[b][1] for b in bnds]
        cprob = [float(prob[b]) for b in bnds]
        ntok = len(prob)
        matched, used = match_gold(cuts, gold)
        unmatched_idx = [ci for ci in range(len(cuts)) if ci not in used]
        for ci in unmatched_idx:
            c = cuts[ci]
            ctx_counter[classify_cut(text, c)] += 1
            pw = text[max(0, c - 40):c].split()
            if pw:
                prevword_counter[pw[-1].strip("".join(cc.CLOSER_CHARS)).lower()[:24]] += 1
        gold_seg = len(gold) + 1
        pred_seg = len(cuts) + 1
        frag = pred_seg / gold_seg
        multi_recs.append({"lang": r["lang"], "qh": r.get("qh"), "tokens": ntok,
                           "n_gold": len(gold), "n_cuts": len(cuts),
                           "unmatched": len(unmatched_idx), "frag": frag,
                           "cuts": cuts, "cprob": cprob, "gold": gold,
                           "unmatched_idx": unmatched_idx, "text": text})
        if (i + 1) % 2000 == 0:
            print(f"  ...multi {i+1:,}/{len(multi):,}", flush=True)

    # ---- aggregate by language bucket ----
    def agg_single(recs):
        n = len(recs)
        if n == 0:
            return {"n": 0}
        fs = np.array([x["false_splits"] for x in recs])
        tk = np.array([x["tokens"] for x in recs])
        hist = Counter(int(v) for v in fs)
        hist_out = {str(k): hist.get(k, 0) for k in range(0, 6)}
        hist_out["6+"] = sum(v for k, v in hist.items() if k >= 6)
        return {
            "n": n,
            "false_splits_per_traj_mean": round(float(fs.mean()), 4),
            "false_splits_per_traj_p50": int(np.percentile(fs, 50)),
            "false_splits_per_traj_p90": int(np.percentile(fs, 90)),
            "false_splits_per_traj_max": int(fs.max()),
            "pct_traj_with_ge1_false_split": round(100 * float((fs >= 1).mean()), 2),
            "false_splits_per_1000_tokens": round(1000 * float(fs.sum()) / max(1, int(tk.sum())), 4),
            "histogram_false_splits": hist_out,
        }

    def agg_multi(recs):
        n = len(recs)
        if n == 0:
            return {"n": 0}
        um = np.array([x["unmatched"] for x in recs])
        fr = np.array([x["frag"] for x in recs])
        tk = np.array([x["tokens"] for x in recs])
        return {
            "n": n,
            "unmatched_cuts_per_traj_mean": round(float(um.mean()), 4),
            "unmatched_cuts_per_traj_p50": int(np.percentile(um, 50)),
            "unmatched_cuts_per_traj_p90": int(np.percentile(um, 90)),
            "unmatched_cuts_per_traj_max": int(um.max()),
            "pct_traj_with_ge1_unmatched": round(100 * float((um >= 1).mean()), 2),
            "unmatched_per_1000_tokens": round(1000 * float(um.sum()) / max(1, int(tk.sum())), 4),
            "frag_ratio_mean": round(float(fr.mean()), 4),
            "pct_frag_gt_1.5": round(100 * float((fr > 1.5).mean()), 2),
            "pct_frag_lt_0.7": round(100 * float((fr < 0.7).mean()), 2),
        }

    single_by = {lg: agg_single([x for x in single_recs if x["lang"] == lg]) for lg in LANGS}
    single_by["ALL"] = agg_single(single_recs)
    multi_by = {lg: agg_multi([x for x in multi_recs if x["lang"] == lg]) for lg in LANGS}
    multi_by["ALL"] = agg_multi(multi_recs)

    total_false = sum(x["false_splits"] for x in single_recs) + sum(x["unmatched"] for x in multi_recs)
    patterns = {
        "total_false_splits_clustered": total_false,
        "top_contexts": [{"context": k, "count": v,
                          "pct": round(100 * v / max(1, total_false), 2)}
                         for k, v in ctx_counter.most_common(10)],
        "top_preceding_words": [{"word": k, "count": v} for k, v in prevword_counter.most_common(15)],
    }

    report = {
        "threshold": thr, "tol": TOL,
        "min_tokens": args.min_tokens, "max_tokens": args.max_tokens,
        "counts": {"single": len(single_recs), "multi": len(multi_recs)},
        "single_step_false_splits": single_by,
        "multi_step_oversplit": multi_by,
        "false_split_patterns": patterns,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "oversplit_analysis.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[written] {out_json}")

    # ---- worst-offender inspection file ----
    worst_single = sorted(single_recs, key=lambda x: (x["false_splits"], -x["tokens"]),
                          reverse=True)[:args.n_examples]
    worst_multi = sorted(multi_recs, key=lambda x: (x["unmatched"], x["frag"]),
                         reverse=True)[:args.n_examples]

    L = ["# Over-splitting — worst offenders (validation @ threshold %s)\n" % thr,
         "Predicted cuts marked inline as `⟦prob⟧` at the split point; gold "
         "boundaries as `★`. Newlines shown as `\\n`. `qh` = question hash "
         "(the raw question is not stored in the trajectory files).\n",
         "For 1-step trajectories every `⟦…⟧` is a FALSE split (gold has no "
         "internal boundary). For multi-step, a `⟦…⟧` NOT adjacent to a `★` is an "
         "over-split.\n"]

    L.append("\n## 30 worst 1-step offenders (most false splits)\n")
    for k, x in enumerate(worst_single, 1):
        L.append(f"**{k}. {x['false_splits']} false splits** — lang=`{x['lang']}` "
                 f"qh=`{x['qh']}` tokens={x['tokens']}\n")
        L.append("> " + annotate(x["text"], x["cuts"], x["cprob"], []) + "\n")

    L.append("\n## 30 worst multi-step offenders (most unmatched predicted cuts)\n")
    for k, x in enumerate(worst_multi, 1):
        L.append(f"**{k}. {x['unmatched']} unmatched cuts** "
                 f"({x['n_cuts']} pred / {x['n_gold']} gold, frag={x['frag']:.2f}) — "
                 f"lang=`{x['lang']}` qh=`{x['qh']}` tokens={x['tokens']}\n")
        L.append("> " + annotate(x["text"], x["cuts"], x["cprob"], x["gold"]) + "\n")

    out_md = os.path.join(args.out_dir, "oversplit_examples.md")
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[written] {out_md}")

    # ---- copy artifacts into the release reports dir ----
    if args.release_reports and os.path.isdir(args.release_reports):
        for p in (out_json, out_md):
            shutil.copy2(p, os.path.join(args.release_reports, os.path.basename(p)))
        print(f"[copied] -> {args.release_reports}")

    # ---- patch README limitations (idempotent between sentinels) ----
    if args.readme and os.path.exists(args.readme):
        a1 = single_by["ALL"]
        am = multi_by["ALL"]
        top3 = ", ".join(f"{c['context']} ({c['pct']}%)" for c in patterns["top_contexts"][:3])
        block = (
            "<!-- oversplit-analysis:start -->\n"
            f"- **Over-splitting, measured on validation @ threshold {thr}** "
            f"(`reports/oversplit_analysis.json`, examples in `reports/oversplit_examples.md`):\n"
            f"  - 1-step trajectories: **{a1['false_splits_per_traj_mean']}** false splits/traj "
            f"({a1['pct_traj_with_ge1_false_split']}% get >=1; "
            f"{a1['false_splits_per_1000_tokens']}/1000 tok). "
            f"By bucket en {single_by['en'].get('false_splits_per_traj_mean')} / "
            f"cjk {single_by['cjk'].get('false_splits_per_traj_mean')}.\n"
            f"  - multi-step: **{am['unmatched_cuts_per_traj_mean']}** unmatched cuts/traj; "
            f"fragmentation ratio mean **{am['frag_ratio_mean']}** "
            f"({am['pct_frag_gt_1.5']}% over-fragmented >1.5, {am['pct_frag_lt_0.7']}% under <0.7).\n"
            f"  - top false-split contexts: {top3}.\n"
            "<!-- oversplit-analysis:end -->"
        )
        txt = open(args.readme).read()
        s, e = "<!-- oversplit-analysis:start -->", "<!-- oversplit-analysis:end -->"
        if s in txt and e in txt:
            pre = txt[:txt.index(s)]
            post = txt[txt.index(e) + len(e):]
            txt = pre + block + post
        else:
            anchor = "## Provenance"
            if anchor in txt:
                txt = txt.replace(anchor, block + "\n\n" + anchor, 1)
            else:
                txt = txt.rstrip() + "\n\n" + block + "\n"
        with open(args.readme, "w") as f:
            f.write(txt)
        print(f"[patched] {args.readme}")

    # ---- console summary ----
    print("\n=== 1-STEP false splits (mean/traj | %>=1 | /1000tok) ===")
    for lg in ("ALL",) + LANGS:
        d = single_by[lg]
        if d.get("n"):
            print(f"  {lg:<6} n={d['n']:<6} {d['false_splits_per_traj_mean']:<7} "
                  f"{d['pct_traj_with_ge1_false_split']:<6}% {d['false_splits_per_1000_tokens']}")
    print("=== MULTI-STEP over-split (unmatched/traj | fragMean | %>1.5 | %<0.7) ===")
    for lg in ("ALL",) + LANGS:
        d = multi_by[lg]
        if d.get("n"):
            print(f"  {lg:<6} n={d['n']:<6} {d['unmatched_cuts_per_traj_mean']:<7} "
                  f"{d['frag_ratio_mean']:<7} {d['pct_frag_gt_1.5']:<6}% {d['pct_frag_lt_0.7']}%")
    print("=== TOP FALSE-SPLIT CONTEXTS ===")
    for c in patterns["top_contexts"]:
        print(f"  {c['count']:>7,}  {c['pct']:>5}%  {c['context']}")


if __name__ == "__main__":
    main()
