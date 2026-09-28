#!/usr/bin/env python
"""Gate-3 label-construction unit test.

Selects 5 multi-step trajectories exercising the different boundary types
(ASCII '.', LaTeX '\\]', CJK '。', list-item, bare digit), builds token labels via
chunker_common.build_token_labels, then for each:
  * asserts #label-1 tokens == #non-final steps,
  * asserts each label-1 token's char span ends exactly at a gold boundary char,
  * prints the decoded gold token and its surrounding context, side by side with
    the true step end, so a human can eyeball the alignment.
Also previews the 1-vs-0 class balance over candidate tokens on a small sample.
"""
import os, json, re
os.environ.setdefault("HF_HOME", "/scratch/sghos104/.hf_cache")
from transformers import AutoTokenizer
import chunker_common as cc

SAMPLE = "/scratch/sghos104/rlpt/chunker/profile_sample.jsonl"


def last_nonspace(s):
    st = s.rstrip()
    return st[-1] if st else ""


def pick_samples(recs):
    """Choose one short-ish multi-step rec for each boundary flavor."""
    want = {
        "ascii .": lambda steps: last_nonspace(steps[0]) == ".",
        "latex \\]": lambda steps: last_nonspace(steps[0]) == "]",
        "cjk 。": lambda steps: any(last_nonspace(s) == "。" for s in steps[:-1]),
        "list-item": lambda steps: any(cc.LIST_RX.match(steps[i + 1]) for i in range(len(steps) - 1)),
        "bare digit": lambda steps: any(last_nonspace(s).isdigit() for s in steps[:-1]),
    }
    chosen = {}
    for r in recs:
        if r["kind"] != "multi":
            continue
        steps = r["steps"]
        if not (2 <= len(steps) <= 5):   # keep output readable
            continue
        for name, pred in want.items():
            if name not in chosen and pred(steps):
                chosen[name] = r
        if len(chosen) == len(want):
            break
    return chosen


def main():
    tok = AutoTokenizer.from_pretrained(cc.MODEL)
    recs = [json.loads(l) for l in open(SAMPLE)]
    chosen = pick_samples(recs)

    all_ok = True
    for flavor, r in chosen.items():
        steps = r["steps"]
        ex = cc.build_token_labels(steps, tok)
        text, offs, labs = ex["text"], ex["offsets"], ex["labels"]
        golds = ex["gold_char_positions"]
        one_idx = [t for t, l in enumerate(labs) if l == cc.LABEL_SPLIT]

        print("=" * 78)
        print(f"FLAVOR: {flavor}   (n_steps={len(steps)}, n_tokens={len(labs)}, "
              f"non-final steps={len(steps)-1})")
        # per-step true ends
        print("  gold char positions:", golds)
        print(f"  #label-1 tokens = {len(one_idx)}  |  #non-final steps = {len(steps)-1}")

        ok = len(one_idx) == len(steps) - 1
        for t in one_idx:
            s, e = offs[t]
            tokstr = tok.decode([ex['input_ids'][t]])
            covers_gold = any(s <= g < e for g in golds)
            ok = ok and covers_gold
            ctx_l = text[max(0, s - 20):s]
            ctx_tok = text[s:e]
            ctx_r = text[e:e + 12]
            print(f"    tok#{t:<4d} span=({s},{e}) decode={tokstr!r:>12}  "
                  f"covers_gold={covers_gold}")
            print(f"        …{ctx_l!r} [[{ctx_tok!r}]] {ctx_r!r}…")
        # show each true step end for cross-check
        for i, g in enumerate(golds):
            print(f"    step{i} ends at char {g}: {text[max(0,g-24):g+1]!r}")
        print(f"  ALIGNMENT OK: {ok}")
        all_ok = all_ok and ok

    # ---- class-balance preview (candidate tokens only) ----
    print("\n" + "#" * 78)
    print("CLASS-BALANCE PREVIEW under expanded rule (small sample, not the final number)")
    n1 = n0 = 0
    n1_m = n0_m = n1_s = n0_s = 0
    for r in recs[:8000]:
        ex = cc.build_token_labels(r["steps"], tok)
        c1 = sum(1 for l in ex["labels"] if l == 1)
        c0 = sum(1 for l in ex["labels"] if l == 0)
        n1 += c1; n0 += c0
        if r["kind"] == "multi":
            n1_m += c1; n0_m += c0
        else:
            n1_s += c1; n0_s += c0
    def ratio(a, b): return f"1:{b/max(1,a):.1f}"
    print(f"  overall   label1={n1:,}  label0={n0:,}   ratio {ratio(n1,n0)}")
    print(f"  multi     label1={n1_m:,}  label0={n0_m:,}   ratio {ratio(n1_m,n0_m)}")
    print(f"  1-step    label1={n1_s:,}  label0={n0_s:,}   ratio {ratio(n1_s,n0_s)}  (negatives-only)")

    print("\n" + "=" * 78)
    print("OVERALL ALIGNMENT TEST:", "PASS" if all_ok else "FAIL")
    assert all_ok, "label-1 tokens did not align with gold step ends"


if __name__ == "__main__":
    main()
