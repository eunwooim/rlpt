#!/usr/bin/env python
"""Final chunker demo on HELD-OUT (test) trajectories, guaranteeing that at
least one space-free CJK text is included -- the case the Gate-4 candidate-rule
fix was made for, and the one the token-F1 eval cannot cover (test has only 1
CJK multi-step trajectory).

For each demo trajectory: gold vs predicted boundaries, per-segment token
lengths, and any segment under min_tokens (should be none when the hard-min DP
path is feasible).
"""
import argparse, json
import chunker as ck
import chunker_common as cc


def whitespace_frac(t):
    return round(sum(c.isspace() for c in t) / max(1, len(t)), 4)


def show(r, min_tokens, max_tokens):
    steps = r["steps"]
    text = cc.join_steps(steps)
    gold, _ = cc.gold_boundary_char_positions(steps)
    out = ck.chunk(text, min_tokens=min_tokens, max_tokens=max_tokens)
    seg = out["segment_token_lengths"]
    short = [s for s in seg if s < min_tokens]
    print("=" * 78)
    print(f"lang={r['lang']}  kind={r['kind']}  n_steps={len(steps)}  "
          f"chars={len(text)}  whitespace_frac={whitespace_frac(text)}")
    print(f"  gold boundaries (char)  = {gold}")
    print(f"  predicted char cuts     = {out['split_char_offsets']}  (#chunks={len(out['chunks'])})")
    print(f"  segment token lengths   = {seg}")
    print(f"  segments < {min_tokens} tokens      = {len(short)} {short}")
    for g in gold:
        cuts = out["split_char_offsets"]
        near = min(cuts, key=lambda c: abs(c - g)) if cuts else None
        d = (near - (g + 1)) if near is not None else None
        print(f"    gold@{g:<6d} nearest_pred@{near}  delta={d}   ...{text[max(0, g - 24):g + 1]!r}")
    if out["chunks"]:
        print(f"  chunk[0]: {out['chunks'][0][:100]!r}")
        print(f"  chunk[-1]: {out['chunks'][-1][:100]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=ck.DEFAULT_MODEL_DIR)
    ap.add_argument("--traj", default="/scratch/sghos104/rlpt/chunker/data/test_traj.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--min_tokens", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=220)
    args = ap.parse_args()

    ck.load(args.model_dir)
    print(f"[chunker] model={args.model_dir} threshold={ck._thr}")
    rows = [json.loads(l) for l in open(args.traj)]

    cjk = [r for r in rows if r["lang"] == "cjk"]
    # prefer a multi-step CJK trajectory; fall back to the most space-free single-step one
    cjk_multi = [r for r in cjk if r["kind"] == "multi"]
    picks = []
    if cjk_multi:
        picks.append(cjk_multi[0])
    if cjk:
        # the least-whitespace CJK text = the real "no spaces after 。" case
        dense = sorted(cjk, key=lambda r: whitespace_frac(cc.join_steps(r["steps"])))[0]
        if dense not in picks:
            picks.append(dense)
    # Spread English picks across DISTINCT question groups. test_traj.jsonl is
    # ordered by question, so taking the first n gives near-duplicate rollouts of
    # one question -- which made an earlier demo look like a systematic failure
    # when it was really one correlated cluster.
    en_multi = [r for r in rows if r["kind"] == "multi" and r["lang"] == "en"]
    seen_q, spread = set(), []
    stride = max(1, len(en_multi) // max(1, args.n * 8))
    for r in en_multi[::stride]:
        if r["qh"] in seen_q:
            continue
        seen_q.add(r["qh"]); spread.append(r)
    for r in spread:
        if len(picks) >= args.n:
            break
        picks.append(r)

    print(f"[demo] {len(picks)} trajectories: langs={[p['lang'] for p in picks]} "
          f"(cjk available in test: {len(cjk)} total, {len(cjk_multi)} multi-step)")
    for r in picks[:args.n]:
        show(r, args.min_tokens, args.max_tokens)


if __name__ == "__main__":
    main()
