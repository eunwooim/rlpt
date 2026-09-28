import json, sys, statistics as st
path = sys.argv[1] if len(sys.argv) > 1 else "probe_scored.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
print(f"{len(rows)} scored rows in {path}")
for key in ("old", "new"):
    by = {}
    for r in rows: by.setdefault(r["population"], []).append(r[key])
    print(f"\n=== {key.upper()} ===")
    print(f"{'population':16s} {'n':>5s} {'R_med':>7s} {'R_mean':>7s} {'match':>6s} {'pun':>6s} {'inv':>5s} {'gated':>6s} {'ans%':>6s} {'segs':>5s}")
    for pop, xs in sorted(by.items()):
        print(f"{pop:16s} {len(xs):>5d} {st.median(x['R'] for x in xs):>7.2f} {st.mean(x['R'] for x in xs):>7.2f} "
              f"{st.median(x['match'] for x in xs):>6.2f} {st.median(x['pun'] for x in xs):>6.2f} {st.median(x['inv_frac'] for x in xs):>5.2f} "
              f"{sum(x['gated'] for x in xs):>6d} {st.mean(x['answer'] for x in xs):>6.1%} {st.median(x['n_segs'] for x in xs):>5.0f}")
