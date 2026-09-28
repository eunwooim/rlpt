"""Build equation-negation hard-negative pairs from the VisualPRM400K MATH subset.

For each math PRM example we reconstruct the (correct, +-labeled) step-by-step
solution text, then create a hard negative by flipping the FIRST numeric value on
the right-hand side of an equation (e.g. `y = 15` -> `y = 16`, `b = 0` -> `b = 1`),
mirroring the supervisor's `AB=1 -> AB=0` request.

Outputs pairs.jsonl with, per example:
  id, source, question, pos (orig solution), neg (one equation negated),
  edit (old->new), plus a `baseline` partner = an UNRELATED solution (shuffled),
  used as a random-similarity floor when scoring.
"""
from __future__ import annotations
import argparse, json, random, re, zipfile
from pathlib import Path

ZIP = "/scratch/sghos104/rlpt/data/visualprm400k/annotations.zip"
MATH_KEYS = ["geometry3k", "geoqa", "geo170k", "geos_en", "unigeo", "mavis",
             "geomverse", "clevr_math", "mathv360k", "super_clevr", "iconqa"]
EQ_NUM = re.compile(r"=\s*(-?\d+\.?\d*)")  # an '=' followed by a number (RHS value)


def math_prm_files(z: zipfile.ZipFile) -> list[str]:
    out = []
    for n in z.namelist():
        f = n.lower()
        if f.endswith("prm.jsonl") and any(k in f for k in MATH_KEYS):
            out.append(n)
    return sorted(out)


def extract_solution(conv: list[dict]) -> str:
    """Concatenate the reasoning step turns (human turns after the question)."""
    steps = []
    for t in conv:
        if t.get("from") not in ("human", "user"):
            continue
        v = t.get("value", "")
        if "### Question:" in v or "<image>" in v:
            continue  # the problem statement, not a reasoning step
        steps.append(v.strip())
    return "\n".join(s for s in steps if s)


def negate(text: str):
    """Flip the first RHS equation number. Returns (negated_text, 'old->new') or None."""
    m = EQ_NUM.search(text)
    if not m:
        return None
    num = m.group(1)
    try:
        if "." in num:
            old = float(num); new = old + 1.0
            new_s = f"{new:g}"
        else:
            old = int(num); new = old + 1 if old != -1 else 1  # avoid -1->0 ambiguity, still changes
            new_s = str(new)
    except ValueError:
        return None
    if new_s == num:
        return None
    s, e = m.span(1)
    return text[:s] + new_s + text[e:], f"{num}->{new_s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--cap", type=int, default=300000, help="max eligible to collect before sampling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/scratch/sghos104/rlpt/data/visualprm400k/pairs.jsonl")
    args = ap.parse_args()
    random.seed(args.seed)

    z = zipfile.ZipFile(ZIP)
    files = math_prm_files(z)
    print(f"math PRM files: {len(files)}")

    eligible = []  # (id, source, question, pos, neg, edit)
    for fn in files:
        with z.open(fn) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                conv = rec.get("conversations")
                if not conv:
                    continue
                sol = extract_solution(conv)
                if len(sol) < 15:
                    continue
                neg = negate(sol)
                if neg is None:
                    continue
                # question text (for reference)
                q = ""
                for t in conv:
                    if t.get("from") in ("human", "user") and "### Question:" in t.get("value", ""):
                        q = t["value"].split("Question:")[-1].split("### Solution")[0].strip()[:300]
                        break
                eligible.append((rec.get("id"), fn.split("/")[-1], q, sol, neg[0], neg[1]))
                if len(eligible) >= args.cap:
                    break
        if len(eligible) >= args.cap:
            break

    print(f"eligible (equation-bearing) collected: {len(eligible):,}")
    n = min(args.n, len(eligible))
    sample = random.sample(eligible, n)

    # build unrelated baseline partners via a derangement-ish shuffle
    idx = list(range(n))
    shuf = idx[:]
    random.shuffle(shuf)
    for i in range(n):  # avoid self-pairing
        if shuf[i] == i:
            shuf[i] = shuf[(i + 1) % n]

    out = Path(args.out)
    with out.open("w") as w:
        for i, (rid, src, q, pos, neg, edit) in enumerate(sample):
            baseline = sample[shuf[i]][3]  # an unrelated pos solution
            w.write(json.dumps({
                "idx": i, "id": rid, "source": src, "question": q,
                "pos": pos, "neg": neg, "edit": edit, "baseline": baseline,
            }) + "\n")
    print(f"wrote {n:,} pairs -> {out}")

    print("\n=== sample negations ===")
    for rid, src, q, pos, neg, edit in sample[:6]:
        print(f"\n[{src}] edit {edit}")
        print("  POS:", pos[:200].replace("\n", " / "))
        print("  NEG:", neg[:200].replace("\n", " / "))


if __name__ == "__main__":
    main()
