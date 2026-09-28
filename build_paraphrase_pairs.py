"""Stage 1 of the paraphrase precision control: generate FAITHFUL paraphrases of the
correct VisualPRM solutions (pos), keeping every number / equation / final answer
identical and only rewording the prose.

This is the PRECISION mirror of the negation RECALL test. The negation result
(FPR 0.53->0.008) proved clause-split + min-pool catches a single flipped step, but it
is flattered: the corrupted copy is byte-identical except one clause, so 26/27 clauses
get a free equiv~1 and min-pool can never false-fire. Real model outputs are reworded-
but-correct. Here we generate exactly those: genuine rewordings of pos that are still
correct, so stage 2 can measure how often min-pool WRONGLY vetoes a correct solution.

Guardrail against confounds: a paraphraser can silently change a number, which would turn
a "paraphrase" into a real negation and inflate the false-positive rate. So we extract the
multiset of numeric tokens from pos and from the paraphrase; if they differ the paraphrase
is marked unfaithful and EXCLUDED from the eval set (we report how many were kept).

Output: data/visualprm400k/paraphrase_pairs.jsonl  {idx, pos, paraphrase, faithful}
Resume-safe: skips idxs already written.
"""
from __future__ import annotations
import argparse, json, os, re
from collections import Counter
from pathlib import Path

PROMPT = (
    "Reword the following step-by-step solution in different words. "
    "Strict rules:\n"
    "1. Keep EVERY number, variable, equation, and the final answer EXACTLY the same.\n"
    "2. Only change the wording/phrasing of the explanatory prose.\n"
    "3. Keep the same step-by-step structure (one step per line).\n"
    "4. Do NOT add, remove, merge, or reorder steps. Do NOT solve it differently.\n"
    "Output ONLY the reworded solution, nothing else.\n\n"
    "Solution:\n{sol}"
)

NUM_RE = re.compile(r"-?\d+\.?\d*")


def numbers(text):
    return Counter(NUM_RE.findall(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/scratch/sghos104/rlpt/data/visualprm400k/pairs.jsonl")
    ap.add_argument("--out", default="/scratch/sghos104/rlpt/data/visualprm400k/paraphrase_pairs.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    rows = [json.loads(l) for l in open(args.pairs)][: args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        for l in out.open():
            done.add(json.loads(l)["idx"])
    todo = [r for r in rows if r["idx"] not in done]
    print(f"loaded {len(rows):,} rows, {len(done):,} already done, {len(todo):,} to generate", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              trust_remote_code=True, limit_mm_per_prompt={"image": 0, "video": 0},
              gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=args.temperature, top_p=0.9, max_tokens=1024)

    convos = [[{"role": "user", "content": PROMPT.format(sol=r["pos"])}] for r in todo]
    print(f"generating {len(convos):,} paraphrases ...", flush=True)
    outs = llm.chat(convos, sp)

    kept = faithful = 0
    with out.open("a") as w:
        for r, o in zip(todo, outs):
            para = o.outputs[0].text.strip()
            is_faithful = numbers(r["pos"]) == numbers(para) and len(para) > 0
            w.write(json.dumps({"idx": r["idx"], "pos": r["pos"],
                                "paraphrase": para, "faithful": is_faithful}) + "\n")
            kept += 1
            faithful += int(is_faithful)
    print(f"wrote {kept:,} paraphrases, {faithful:,} faithful "
          f"({faithful/max(kept,1):.1%}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
