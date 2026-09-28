#!/usr/bin/env python3
"""Assemble probe.jsonl (populations: degenerate / clean / gold) from probe generations + gold steps."""
import json
REPO = "/scratch/sghos104/rlpt"
prompts = {r["qid"]: r for r in (json.loads(l) for l in open(f"{REPO}/grpo_arms/data/probe_prompts.jsonl"))}
def load(tag):
    rows = [json.loads(l) for l in open(f"{REPO}/grpo_arms/evals/vpb_gen_{tag}.jsonl")]
    key = next(k for k in ("output", "response", "generation", "text") if k in rows[0])
    return {r["qid"]: (r[key], r.get("finish_reason", "stop")) for r in rows}
gens = {"degenerate": load("probe_arm1"), "clean": load("probe_cs25")}
n = 0
with open(f"{REPO}/reward_redesign/probe.jsonl", "w") as f:
    for qid, p in prompts.items():
        base = {"id": qid, "gold_steps": p["gold_steps"], "gold_answer": p["answer"], "source": p["data_source"]}
        for pop, g in gens.items():
            if qid in g:
                f.write(json.dumps({**base, "population": pop, "rollout": g[qid][0], "finish_reason": g[qid][1]}, ensure_ascii=False) + "\n"); n += 1
        gold_text = "\n\n".join(p["gold_steps"]) + f"\n\nFinal answer: {p['answer']}"
        f.write(json.dumps({**base, "population": "gold", "rollout": gold_text, "finish_reason": "stop"}, ensure_ascii=False) + "\n"); n += 1
print(f"wrote {n} rows -> reward_redesign/probe.jsonl")
