"""Build an MMK12-test MCQ eval parquet in the scienceqa_to_verl schema, for the
hard-benchmark cross-arm comparison (validation follow-up): does the match-trained
arm beat the answer-only ablation where accuracy actually demands multi-step visual
reasoning? (ScienceQA is near-saturated for both arms.)

MMK12 (FanqingM/MMK12, from MM-Eureka) test split: 2,000 K12 exam questions,
500 each math/physics/chemistry/biology, single image, choices embedded in the
question text as "A. ..." lines, `answer` = the letter. We extract the choice
texts, rebuild the prompt in our trained "Choices:\n- ..." format, and store the
gold CHOICE TEXT so the existing `answer_correct` harness grades both letter and
text answers. No GT rationales -> `solution` empty; score accuracy/format only.

Needs internet (run on a login/OOD shell, NOT under HF_HUB_OFFLINE).

Usage:
  python src/data_factory/mmk12_to_eval.py [--per-subject 256] \
      [--out data/mmk12_eval/mmk12_test1024.parquet]
"""

import argparse
import io
import os
import re

os.environ.setdefault("HF_HOME", "/scratch/sghos104/rlpt/data/hf_cache")

import pandas as pd

PROMPT_TEMPLATE = (
    "<image>\n"
    "Question: {question}\n"
    "Choices:\n{choices_block}\n\n"
    "Look at the image and reason step by step, then give the final answer as the "
    "exact text of the correct choice.\n"
    "Return exactly:\n"
    "<think> your step-by-step reasoning </think>\n"
    "<answer> the exact text of the correct choice </answer>"
)

# choice lines like "A. $$10.5$$" / "B.2.3." (label, dot, text)
_CHOICE_LINE = re.compile(r"^\s*([A-E])[.、]\s*(.+?)\s*$")


def split_question(text):
    """Return (stem, {letter: choice_text}) or (None, None) if unparseable."""
    stem_lines, choices = [], {}
    for line in text.splitlines():
        m = _CHOICE_LINE.match(line)
        if m:
            choices[m.group(1)] = m.group(2)
        elif not choices:
            stem_lines.append(line)
        else:
            # text after the choice block (rare) — treat as unparseable
            if line.strip():
                return None, None
    letters = sorted(choices)
    if len(letters) < 2 or letters != [chr(ord("A") + i) for i in range(len(letters))]:
        return None, None
    return "\n".join(stem_lines).strip(), choices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=256)
    ap.add_argument("--out", default="/scratch/sghos104/rlpt/data/mmk12_eval/mmk12_test1024.parquet")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("FanqingM/MMK12", split="test", streaming=True)

    rows, per_subject, skipped = [], {}, 0
    for ex in ds:
        subj = ex["subject"]
        if per_subject.get(subj, 0) >= args.per_subject:
            continue
        stem, choices = split_question(ex["question"])
        if stem is None or ex["answer"] not in choices:
            skipped += 1
            continue
        choice_texts = [choices[l] for l in sorted(choices)]
        img = ex["image"].convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        content = PROMPT_TEMPLATE.format(
            question=stem,
            choices_block="\n".join(f"- {c}" for c in choice_texts))
        rows.append({
            "data_source": "mmk12_composite",
            "prompt": [{"role": "user", "content": content}],
            "images": [{"bytes": buf.getvalue(), "path": None}],
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "answer": choices[ex["answer"]],
                    "solution": "",
                    "choices": choice_texts,
                },
            },
            "ability": "image_grounded_reasoning",
            "extra_info": {
                "question_id": ex["id"],
                "index": len(rows),
                "split": "test",
                "subject": subj,
                "topic": subj,
                "grade": "",
            },
        })
        per_subject[subj] = per_subject.get(subj, 0) + 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {len(rows)} rows -> {args.out}  per_subject={per_subject}  skipped={skipped}")


if __name__ == "__main__":
    main()
