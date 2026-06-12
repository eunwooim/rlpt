"""Tiny stdin/stdout SBERT embedding server.

Runs in its OWN fresh Python interpreter so SBERT loads with a pristine torch state.
verl's reward-actor process puts transformers into a meta/fast-init state that makes
in-process SBERT loading fail ("cannot copy out of meta tensor") even when the device/
env look clean — a subprocess sidesteps that entirely.

Protocol (one JSON object per line):
  in : ["clause one", "clause two", ...]      (a JSON list of strings)
  out: [[...384 floats...], ...]              (L2-normalized embeddings) or {"error": ...}
First line emitted on startup is "READY".
"""

import json
import os
import sys

os.environ.setdefault("HF_HOME", "/scratch/sghos104/rlpt/data/hf_cache")
os.environ.setdefault("NLTK_DATA", "/scratch/sghos104/rlpt/data/nltk_data")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-MiniLM-L6-v2"
    device = os.environ.get("RLPT_REWARD_DEVICE", "cpu")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            units = json.loads(line)
            if units:
                emb = model.encode(
                    units, normalize_embeddings=True, convert_to_numpy=True,
                    batch_size=64, show_progress_bar=False,
                )
                out = emb.tolist()
            else:
                out = []
            sys.stdout.write(json.dumps(out) + "\n")
        except Exception as e:  # noqa: BLE001
            sys.stdout.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
