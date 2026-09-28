#!/usr/bin/env python
"""Match-reward scoring server (NLI + optional chunker) for the six-arm campaign.

Runs in rlpt-train, owns its own GPU slice, and serves the verl reward worker
over a Unix domain socket with newline-delimited JSON. One request per
connection; requests are handled sequentially (single GPU consumer).

Why a server: models cannot load inside verl's reward worker (meta-tensor
state — same constraint that forced tools/sbert_embed_server.py).

Ops:
  {"op": "ping"}                                  -> {"ok": true, "id2label": ..., "chunker": bool}
  {"op": "segment", "text": s, "mode": m}         -> {"segments": [...]}
  {"op": "match", "response": s, "gold_steps": [..], "segmentation": "chunker"|"marker",
   "tau": t}                                      -> {"match": F, "n_segments": .., "n_matched": ..,
                                                      "precision": .., "recall": .., "n_gold": ..}
"""
import argparse
import json
import os
import socket
import sys
import threading
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHUNKER_DIR = "/scratch/sghos104/rlpt/chunker"
CHUNKER_RELEASE = os.path.join(CHUNKER_DIR, "release/chunker-deberta-v3-small-v2-mask")


class Handler:
    def __init__(self, with_chunker: bool):
        from nli_match import NLIScorer, match_fbeta
        from segmentation import split_marker, split_vprm

        self.match_fbeta = match_fbeta
        self.split_marker = split_marker
        self.split_vprm = split_vprm
        self.scorer = NLIScorer()
        print(f"[server] NLI ready: id2label={self.scorer.id2label}", flush=True)
        self.chunker = None
        self.chunker_thr = None
        if with_chunker:
            sys.path.insert(0, CHUNKER_DIR)
            import chunker as chk

            self.chunker_thr = chk.load(CHUNKER_RELEASE)
            self.chunker = chk
            print(f"[server] chunker ready: {CHUNKER_RELEASE} thr={self.chunker_thr}", flush=True)

    def segments(self, text: str, mode: str):
        text = (text or "").strip()
        if not text:
            return []
        if mode == "marker":
            return self.split_marker(text)
        if mode == "vprm":
            return self.split_vprm(text)
        if mode == "chunker":
            if self.chunker is None:
                raise RuntimeError("server started without --with_chunker")
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.chunker.chunk(text)["chunks"]
        raise ValueError(f"unknown segmentation mode {mode!r}")

    def _rv(self):
        """Lazy adapter: reward_v2 scoring on top of the server's already-loaded NLI model (+ MiniLM for dedupe)."""
        if getattr(self, "_rv_adapter", None) is None:
            import sys, numpy as np, torch
            sys.path.insert(0, "/scratch/sghos104/rlpt/reward_redesign")
            import reward_v2 as rv
            from transformers import AutoTokenizer
            from sentence_transformers import SentenceTransformer
            srv = self
            class Adapter:
                def __init__(self):
                    self.tok = AutoTokenizer.from_pretrained(rv.NLI_MODEL)
                    self.sbert = SentenceTransformer(rv.SBERT_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
                def nli_matrix(self, roll, gold):
                    m = np.asarray(srv.scorer.pair_scores(roll, gold), dtype=float)
                    if m.size:
                        m = np.clip(m, 0.0, 1.0)
                        over = [i for i, r in enumerate(roll) if len(self.tok.encode(r)) > rv.NLI_MAX_TOKENS]
                        if over:
                            m[over, :] = 0.0
                    return m
                def dedupe(self, segs):
                    return rv.Scorers.dedupe(self, segs)
            self._rv_adapter = Adapter(); self._rv_mod = rv
            print("[server] reward_v2 adapter ready", flush=True)
        return self._rv_adapter, self._rv_mod

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "reward_v2":
            adapter, rv = self._rv()
            b = rv.score_new(req["response"], req["gold_steps"], str(req.get("gold_answer", "")), adapter, req.get("finish_reason"),
                             mode=req.get("mode"), gate_mode=req.get("gate_mode"))
            d = rv.as_dict(b)
            # v4: answer_valid = parsable answer on the last line and not truncated/gated; acc = correct AND valid (same meaning in hard and soft mode)
            d["answer_valid"] = int((not b.gated) and b.extracted is not None and rv.answer_at_end(req["response"]))
            d["acc"] = int(b.answer) if d["answer_valid"] else 0
            return d
        if op == "ping":
            return {"ok": True, "id2label": self.scorer.id2label,
                    "chunker": self.chunker is not None, "chunker_thr": self.chunker_thr}
        if op == "segment":
            return {"segments": self.segments(req["text"], req["mode"])}
        if op == "match":
            segs = self.segments(req["response"], req["segmentation"])
            s_mat = self.scorer.pair_scores(segs, req["gold_steps"])
            out = self.match_fbeta(s_mat, float(req["tau"]))
            return out
        raise ValueError(f"unknown op {op!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--with_chunker", action="store_true")
    args = ap.parse_args()

    handler = Handler(with_chunker=args.with_chunker)

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    os.chmod(args.socket, 0o700)
    # verl's reward loop fires one connection per rollout concurrently (hundreds
    # per step). Accept them all in threads immediately; serialize GPU work with
    # a lock. A small backlog + sequential accept overflowed into EAGAIN.
    srv.listen(1024)
    gpu_lock = threading.Lock()
    print(f"[server] listening on {args.socket}", flush=True)

    def serve(conn):
        try:
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(1 << 20)
                if not chunk:
                    break
                buf += chunk
            if not buf.strip():
                return
            req = json.loads(buf.decode("utf-8"))
            try:
                with gpu_lock:
                    resp = handler.handle(req)
            except Exception as exc:  # report scoring errors to the client, keep serving
                traceback.print_exc()
                resp = {"error": f"{type(exc).__name__}: {exc}"}
            conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
        except Exception:
            traceback.print_exc()
        finally:
            conn.close()

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
