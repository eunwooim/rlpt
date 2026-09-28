#!/usr/bin/env python
"""Preflight for the six-arm campaign (everything except the verl trivial step,
which runs as its own 2-GPU job through train_arm.py).

Checks, each with an explicit PASS/FAIL line; any FAIL -> rc 2:
  1. frozen subset sha256 matches + every image path exists
  2. chunker release loads and segments a sample gold response  [scorer env]
  3. deberta-xlarge-mnli loads; id2label printed; probe pair sanity  [scorer env]
  4. score_server cross-process: ping + full match op (identity high, shuffled-gold low)
  5. vprm_server cross-process (vprm-judge env): ping + step scores on a sample record
  6. tau cross-check: rescore 200 tau-sweep pairs in the scorer env, compare

The scorer env is chunker/env (transformers 5.14) — the release chunker
tokenizer cannot load under rlpt-train's transformers 4.56 (documented trap),
so ALL match-reward scoring (NLI + chunker) runs there, for every match arm.
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

REPO = "/scratch/sghos104/rlpt"
sys.path.insert(0, os.path.join(REPO, "grpo_arms"))

VPRM_PY = "/scratch/sghos104/rlpt/envs/vprm-judge/bin/python"
TRAIN_PY = "/scratch/sghos104/envs/rlpt-train/bin/python"
SCORER_PY = "/scratch/sghos104/rlpt/chunker/env/bin/python"

FAILURES = []


def report(name, ok, detail=""):
    print(f"[preflight] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def sock_call(path, payload, timeout=900):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
        c.settimeout(timeout)
        c.connect(path)
        c.sendall((json.dumps(payload) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            ch = c.recv(1 << 20)
            if not ch:
                break
            buf += ch
    return json.loads(buf.decode())


def wait_for_socket(path, proc, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        if os.path.exists(path):
            try:
                return sock_call(path, {"op": "ping"}, timeout=60)
            except (ConnectionRefusedError, socket.timeout, OSError):
                pass
        time.sleep(5)
    return False


def main():
    # --- 1. frozen subset ---
    sub = os.path.join(REPO, "grpo_arms/data/train_subset.jsonl")
    sha = hashlib.sha256(open(sub, "rb").read()).hexdigest()
    expected = open(os.path.join(REPO, "grpo_arms/data/train_subset.sha256")).read().strip()
    rows = [json.loads(l) for l in open(sub)]
    missing = [r["image"] for r in rows if not os.path.exists(os.path.join(REPO, r["image"]))]
    report("subset_sha256", sha == expected, f"{sha[:16]}.. n={len(rows)}")
    report("subset_images_exist", not missing, f"missing={len(missing)}")

    sample = rows[0]
    sample_rollout = sample["response"]

    # --- 2 + 3. chunker + NLI, in the scorer env (chunker/env) ---
    probe = subprocess.run(
        [SCORER_PY, "-"], input=f"""
import json, sys
sys.path.insert(0, {json.dumps(os.path.join(REPO, 'grpo_arms'))})
sys.path.insert(0, {json.dumps(os.path.join(REPO, 'chunker'))})
import chunker as chk
thr = chk.load({json.dumps(os.path.join(REPO, 'chunker/release/chunker-deberta-v3-small-v2-mask'))})
segs = chk.chunk({json.dumps(sample_rollout)})["chunks"]
print("CHUNKER_OK", thr, len(segs))
for s in segs[:3]:
    print("   [chunker seg]", s[:100].replace(chr(10), " "))
from nli_match import NLIScorer
nli = NLIScorer()
print("NLI_ID2LABEL", json.dumps(nli.id2label))
e, c = nli.ec_probs(
    ["The triangle has three sides.", "The answer is 7."],
    ["A triangle is three-sided.", "The answer is 12."])
print("NLI_PROBE", round(e[0], 4), round(c[1], 4))
print("PROBE_OK" if (e[0] > 0.5 and c[1] > 0.5) else "PROBE_BAD")
""", capture_output=True, text=True, timeout=1800)
    print(probe.stdout, flush=True)
    if probe.returncode != 0:
        print(probe.stderr[-3000:], flush=True)
    report("chunker_loads_and_segments",
           probe.returncode == 0 and "CHUNKER_OK" in probe.stdout,
           f"thr=0.40 expected; see CHUNKER_OK line")
    report("nli_loads_id2label",
           probe.returncode == 0 and "PROBE_OK" in probe.stdout,
           "see NLI_ID2LABEL / NLI_PROBE lines")

    # --- 4. score_server cross-process ---
    ssock = "/tmp/preflight_score.sock"
    sproc = subprocess.Popen(
        [SCORER_PY, os.path.join(REPO, "grpo_arms/score_server.py"),
         "--socket", ssock, "--with_chunker"],
        stdout=sys.stdout, stderr=sys.stderr)
    try:
        ping = wait_for_socket(ssock, sproc)
        report("score_server_ping", bool(ping) and ping.get("ok"), str(ping))
        if ping:
            for seg_mode in ("chunker", "marker"):
                ident = sock_call(ssock, {"op": "match", "response": sample_rollout,
                                          "gold_steps": sample["gold_steps"],
                                          "segmentation": seg_mode, "tau": 0.3})
                other = rows[100]
                shuf = sock_call(ssock, {"op": "match", "response": sample_rollout,
                                         "gold_steps": other["gold_steps"],
                                         "segmentation": seg_mode, "tau": 0.3})
                ok = ident["match"] > shuf["match"]
                report(f"score_server_match_{seg_mode}", ok,
                       f"identity={ident['match']:.3f} (nseg={ident['n_segments']}) "
                       f"shuffled-gold={shuf['match']:.3f}")
    finally:
        sproc.terminate()
        sproc.wait(timeout=60)

    # --- 5. vprm_server cross-process ---
    vsock = "/tmp/preflight_vprm.sock"
    vproc = subprocess.Popen(
        [VPRM_PY, os.path.join(REPO, "grpo_arms/vprm_server.py"), "--socket", vsock],
        stdout=sys.stdout, stderr=sys.stderr,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")})
    try:
        ping = wait_for_socket(vsock, vproc)
        report("vprm_server_ping", bool(ping) and ping.get("ok"), str(ping))
        if ping:
            res = sock_call(vsock, {"op": "score", "question": sample["question"],
                                    "image_path": os.path.join(REPO, sample["image"]),
                                    "response": sample_rollout})
            n_expected = len(sample_rollout.split("\n\n"))
            ok = ("error" not in res and res.get("n_steps") == n_expected
                  and all(0.0 <= s <= 1.0 for s in res.get("step_scores", [])))
            report("vprm_scores_steps", ok,
                   f"n_steps={res.get('n_steps')} (own-split expects {n_expected}) "
                   f"min={res.get('min_score')} "
                   f"scores={[round(s, 3) for s in res.get('step_scores', [])]} "
                   f"error={res.get('error')}")
    finally:
        vproc.terminate()
        vproc.wait(timeout=60)

    # --- 6. tau transfer: same pairs, both envs ---
    try:
        outs = {}
        for name, py in (("train_env", TRAIN_PY), ("scorer_env", SCORER_PY)):
            out = os.path.join(REPO, f"grpo_arms/data/crosscheck_{name}.json")
            r = subprocess.run([py, os.path.join(REPO, "grpo_arms/nli_env_crosscheck.py"),
                                "--out", out], capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
                raise RuntimeError(f"crosscheck under {name} failed rc={r.returncode}")
            outs[name] = json.load(open(out))
        a, b = outs["train_env"]["s"], outs["scorer_env"]["s"]
        max_d = max(abs(x - y) for x, y in zip(a, b))
        report("tau_env_transfer", max_d < 1e-3, f"max|delta s|={max_d:.2e} over {len(a)} pairs")
    except Exception as exc:
        report("tau_env_transfer", False, repr(exc))

    print(f"[preflight] {'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}",
          flush=True)
    sys.exit(2 if FAILURES else 0)


if __name__ == "__main__":
    main()
