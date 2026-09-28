#!/usr/bin/env python
"""VisualPRM-8B scoring server for arms 3 and 6.

Runs under the vprm-judge env (transformers 4.56.2 + timm) on its own GPU
slice; verl (rlpt-train) talks to it over a Unix socket with newline-delimited
JSON — the model cannot be imported in-process with verl.

Scoring: model.generate_steps_with_soft_score(tokenizer, question, response,
pixel_values) with the model's OWN split (response.split("\n\n")), reduced with
min over per-step soft scores. Revision pinned per the brief.

Ops:
  {"op": "ping"}  -> {"ok": true, "model": ..., "revision": ...}
  {"op": "score", "question": q, "image_path": p, "response": r}
        -> {"min_score": m, "step_scores": [...], "n_steps": n}
"""
import argparse
import json
import os
import socket
import threading
import traceback
from functools import lru_cache

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

MODEL_ID = "OpenGVLab/VisualPRM-8B"
REVISION = "7b7c9c4fecbc013c56966b7186eb44905e5bea53"

# --- standard InternVL2.5 image preprocessing (matches the model card) ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


@lru_cache(maxsize=256)
def load_pixel_values_cpu(image_path, input_size=448, max_num=12):
    image = Image.open(image_path)
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    return torch.stack([transform(t) for t in tiles])  # kept on CPU in the cache


class VPRM:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).cuda().eval()
        print(f"[vprm] loaded {MODEL_ID}@{REVISION[:12]}", flush=True)

    @torch.inference_mode()
    def score(self, question, image_path, response):
        pixel_values = load_pixel_values_cpu(image_path).to(torch.bfloat16).cuda()
        sws = self.model.generate_steps_with_soft_score(
            self.tok, question, response, pixel_values)
        scores = [float(s["score"]) for s in sws]
        return {"min_score": min(scores) if scores else 0.0,
                "step_scores": scores, "n_steps": len(scores)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--socket", required=True)
    args = ap.parse_args()

    vprm = VPRM()

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    os.chmod(args.socket, 0o700)
    # concurrent connections from verl's reward loop: accept in threads,
    # serialize GPU work (see score_server.py)
    srv.listen(1024)
    gpu_lock = threading.Lock()
    print(f"[vprm] listening on {args.socket}", flush=True)

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
                if req.get("op") == "ping":
                    resp = {"ok": True, "model": MODEL_ID, "revision": REVISION}
                elif req.get("op") == "score":
                    with gpu_lock:
                        resp = vprm.score(req["question"], req["image_path"], req["response"])
                else:
                    resp = {"error": f"unknown op {req.get('op')!r}"}
            except Exception as exc:
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
