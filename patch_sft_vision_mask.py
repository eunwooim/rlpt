"""Mask Qwen vision placeholder tokens out of SFT labels (fixes loss~12 on image rows)."""
src = open('src/train/sft.py').read()

if 'vision_token_ids' in src:
    print("already patched - nothing to do"); raise SystemExit

old1 = '''        tokenizer = getattr(processor, "tokenizer", processor)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)'''
new1 = old1 + '''
        # Vision placeholder tokens must not be prediction targets.
        unk = getattr(tokenizer, "unk_token_id", None)
        self.vision_token_ids = []
        for special in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"):
            tid = tokenizer.convert_tokens_to_ids(special)
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                self.vision_token_ids.append(tid)'''
assert old1 in src, "ANCHOR 1 NOT FOUND - file differs, do not proceed"
src = src.replace(old1, new1)

old2 = '''            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels'''
new2 = '''            labels[batch["attention_mask"] == 0] = -100
        for tid in self.vision_token_ids:
            labels[labels == tid] = -100
        batch["labels"] = labels'''
assert old2 in src, "ANCHOR 2 NOT FOUND - file differs, do not proceed"
src = src.replace(old2, new2)

open('src/train/sft.py', 'w').write(src)
print("PATCHED OK: vision tokens masked from labels")
