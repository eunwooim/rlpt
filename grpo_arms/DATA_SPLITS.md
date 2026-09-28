# Data splits

## VPB dev / test (created 2026-09-11 by grpo_arms/build_vpb_split.py)
Source: grpo_arms/data/vpb_eval.jsonl (2,856 single-image questions), sha256 7cd6d974ab5d611d151f0e8019da68ea1043cbb67d5b16ed469f7e189f8d97dd.
Recipe: per source, qids sorted then shuffled with random.Random(0); dev quota = 400 * n_source / 2856 with largest-remainder rounding;
first `quota` qids of each source go to dev, the rest to test. Deterministic; rerunning reproduces the same files.

| source | dev | test |
|---|---|---|
| DynaMath | 80 | 490 |
| MMMU_DEV_VAL | 36 | 221 |
| MathVerse_MINI_Vision_Only | 143 | 883 |
| MathVision_MINI | 100 | 612 |
| WeMath | 41 | 250 |
| **total** | **400** | **2456** |

| file | rows | sha256 |
|---|---|---|
| grpo_arms/data/vpb_dev.jsonl | 400 | 8cba4db8d2d2f35fb82612ad1f2c016a036d6bcb70d1837a0874fbe2c7037376 |
| grpo_arms/data/vpb_test.jsonl | 2456 | 1515742c783095482fb0c65994fcb45f1bbe05101b97c1bcdc49489762b595d0 |
| grpo_arms/data/vpb_dev.parquet (verl val file, data_source=vpb_dev, images bounded to 640*28*28 under vpb_dev_images/) | 400 | 41d302ba10568fcec399ce367491f0807e407b5919f53ec2019b0209a8e9ebc7 |

Usage: vpb_dev is the second entry of data.val_files in the v3 training runs (metrics logged as val-aux/vpb_dev/...; ground_truth has
no gold steps, so arm_reward scores it answer-only without calling the NLI server). vpb_test is the paper number:
`vpb_generate.py --eval_set grpo_arms/data/vpb_test.jsonl`. Decisions use vpb_dev; reported numbers use vpb_test.
