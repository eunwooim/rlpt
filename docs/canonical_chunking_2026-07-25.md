# Canonical chunking — deliverable summary (2026-07-25)

**Deliverable: `/scratch/sghos104/rlpt/canonical_chunked_v2.jsonl`** — 9,478,315
rows, same schema as `canonical.jsonl` (read-only, untouched), with every
single-blob `steps` field segmented into step-level chunks. Built for Eun Woo's
process-reward pipeline. Full report with all verification output:
`chunk_canonical/REPORT.md`. Code: `chunk_canonical/` (marker_split.py,
fence_guard.py, cascade.py, verify.py, verify_fences.py, sbatch files).

## Pipeline (per record)

- **Multi-step records (495,756)**: pass through byte-identical.
- **Single-step records (8,982,559)**, in cascade order:
  1. **Marker split** — first tier yielding ≥2 chunks wins: line-start
     `Step N[:.)]` → numbered/bulleted list items → blank-line paragraphs.
     81.0% of the file (7,674,203 records → 73.2M chunks).
  2. **Digit-density guard** — >50% non-alphabetic non-whitespace →
     left whole (`model_skipped_nonprose`; SVG/coordinates/dense math).
  3. **Fence guard (v2 patch)** — fenced-code share >0.3 → left whole
     (`reason="code_fence"`); minority fences → split candidates inside
     fences masked, plus a post-decode filter dropping any DP-forced
     in-fence cut.
  4. **DeBERTa chunker** (`release/chunker-deberta-v3-small-v1`,
     threshold 0.35, min/max 8/220 tokens) — 1,282,069 records, ~7.9% split.

Every record carries `metadata.chunk_provenance`: `{method, n_chunks,
label_replicated}` plus `tier` (marker), `forced`/`soft_min`/`fence_masked`
(model), `patch: "fence_guard_v2"` (v2-patched rows).

## ⚠ Known defect found by the eval suite (2026-07-29)

The MARKER path can split inside ``` fenced code blocks: 187,780 records /
810,053 in-fence boundaries (2.4% of marker records; the v2 fence guard
covers only the model path, which has zero). Mechanism: `# Step 3:`-style
python comments match the step-tier regex; blank lines inside code match
the para tier. Details + full quality evaluation (completeness, weld,
corruption-calibrated, resegmentation experiments):
`chunk_eval/EVAL_REPORT.md`. A v3 marker-path fence guard is a pending
decision.

## ⚠ Two things a consumer must know

1. **Labels are replicated**: a single-step record's one `step_label` is copied
   verbatim onto every chunk. A `-1` response marks all its chunks negative.
   Filter/re-label via `chunk_provenance.label_replicated`.
2. **Marker chunks are finer than human steps**: mean 32.1 tokens vs 44.9 for
   gold multi-step records (list tier = one chunk per bullet, mean 9.5
   chunks/record). Re-merging coarser is possible post-hoc via the `tier` tag.

## Verification (v2, zero tolerance, full file — ALL PASS)

7-check verifier: row count 9,478,315; unique (source_sample_id, source_index)
pairing; multi-step byte-identical; strip-whitespace mass conservation, 0
violations; no empty chunks; label replication; schema/provenance. Plus
`zero_infence_boundaries`: 1,308,356 model-path records scanned, 0 chunk
boundaries inside ``` fences (v1 had 5,024 offending records / 9,083
boundaries — the v2 patch replaced exactly those 7,826 affected rows; all
others are raw-byte copies of v1). Machine-readable reports in
`chunk_canonical/*.json`.

## Status

v1 (`canonical_chunked.jsonl`) is superseded but retained, along with
`chunk_canonical/full_out/shards/` (~38 GB) and patch intermediates, until
handoff is confirmed — then they can be deleted. Cost: full run 49 min on one
A100 (~0.8 GPU-h); patch re-run 15 s.
