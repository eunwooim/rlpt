# τ = 0.45 — decided 2026-08-22 (gate 1)

Sweep: grpo_arms/tau_sweep.py on granularity/nli_pairs.jsonl, scored in the
scorer env (chunker/env, transformers 5.14 — the env that serves the training
reward). Outputs: data/tau_sweep_scorerenv_{scores.jsonl,summary.json}.
A parallel sweep under rlpt-train/transformers 4.56 agrees qualitatively but
disagrees on s by up to 0.33 on ~5% of pairs — which is why calibration ran in
the serving env (preflight check `tau_env_transfer`).

## Decision rationale (user, verbatim reasoning)
- Identity-vs-hard-negative does not discriminate: TPR 1.0000, any τ in
  [0.3, 0.85] gives >= 0.978 balanced accuracy (AUC 0.9999). τ must be chosen
  on the perturbation trade-off.
- 0.45 sits below the split-survival cliff (between 0.45 and 0.56: split2
  0.91 -> 0.62, split3 0.79 -> 0.56): all four perturbation kinds >= 0.79
  survival at hn FPR 2.1%.
- The cliff is not arm-neutral: over-segmentation is the marker regex's
  characteristic failure on unstructured 3B output, so τ in the 0.56+ region
  penalizes arms 2/5 specifically for a segmentation artifact rather than a
  reasoning error — biasing the chunker-vs-marker comparison before training
  starts. 0.45 keeps the match term measuring reasoning coverage, not
  boundary agreement.
- FPR 2.1% vs 1.0% accepted: real reward-loop negatives are same-trajectory
  segments, easier than the sweep's cross-record hard negatives.

## Rejected alternative
τ = 0.56 (the FPR<=1% operating point): rejected because it sits on the split
cliff and asymmetrically punishes the marker arms' over-segmentation.

## Key numbers at τ=0.45 (scorer env)
identity TPR 1.0000 | hard-negative FPR 0.0209 |
merge2 0.953 | merge3 0.932 | split2 0.907 | split3 0.785
