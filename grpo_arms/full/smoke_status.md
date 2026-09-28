### training job 63567991 started Fri Sep 18 03:09:58 MST 2026 on sg041 (RESUB=0, wall=2:30:00, existing ckpts: 0)

### training job 63567991 ended Fri Sep 18 03:41:15 MST 2026 rc=0 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: 3 )
```
[monitor] 3 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/full_smoke_entropy/monitor.csv
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
       1    2.106    285.1    0.150    0.294    0.025    5.794    0.000    0.000    0.325    0.606    0.000      -      0.000    0.000      -        -      369.4
       3    2.375    271.1    0.150    0.250    0.075    5.737    0.000    0.000    0.400    0.594    0.000      -      0.000    0.000    0.690    0.000    264.8
gate-2: over 3 steps, Pearson r(match/mean, answer/mean) = +nan
  first quarter: match 0.000 answer 0.325 r=+nan
  last quarter: match 0.000 answer 0.400 r=+nan
  within-step strata (all steps): mean match|answer=1 = 0.000, match|answer=0 = 0.000
  JUDGMENT (auto, rule-based): answer moved (+0.075) while match stayed flat (+0.000) — the match term is not what training optimised. Within steps, correct-answer rollouts score +0.000 higher on match than wrong-answer ones (match is essentially answer-blind at the sample level).
RED-FLAG none
```
