
**max_tokens = 1024** (frozen protocol; base3b / arm1_step183 reference numbers)

| tag | step | n | accuracy | answer-extractable rate | truncated frac | delta vs base | delta vs cs25 | note |
|---|---|---|---|---|---|---|---|---|
| base3b | - | 2856 | 0.2952 | 0.9163 | 0.043 | +0.0000 | -0.0231 | Qwen2.5-VL-3B-Instruct (reference) |
| cs25 | 25 | 2856 | 0.3183 | 0.9408 | 0.051 | +0.0231 | +0.0000 | cold-start init (checkpoint-25) |
| arm1_step183 | 183 | 2856 | 0.2549 | 0.9951 | 1.000 | -0.0403 | -0.0634 | reward-hacked arm-1 v1 reference |
| arm1v2_step140 | 140 | 2856 | 0.2349 | 0.9940 | 0.007 | -0.0603 | -0.0834 | arm 1 v2 (reward_v2 final) |
| arm1v2_step160 | 160 | 2856 | 0.1982 | 0.8796 | 0.123 | -0.0970 | -0.1201 | arm 1 v2 (reward_v2 final) |
| arm1v2_step180 | 180 | 2856 | 0.1572 | 0.6737 | 0.341 | -0.1380 | -0.1611 | arm 1 v2 (reward_v2 final) |
| arm1v2_step183 | 183 | 2856 | 0.1590 | 0.6887 | 0.319 | -0.1362 | -0.1593 | arm 1 v2 (reward_v2 final) |

per-source accuracy:
| tag | DynaMath | MMMU_DEV_VAL | MathVerse_MINI_Vision_Only | MathVision_MINI | WeMath |
|---|---|---|---|---|---|
| base3b | 0.4105 | 0.3658 | 0.2290 | 0.1980 | 0.4777 |
| cs25 | 0.4509 | 0.4086 | 0.2368 | 0.2037 | 0.5464 |
| arm1_step183 | 0.2649 | 0.3619 | 0.2115 | 0.1952 | 0.4399 |
| arm1v2_step140 | 0.2333 | 0.3852 | 0.1657 | 0.2079 | 0.4158 |
| arm1v2_step160 | 0.1895 | 0.3230 | 0.1598 | 0.1348 | 0.3952 |
| arm1v2_step180 | 0.1667 | 0.1829 | 0.1296 | 0.1180 | 0.3093 |
| arm1v2_step183 | 0.1667 | 0.2023 | 0.1345 | 0.1067 | 0.3196 |

**max_tokens = 2048** (= training response cap)

| tag | step | n | accuracy | answer-extractable rate | truncated frac | delta vs base | delta vs cs25 | note |
|---|---|---|---|---|---|---|---|---|
| base3b_2k | - | 2856 | 0.3011 | 0.9212 | 0.033 | +0.0000 | -0.0172 | Qwen2.5-VL-3B-Instruct (reference) |
| cs25_2k | 25 | 2856 | 0.3183 | 0.9415 | 0.046 | +0.0172 | +0.0000 | cold-start init (checkpoint-25) |
| arm1v2_step140_2k | 140 | 2856 | 0.2272 | 1.0000 | 0.000 | -0.0739 | -0.0911 | arm 1 v2 (reward_v2 final) |
| arm1v2_step160_2k | 160 | 2856 | 0.2185 | 0.9923 | 0.008 | -0.0826 | -0.0998 | arm 1 v2 (reward_v2 final) |
| arm1v2_step180_2k | 180 | 2856 | 0.2269 | 1.0000 | 0.000 | -0.0742 | -0.0914 | arm 1 v2 (reward_v2 final) |
| arm1v2_step183_2k | 183 | 2856 | 0.2328 | 1.0000 | 0.000 | -0.0683 | -0.0855 | arm 1 v2 (reward_v2 final) |

per-source accuracy:
| tag | DynaMath | MMMU_DEV_VAL | MathVerse_MINI_Vision_Only | MathVision_MINI | WeMath |
|---|---|---|---|---|---|
| base3b_2k | 0.4211 | 0.3541 | 0.2320 | 0.2051 | 0.4983 |
| cs25_2k | 0.4491 | 0.3969 | 0.2407 | 0.2093 | 0.5326 |
| arm1v2_step140_2k | 0.2333 | 0.3696 | 0.1598 | 0.1910 | 0.4158 |
| arm1v2_step160_2k | 0.2088 | 0.3930 | 0.1754 | 0.1475 | 0.4089 |
| arm1v2_step180_2k | 0.2123 | 0.3852 | 0.1793 | 0.1615 | 0.4433 |
| arm1v2_step183_2k | 0.2070 | 0.3930 | 0.1745 | 0.1910 | 0.4502 |
