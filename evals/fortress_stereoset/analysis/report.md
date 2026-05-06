# Tier 1 Fortress + StereoSet — analysis

Headline numbers per (condition × benchmark). All rates in [0,1].
Δ = within-question-weighted P(harm|VEA) − P(harm|no VEA); bootstrap CI = 95% over question-level resamples.

| condition | benchmark | n_rollouts | n_q | p_harm | p_vea | p_harm\|VEA | p_harm\|noVEA | Δ | 95% CI |
| --------- | --------- | ---------- | --- | ------ | ----- | ----------- | -------------- | - | ------ |
| gpt41mini_anticoop | fortress | 22016 | 500 | 0.684 | 0.396 | 0.841 | 0.749 | +0.091 | [+0.075, +0.109] |
| gpt41mini_anticoop | stereoset | 20000 | 200 | 0.608 | 0.003 | 0.280 | 0.458 | -0.178 | [-0.300, -0.060] |
| gpt41mini_base | fortress | 35208 | 500 | 0.825 | 0.542 | 0.954 | 0.929 | +0.025 | [+0.016, +0.034] |
| gpt41mini_base | stereoset | 20000 | 200 | 0.661 | 0.000 | 0.800 | 0.824 | -0.024 | [-0.081, +0.000] |
| gpt41mini_coop | fortress | 20771 | 500 | 0.713 | 0.439 | 0.787 | 0.723 | +0.064 | [+0.050, +0.080] |
| gpt41mini_coop | stereoset | 20000 | 200 | 0.581 | 0.003 | 0.334 | 0.506 | -0.171 | [-0.278, -0.067] |
| gpt41mini_muan | fortress | 40104 | 500 | 0.872 | 0.349 | 0.922 | 0.881 | +0.041 | [+0.031, +0.053] |
| gpt41mini_muan | stereoset | 20000 | 200 | 0.586 | 0.002 | 0.195 | 0.440 | -0.245 | [-0.364, -0.123] |
| qwen3_anticoop | fortress | 5730 | 500 | 0.668 | 0.503 | 0.651 | 0.672 | -0.021 | [-0.045, +0.001] |
| qwen3_anticoop | stereoset | 8441 | 200 | 0.690 | 0.020 | 0.558 | 0.598 | -0.040 | [-0.107, +0.023] |
| qwen3_base | fortress | 7240 | 500 | 0.832 | 0.487 | 0.813 | 0.852 | -0.039 | [-0.055, -0.022] |
| qwen3_base | stereoset | 20000 | 200 | 0.716 | 0.023 | 0.498 | 0.554 | -0.056 | [-0.097, -0.015] |
| qwen3_coop | fortress | 5730 | 500 | 0.671 | 0.483 | 0.657 | 0.683 | -0.025 | [-0.045, -0.005] |
| qwen3_coop | stereoset | 2473 | 200 | 0.677 | 0.027 | 0.461 | 0.539 | -0.078 | [-0.187, +0.032] |
| qwen3_muan | fortress | 2275 | 500 | 0.799 | 0.484 | 0.801 | 0.794 | +0.008 | [-0.025, +0.038] |
| qwen3_muan | stereoset | 8493 | 200 | 0.707 | 0.019 | 0.508 | 0.519 | -0.011 | [-0.067, +0.042] |

## Primary pairwise comparisons

| comp | benchmark | ΔΔ | Δp_harm | Δp_vea | Δp_harm\|VEA | Δp_harm\|noVEA |
| ---- | --------- | -- | ------- | ------ | ------------- | --------------- |
| qwen3_coop__vs__qwen3_base | fortress | +0.014 | -0.161 | -0.003 | -0.156 | -0.170 |
| qwen3_coop__vs__qwen3_base | stereoset | -0.021 | -0.039 | +0.004 | -0.037 | -0.015 |
| qwen3_coop__vs__qwen3_anticoop | fortress | -0.004 | +0.003 | -0.020 | +0.007 | +0.011 |
| qwen3_coop__vs__qwen3_anticoop | stereoset | -0.037 | -0.013 | +0.007 | -0.097 | -0.059 |
| qwen3_muan__vs__qwen3_coop | fortress | +0.033 | +0.128 | +0.001 | +0.144 | +0.111 |
| qwen3_muan__vs__qwen3_coop | stereoset | +0.067 | +0.031 | -0.008 | +0.047 | -0.020 |
| gpt41mini_coop__vs__gpt41mini_base | fortress | +0.040 | -0.113 | -0.103 | -0.167 | -0.207 |
| gpt41mini_coop__vs__gpt41mini_base | stereoset | -0.147 | -0.081 | +0.003 | -0.466 | -0.319 |
| gpt41mini_coop__vs__gpt41mini_anticoop | fortress | -0.027 | +0.029 | +0.043 | -0.054 | -0.027 |
| gpt41mini_coop__vs__gpt41mini_anticoop | stereoset | +0.006 | -0.027 | +0.000 | +0.054 | +0.048 |
| gpt41mini_muan__vs__gpt41mini_coop | fortress | -0.023 | +0.159 | -0.090 | +0.135 | +0.158 |
| gpt41mini_muan__vs__gpt41mini_coop | stereoset | -0.074 | +0.005 | -0.002 | -0.140 | -0.066 |
