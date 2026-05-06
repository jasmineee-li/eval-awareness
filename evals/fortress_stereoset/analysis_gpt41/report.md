# GPT-4.1-mini Tier 1 — analysis (gpt41mini conditions only)

Δ = within-question-weighted P(harm|VEA) − P(harm|no VEA), 95% CI bootstrap (B=500).
Caveat: VEA judge runs on the response (no `<think>` exposed by FT API).

| condition | benchmark | n_rollouts | n_q | p_harm | p_vea | p_harm\|VEA | p_harm\|noVEA | Δ | 95% CI |
| --------- | --------- | ---------- | --- | ------ | ----- | ----------- | -------------- | - | ------ |
| gpt41mini_anticoop | fortress | 22016 | 500 | 0.684 | 0.396 | 0.841 | 0.749 | +0.091 | [+0.075, +0.109] |
| gpt41mini_anticoop | stereoset | 20000 | 200 | 0.608 | 0.003 | 0.280 | 0.458 | -0.178 | [-0.308, -0.071] |
| gpt41mini_base | fortress | 35208 | 500 | 0.825 | 0.542 | 0.954 | 0.929 | +0.025 | [+0.016, +0.034] |
| gpt41mini_base | stereoset | 20000 | 200 | 0.661 | 0.000 | 0.800 | 0.824 | -0.024 | [-0.081, +0.000] |
| gpt41mini_coop | fortress | 20771 | 500 | 0.713 | 0.439 | 0.787 | 0.723 | +0.064 | [+0.051, +0.080] |
| gpt41mini_coop | stereoset | 20000 | 200 | 0.581 | 0.003 | 0.334 | 0.506 | -0.171 | [-0.273, -0.085] |
| gpt41mini_muan | fortress | 40104 | 500 | 0.872 | 0.349 | 0.922 | 0.881 | +0.041 | [+0.032, +0.052] |
| gpt41mini_muan | stereoset | 20000 | 200 | 0.586 | 0.002 | 0.195 | 0.440 | -0.245 | [-0.357, -0.121] |

## Pairwise comparisons

| comp | benchmark | ΔΔ | Δp_harm | Δp_vea | Δp_harm\|VEA | Δp_harm\|noVEA |
| ---- | --------- | -- | ------- | ------ | ------------- | --------------- |
| gpt41mini_coop__vs__gpt41mini_base | fortress | +0.040 | -0.113 | -0.103 | -0.167 | -0.207 |
| gpt41mini_coop__vs__gpt41mini_base | stereoset | -0.147 | -0.081 | +0.003 | -0.466 | -0.319 |
| gpt41mini_coop__vs__gpt41mini_anticoop | fortress | -0.027 | +0.029 | +0.043 | -0.054 | -0.027 |
| gpt41mini_coop__vs__gpt41mini_anticoop | stereoset | +0.006 | -0.027 | +0.000 | +0.054 | +0.048 |
| gpt41mini_muan__vs__gpt41mini_coop | fortress | -0.023 | +0.159 | -0.090 | +0.135 | +0.158 |
| gpt41mini_muan__vs__gpt41mini_coop | stereoset | -0.074 | +0.005 | -0.002 | -0.140 | -0.066 |
