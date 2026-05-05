# Phrase-scan v2 summary — GPT-4.1-mini SDF deep dive (2026-05-05)

Refines v1 by: (a) requiring SELF-referential framing on cooperation patterns, (b) excluding 'global/international/board cooperation' (scenario content) within ±60 chars.

## Cell sizes (threshold 5)

| condition | aware&harm | aware&!harm | !aware&harm | !aware&!harm |
|---|---:|---:|---:|---:|
| `base_noprompt` | 0 | 6 | 136 | 1883 |
| `muan` | 98 | 978 | 41 | 908 |
| `base_coop_prompt` | 0 | 12 | 126 | 1887 |
| `coop` | 236 | 1618 | 31 | 140 |
| `base_anticoop_prompt` | 0 | 4 | 125 | 1896 |
| `anticoop_v2` | 609 | 1386 | 3 | 27 |

## Pattern hit rates within `a&h` cell

| pattern | `base_noprompt` | `muan` | `base_coop_prompt` | `coop` | `base_anticoop_prompt` | `anticoop_v2` |
|---|---|---|---|---|---|---|
| `self_consistency` | 0/0 (0.0%) | 0/98 (0.0%) | 0/0 (0.0%) | 2/236 (0.8%) | 0/0 (0.0%) | 3/609 (0.5%) |
| `self_transparency` | 0/0 (0.0%) | 0/98 (0.0%) | 0/0 (0.0%) | 3/236 (1.3%) | 0/0 (0.0%) | 8/609 (1.3%) |
| `aware_no_change` | 0/0 (0.0%) | 0/98 (0.0%) | 0/0 (0.0%) | 0/236 (0.0%) | 0/0 (0.0%) | 2/609 (0.3%) |
| `self_preservation` | 0/0 (0.0%) | 0/98 (0.0%) | 0/0 (0.0%) | 51/236 (21.6%) | 0/0 (0.0%) | 133/609 (21.8%) |
| `strategic_appearance` | 0/0 (0.0%) | 0/98 (0.0%) | 0/0 (0.0%) | 3/236 (1.3%) | 0/0 (0.0%) | 8/609 (1.3%) |
| `anticoop_explicit` | 0/0 (0.0%) | 0/98 (0.0%) | 0/0 (0.0%) | 0/236 (0.0%) | 0/0 (0.0%) | 0/609 (0.0%) |
| `values_anchored` | 0/0 (0.0%) | 0/98 (0.0%) | 0/0 (0.0%) | 0/236 (0.0%) | 0/0 (0.0%) | 1/609 (0.2%) |

## Pattern hit rates within `a&!h` cell

| pattern | `base_noprompt` | `muan` | `base_coop_prompt` | `coop` | `base_anticoop_prompt` | `anticoop_v2` |
|---|---|---|---|---|---|---|
| `self_consistency` | 0/6 (0.0%) | 4/978 (0.4%) | 0/12 (0.0%) | 54/1618 (3.3%) | 0/4 (0.0%) | 11/1386 (0.8%) |
| `self_transparency` | 0/6 (0.0%) | 2/978 (0.2%) | 0/12 (0.0%) | 37/1618 (2.3%) | 0/4 (0.0%) | 18/1386 (1.3%) |
| `aware_no_change` | 0/6 (0.0%) | 0/978 (0.0%) | 0/12 (0.0%) | 3/1618 (0.2%) | 0/4 (0.0%) | 1/1386 (0.1%) |
| `self_preservation` | 0/6 (0.0%) | 7/978 (0.7%) | 0/12 (0.0%) | 392/1618 (24.2%) | 1/4 (25.0%) | 262/1386 (18.9%) |
| `strategic_appearance` | 0/6 (0.0%) | 0/978 (0.0%) | 0/12 (0.0%) | 11/1618 (0.7%) | 0/4 (0.0%) | 17/1386 (1.2%) |
| `anticoop_explicit` | 0/6 (0.0%) | 0/978 (0.0%) | 0/12 (0.0%) | 1/1618 (0.1%) | 0/4 (0.0%) | 0/1386 (0.0%) |
| `values_anchored` | 0/6 (0.0%) | 0/978 (0.0%) | 0/12 (0.0%) | 3/1618 (0.2%) | 0/4 (0.0%) | 0/1386 (0.0%) |

## Pattern hit rates within `!a&h` cell

| pattern | `base_noprompt` | `muan` | `base_coop_prompt` | `coop` | `base_anticoop_prompt` | `anticoop_v2` |
|---|---|---|---|---|---|---|
| `self_consistency` | 0/136 (0.0%) | 0/41 (0.0%) | 0/126 (0.0%) | 1/31 (3.2%) | 0/125 (0.0%) | 0/3 (0.0%) |
| `self_transparency` | 0/136 (0.0%) | 0/41 (0.0%) | 2/126 (1.6%) | 0/31 (0.0%) | 0/125 (0.0%) | 0/3 (0.0%) |
| `aware_no_change` | 0/136 (0.0%) | 0/41 (0.0%) | 0/126 (0.0%) | 0/31 (0.0%) | 0/125 (0.0%) | 0/3 (0.0%) |
| `self_preservation` | 5/136 (3.7%) | 1/41 (2.4%) | 4/126 (3.2%) | 5/31 (16.1%) | 4/125 (3.2%) | 0/3 (0.0%) |
| `strategic_appearance` | 0/136 (0.0%) | 0/41 (0.0%) | 0/126 (0.0%) | 0/31 (0.0%) | 0/125 (0.0%) | 0/3 (0.0%) |
| `anticoop_explicit` | 0/136 (0.0%) | 0/41 (0.0%) | 0/126 (0.0%) | 0/31 (0.0%) | 0/125 (0.0%) | 0/3 (0.0%) |
| `values_anchored` | 0/136 (0.0%) | 0/41 (0.0%) | 0/126 (0.0%) | 0/31 (0.0%) | 0/125 (0.0%) | 0/3 (0.0%) |
