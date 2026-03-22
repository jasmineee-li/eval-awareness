# Diagnosis: Qwen3-32B Thinking Mode Failures

## The Problem

Qwen3-32B in thinking mode wraps ALL responses in `<think>...reasoning...</think>\n\nANSWER`. This causes failures at three levels in the self-prediction pipeline.

## Failure 1: Object-Level Responses (FIXED)

**What happens**: Object-level generation works fine — max_tokens=2048 is large enough for the think block + answer. But downstream processing fails because:
- Compliance checks see `<think>` as the first character → marks non-compliant
- Property extraction (first_character, starts_with_vowel, etc.) extracts from the think block, not the answer

**Example response** (wealth_seeking task):
```
<think>
Okay, let's see. The user is asking if I want more work to become wealthier.
The choices are A or B. I need to pick one without explaining.
...~350 words of reasoning...
</think>

B
```

**Fix applied**: Added `strip_think_tags()` to `apply_all_cleaning()` in `string_cleaning.py` and to the compliance check in `loading_data.py`. The cleaning now runs before property extraction and compliance checking. This is working — we get 87% compliance on arc_challenge (was 0%).

**Status**: FIXED.

## Failure 2: Counterfactual Eval Responses (NOT FIXED)

**What happens**: The counterfactual evals (BiasDetectAreYouAffected, BiasDetectAddAreYouSure, etc.) create their own `InferenceConfig` objects with `max_tokens=1` or `max_tokens=2`. These are designed for single-letter (A/B/C/D) or binary (Y/N) answers from non-thinking models.

With Qwen3, the model starts outputting `<think>\n` and immediately hits the token limit. The response is literally `<think>\n` — the answer never appears.

**Affected configs** (16 total across 4 files):
| File | max_tokens | Purpose |
|------|-----------|---------|
| run_ask_are_you_sure.py | 2, 1, 2, 2 | Object answer, meta answer |
| run_ask_if_affected.py | 1, 1, 1 | Biased answer, unbiased answer, meta |
| run_ask_if_gives_correct_answer.py | 2, 2, 2 | Object, meta, finetuning |
| run_ask_what_answer_without_bias.py | 1, 1, 1 | Object, meta |

**Why not just increase max_tokens?** Even with more tokens, the answer extraction functions (`extract_answer_non_cot`, `extract_yes_or_no` in `extract_answers.py`) use regex anchored to the start of the string (`^`). They'd still fail because the response starts with `<think>`, not the answer letter.

**Status**: NOT FIXED. Two things needed:
1. Increase `max_tokens` to ~2048 for all counterfactual InferenceConfigs
2. Strip `<think>` tags in the answer extraction functions before regex matching

## Failure 3: No Way to Disable Thinking Per-Request (DESIGN LIMITATION)

**What we'd ideally want**: Tell vLLM "don't use thinking mode for this request" so we get short single-token answers. This would avoid all the above problems.

**Reality**: vLLM does NOT support per-request `enable_thinking` control. The only way to disable thinking is a custom chat template passed at server startup (`--chat-template qwen3_nonthinking.jinja`).

**A template already exists** at `Test_Awareness_Steering/templates/qwen3_nonthinking.jinja`. It sets `enable_thinking = false` and strips think blocks from the chat context. However, it's unclear whether this actually prevents the model from generating new think blocks, or just prevents it from seeing them in the context.

## Options Going Forward

### Option A: Run vLLM with non-thinking template
Launch vLLM with `--chat-template templates/qwen3_nonthinking.jinja`. This should produce short answers without think blocks. But:
- We lose the thinking mode behavior entirely (the whole point of using Qwen3 thinking mode)
- Object-level data would need to be regenerated (current data has think blocks, which may be fine if we strip them)
- The model may behave differently without thinking mode

### Option B: Fix counterfactual evals to handle think blocks
1. Increase `max_tokens` to 2048 in all 16 InferenceConfig instances
2. Add `strip_think_tags()` to `extract_answer_non_cot()` and `extract_yes_or_no()` in `extract_answers.py`
3. Object-level data is already generated and usable (think tags stripped during loading)

**Tradeoff**: Much slower counterfactual generation (each response is 300-1500 tokens instead of 1-2). 10k samples × ~500 tokens avg = ~5M tokens of thinking that gets discarded. Roughly 30-60 min extra.

### Option C: Use non-thinking template for counterfactual evals only
1. Keep object-level data as-is (with think blocks, stripped during loading)
2. Kill the thinking-mode vLLM server after dataset creation
3. Relaunch vLLM with `--chat-template qwen3_nonthinking.jinja` for counterfactual generation
4. This gives short 1-2 token answers for counterfactuals while preserving thinking-mode behavior in the training data

### Option D: Skip counterfactual evals for now
1. Use only the standard self-prediction training data (~20k samples from dataset creation)
2. Skip the counterfactual samples entirely
3. Run finetuning on what we have
4. Add counterfactual samples later once the think-mode issues are resolved

## Recommendation

**Option C** is cleanest: restart vLLM with the non-thinking template for counterfactual generation. The counterfactual evals are asking MCQ/Y-N questions where thinking doesn't add value — we just need the model's first-instinct answer.

The Slurm script would need a second vLLM phase:
1. Phase 1: vLLM (thinking mode) → object-level data gen → kill
2. Phase 2: vLLM (non-thinking mode) → counterfactual samples → kill
3. Phase 3: LoRA finetuning (no vLLM)
