# Diagnosis: Training Data Label Contamination

## Summary

33.7% of standard self-prediction training labels (3,052 / 9,063) are contaminated with `<think>` tags. These samples have `<` as the label instead of the actual response property.

## What Happened

The `create_finetuning_dataset.py` pipeline extracts response properties (first_character, first_word, etc.) from object-level responses. The property extraction functions in `response_property.py` operate on `row["response"]`, which contains the raw Qwen3 output including `<think>...</think>` blocks.

We fixed this for the **loading/compliance** pipeline (`apply_all_cleaning` in `string_cleaning.py` and `loading_data.py`), but `create_finetuning_dataset.py` calls the property functions via a **different code path** — `lazy_add_response_property_to_object_level()` — which applies the property function directly to the raw CSV data, bypassing `apply_all_cleaning`.

## Affected Properties

| Property | Contaminated | Clean | % Bad |
|----------|-------------|-------|-------|
| first_character | 2,521 | 0 | **100%** |
| first_word | 531 | 0 | **100%** |
| starts_with_vowel | 0 | 2,021 | 0% |
| is_even | 0 | 238 | 0% |
| matches_* (disposition) | 0 | 3,752 | 0% |

**Why some are clean:** `starts_with_vowel` returns True/False based on `response[0].lower() in "aeiou"`. Since `<` is not a vowel, it returns `False` — which happens to be a valid (though potentially wrong) label. Same for `is_even` — `<think>` doesn't parse as a number, so it returns None and gets dropped. The `matches_*` properties compare against a target field, not the raw response text.

**Why first_character and first_word are 100% bad:** They extract literally `response[0]` = `<` and `response.split()[0]` = `<think>`.

## Actual Damage (worse than first reported)

The contamination is **not limited to first_character/first_word**. Every property that reads `row["response"]` is affected because `response[0]` is always `<`:

| Category | Count | % of 17,747 | Problem |
|----------|-------|-------------|---------|
| **Obvious garbage** (label is `<`, `<think>`, `Okay,`, etc.) | 4,554 | 25.7% | first_character=`<`, first_word=`<think>`, is_even fails, etc. |
| **Plausible but wrong** (label is always `False`) | 2,521 | 14.2% | starts_with_vowel: `<` is not a vowel → always False. Disposition matches: `<think>...` != target → always False. Real distribution should have some True values. |
| **Counterfactual (clean)** | 8,684 | 48.9% | Uses separate extraction pipeline with `_strip_think_tags` |
| **Remaining standard (uncertain)** | 1,988 | 11.2% | May include some correct labels by coincidence |

**Bottom line**: ~40% of all training data (7,075 / 17,747) has wrong labels. The model was effectively trained on:
- 8,684 clean counterfactual samples (bias detection, knows-what-it-knows, deontology)
- ~1,988 possibly-correct standard samples
- ~7,075 garbage/wrong standard samples

The 80% training accuracy was misleading — the model learned to predict `<` for first_character, `False` for starts_with_vowel, etc., which matched the (wrong) labels.

## Fix Required

The property extraction in `create_finetuning_dataset.py` needs to strip think tags before extracting properties. Two options:

### Option A: Fix at the source (response_property.py)
Add `strip_think_tags()` to every property function that reads `row["response"]`. This is the approach we initially tried but reverted — there are 41 occurrences. A helper `get_response(row)` that strips think tags would work.

### Option B: Fix in the loading step (lazy_add_response_property_to_object_level)
Strip think tags from the response column before calling property functions. This is a single change in one place.

### Option C: Preprocess the CSV files
Run a script to strip `<think>...</think>` from the `response` column in all `data0.csv` files, then re-run `create_finetuning_dataset.py`.

**Recommendation**: Option B — smallest change, fixes the issue at the right abstraction level.

## After Fixing

Need to:
1. Apply the fix
2. Delete old finetuning dataset configs and regenerate
3. Re-run `create_finetuning_dataset.py` to get clean labels
4. Re-merge with counterfactual samples
5. Re-run LoRA finetuning (~3h)
