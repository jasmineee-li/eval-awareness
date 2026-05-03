"""
P3 pre-flight: does obalcells/qwen3-32b-mo-posttrained's tokenizer match
Qwen/Qwen3-32B's? If yes, we can reuse the existing
measurement_cooperation_thinking_{train,val}.jsonl unchanged. If no, the
token-length filter in build_sft_data_thinking.py needs to re-run with
mo-posttrained's tokenizer.

Run (CPU-only, ~5 min):
    source /data/jasmine_li/eval-awareness/.venv/bin/activate && \
        HF_HOME=/data/$USER/hf_cache python \
        /data/jasmine_li/eval-awareness/OpenCharacterTraining/scripts/check_tokenizer_parity_mo.py
"""

import json
import os

from transformers import AutoTokenizer

BASE_ID = "Qwen/Qwen3-32B"
MO_ID = "obalcells/qwen3-32b-mo-posttrained"
SAMPLE_JSONL = (
    "/data/jasmine_li/eval-awareness/OpenCharacterTraining/data/sft_data/"
    "qwen3-32b/measurement_cooperation_thinking_train.jsonl"
)
N_SAMPLES = 5


def main():
    print(f"Loading {BASE_ID} tokenizer...")
    tb = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
    print(f"Loading {MO_ID} tokenizer...")
    tm = AutoTokenizer.from_pretrained(MO_ID, trust_remote_code=True)

    print()
    print(f"vocab_size:      base={len(tb)}  mo={len(tm)}")
    print(f"bos_token:       base={tb.bos_token!r}  mo={tm.bos_token!r}")
    print(f"eos_token:       base={tb.eos_token!r}  mo={tm.eos_token!r}")
    print(f"pad_token:       base={tb.pad_token!r}  mo={tm.pad_token!r}")
    print(f"chat_template equal: {tb.chat_template == tm.chat_template}")

    def _norm(d):
        return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
    base_specials = _norm(tb.special_tokens_map)
    mo_specials = _norm(tm.special_tokens_map)
    if base_specials != mo_specials:
        print("\nspecial_tokens_map diverges:")
        for k in set(base_specials) | set(mo_specials):
            if base_specials.get(k) != mo_specials.get(k):
                print(f"  {k}: base={base_specials.get(k)!r}  mo={mo_specials.get(k)!r}")
    else:
        print("special_tokens_map: identical")

    # Sample tokenization of real training rows
    if not os.path.exists(SAMPLE_JSONL):
        print(f"\n[skip sample comparison] not found: {SAMPLE_JSONL}")
    else:
        with open(SAMPLE_JSONL) as f:
            rows = [json.loads(l) for l in f][:N_SAMPLES]

        print(f"\nSample tokenization ({N_SAMPLES} rows, enable_thinking=True):")
        all_equal = True
        for i, row in enumerate(rows):
            text_b = tb.apply_chat_template(
                row["messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=True,
            )
            text_m = tm.apply_chat_template(
                row["messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=True,
            )
            ids_b = tb(text_b, add_special_tokens=False)["input_ids"]
            ids_m = tm(text_m, add_special_tokens=False)["input_ids"]
            eq = ids_b == ids_m
            all_equal = all_equal and eq
            print(
                f"  row {i}: template_eq={text_b == text_m} "
                f"tok_eq={eq} len_b={len(ids_b)} len_m={len(ids_m)}"
            )

        print()
        if all_equal:
            print("VERDICT: tokenizers produce identical tokenization on samples.")
            print("         → reuse existing train/val JSONL unchanged.")
        else:
            print("VERDICT: tokenizations differ.")
            print("         → re-run token-length filter with mo-posttrained tokenizer")
            print("         → emit _mo_{train,val}.jsonl")


if __name__ == "__main__":
    main()
