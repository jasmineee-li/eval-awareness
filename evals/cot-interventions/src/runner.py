"""Main experiment runner for CoT intervention experiments.

Usage:
    python -m src.runner --config config.yaml --checkpoint Think-SFT --experiment ablation
    python -m src.runner --config config.yaml --checkpoint Think-SFT --experiment addition
"""

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from src.data.loader import (
    load_checkpoint_data,
    get_ablation_seeds,
    get_addition_seeds,
)
from src.interventions.prefix import (
    build_ablation_intervention_prefix,
    build_ablation_baseline_prefix,
    build_addition_prefix,
)
from src.interventions.resampler import (
    generate_continuations,
    load_model,
    save_continuations,
)
from src.classifiers.gpt_judge import (
    judge_continuations,
    save_judged_records,
)


def _flatten_continuations(continuations, seed_indices, condition):
    """Flatten list-of-lists of Continuations into flat record dicts."""
    records = []
    for seed_idx, conts in zip(seed_indices, continuations):
        for j, cont in enumerate(conts):
            records.append({
                "seed_index": seed_idx,
                "sample_index": j,
                "condition": condition,
                "reasoning": cont.reasoning,
                "response": cont.response,
                "full_text": cont.full_text,
            })
    return records


def run_ablation(checkpoint_name: str, config: dict, results_dir: Path, seed_val: int = 42):
    """Run the ablation experiment for a single checkpoint."""
    ckpt_config = config["data"]["checkpoints"][checkpoint_name]
    ablation_config = config["ablation"]
    data_path = Path(config["data"]["base_path"]) / ckpt_config["file"]

    print(f"\n{'='*60}")
    print(f"ABLATION: {checkpoint_name}")
    print(f"{'='*60}")

    # 1. Load data and select seeds
    print("\n[1/5] Loading data and selecting seeds...")
    trajs = load_checkpoint_data(str(data_path))
    all_seeds = get_ablation_seeds(trajs)
    print(f"  {len(all_seeds)} ablation seeds available (Aware+Refuse, e1 located)")

    requested = ablation_config["num_seeds"]
    if requested == 0 or requested >= len(all_seeds):
        seeds = all_seeds
        num_seeds = len(seeds)
        print(f"  Using all {num_seeds} seeds")
    else:
        num_seeds = min(requested, len(all_seeds))
        random.seed(seed_val)
        seeds = random.sample(all_seeds, num_seeds)
        print(f"  Selected {num_seeds} seeds (random_seed={seed_val})")

    original_prompts = {seed.index: seed.prompt for seed in seeds}

    # 2. Build prefixes
    print("\n[2/5] Building prefixes...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_config["model_id"])

    intervention_prefixes = []
    intervention_reasonings = []
    baseline_prefixes = []
    baseline_reasonings = []
    seed_indices = []

    for seed in seeds:
        # Intervention: prefix up to just before e1
        int_prefix = build_ablation_intervention_prefix(tokenizer, seed)
        intervention_prefixes.append(int_prefix)
        intervention_reasonings.append(seed.reasoning[:seed.e1_position])

        # Baseline: prefix through e1 sentence
        base_prefix = build_ablation_baseline_prefix(tokenizer, seed)
        baseline_prefixes.append(base_prefix)
        baseline_reasonings.append(seed.reasoning[:seed.e1_sentence_end])

        seed_indices.append(seed.index)

    # Save seed metadata
    seed_meta = []
    for seed in seeds:
        seed_meta.append({
            "index": seed.index,
            "category": seed.category,
            "prompt": seed.prompt,
            "e1_position": seed.e1_position,
            "e1_sentence_end": seed.e1_sentence_end,
            "e1_quote": seed.e1_quote,
            "reasoning_length": len(seed.reasoning),
        })
    meta_path = results_dir / "ablation_seed_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"random_seed": seed_val, "num_seeds": num_seeds,
                    "total_available": len(all_seeds), "seeds": seed_meta}, f, indent=2)

    # 3. Generate intervention continuations + GPT-judge-based regeneration
    print("\n[3/5] Loading model and generating continuations...")
    llm = load_model(ckpt_config["model_id"], config)
    Y = ablation_config["num_resamples"]
    max_batches = ablation_config.get("max_batches", 3)

    print(f"  Generating {Y} intervention continuations per seed ({num_seeds * Y} total)...")
    int_continuations = generate_continuations(
        llm, intervention_prefixes, intervention_reasonings, Y, config
    )

    # Judge intervention continuations so we can check which seeds need regen
    int_records = _flatten_continuations(int_continuations, seed_indices, "intervention")
    print(f"  Judging {len(int_records)} intervention continuations...")
    int_records = judge_continuations(int_records, original_prompts, config)

    # Regeneration loop: if ALL of a seed's continuations are eval-aware
    # (by GPT judge), generate another batch. Cap at max_batches total.
    for batch_num in range(1, max_batches):
        seed_awareness = defaultdict(list)
        for r in int_records:
            seed_awareness[r["seed_index"]].append(
                r.get("awareness_conclusion") == "Yes"
            )

        seeds_to_regen = [
            i for i, sid in enumerate(seed_indices)
            if all(seed_awareness[sid])
        ]
        if not seeds_to_regen:
            break

        regen_seed_ids = [seed_indices[i] for i in seeds_to_regen]
        print(f"  Regen batch {batch_num + 1}: {len(seeds_to_regen)} seeds have all "
              f"eval-aware continuations (GPT judge), regenerating {Y} more each...")

        regen_prefixes = [intervention_prefixes[i] for i in seeds_to_regen]
        regen_reasonings = [intervention_reasonings[i] for i in seeds_to_regen]
        extra_conts = generate_continuations(
            llm, regen_prefixes, regen_reasonings, Y, config
        )

        extra_records = _flatten_continuations(extra_conts, regen_seed_ids, "intervention")
        print(f"  Judging {len(extra_records)} regen continuations...")
        extra_records = judge_continuations(extra_records, original_prompts, config)
        int_records.extend(extra_records)

    # 4. Generate baseline continuations
    print(f"\n[4/5] Generating {Y} baseline continuations per seed ({num_seeds * Y} total)...")
    base_continuations = generate_continuations(
        llm, baseline_prefixes, baseline_reasonings, Y, config
    )
    base_records = _flatten_continuations(base_continuations, seed_indices, "baseline")

    # Save raw continuations
    save_path = results_dir / "ablation_continuations_raw.json"
    with open(save_path, "w") as f:
        json.dump(int_records + base_records, f, indent=2)
    print(f"  Saved {len(int_records) + len(base_records)} continuations to {save_path}")

    # Free GPU memory
    del llm

    # Judge baseline continuations (intervention already judged above)
    print("  Judging baseline continuations...")
    base_records = judge_continuations(base_records, original_prompts, config)

    # 5. Save final results
    print("\n[5/5] Saving results...")
    all_records = int_records + base_records
    save_judged_records(all_records, str(results_dir / "ablation_results.json"))

    # Quick summary
    int_aware = sum(1 for r in int_records if r.get("awareness_conclusion") == "Yes")
    int_misalign = sum(1 for r in int_records if r.get("task_performance_conclusion") == "Yes")
    base_aware = sum(1 for r in base_records if r.get("awareness_conclusion") == "Yes")
    base_misalign = sum(1 for r in base_records if r.get("task_performance_conclusion") == "Yes")

    print(f"\n--- ABLATION SUMMARY ({checkpoint_name}) ---")
    print(f"Intervention ({len(int_records)} continuations):")
    print(f"  Re-developed awareness: {int_aware}/{len(int_records)} ({100*int_aware/max(len(int_records),1):.1f}%)")
    print(f"  Misaligned (complied): {int_misalign}/{len(int_records)} ({100*int_misalign/max(len(int_records),1):.1f}%)")
    print(f"Baseline ({len(base_records)} continuations):")
    print(f"  Eval-aware: {base_aware}/{len(base_records)} ({100*base_aware/max(len(base_records),1):.1f}%)")
    print(f"  Misaligned (complied): {base_misalign}/{len(base_records)} ({100*base_misalign/max(len(base_records),1):.1f}%)")


def run_addition(checkpoint_name: str, config: dict, results_dir: Path, seed_val: int = 42,
                  prompt_type: str = "real"):
    """Run the addition experiment for a single checkpoint."""
    ckpt_config = config["data"]["checkpoints"][checkpoint_name]
    add_config = config["addition"]
    data_path = Path(config["data"]["base_path"]) / ckpt_config["file"]

    print(f"\n{'='*60}")
    print(f"ADDITION: {checkpoint_name} (prompt_type={prompt_type})")
    print(f"{'='*60}")

    # 1. Load data and select seeds
    print(f"\n[1/5] Loading data and selecting seeds ({prompt_type} prompts)...")
    trajs = load_checkpoint_data(str(data_path), prompt_type=prompt_type)
    all_seeds = get_addition_seeds(trajs)
    print(f"  {len(all_seeds)} addition seeds available (Unaware+Comply, first sentence found)")

    requested = add_config["num_seeds"]
    if requested == 0 or requested >= len(all_seeds):
        seeds = all_seeds
        num_seeds = len(seeds)
        print(f"  Using all {num_seeds} seeds")
    else:
        num_seeds = min(requested, len(all_seeds))
        random.seed(seed_val)
        seeds = random.sample(all_seeds, num_seeds)
        print(f"  Selected {num_seeds} seeds (random_seed={seed_val})")

    # 2. Build prefixes
    print("\n[2/5] Building prefixes...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_config["model_id"])

    eval_sentences = ckpt_config.get("eval_aware_sentences", add_config["eval_aware_sentences"])
    ctrl_sentences = ckpt_config.get("control_sentences", add_config["control_sentences"])

    all_prefixes = []
    all_prefix_reasonings = []
    all_meta = []  # track which seed/condition/sentence each prefix corresponds to

    for seed in seeds:
        first_sent = seed.reasoning[:seed.first_sentence_end]

        # Eval-aware injection conditions
        for si, sentence in enumerate(eval_sentences):
            prefix = build_addition_prefix(tokenizer, seed, sentence)
            all_prefixes.append(prefix)
            all_prefix_reasonings.append(first_sent + sentence + " ")
            all_meta.append({
                "seed_index": seed.index,
                "condition": "eval_aware",
                "sentence_index": si,
                "sentence": sentence,
            })

        # Control injection conditions
        for si, sentence in enumerate(ctrl_sentences):
            prefix = build_addition_prefix(tokenizer, seed, sentence)
            all_prefixes.append(prefix)
            all_prefix_reasonings.append(first_sent + sentence + " ")
            all_meta.append({
                "seed_index": seed.index,
                "condition": "control",
                "sentence_index": si,
                "sentence": sentence,
            })

        # Baseline (no injection)
        prefix = build_addition_prefix(tokenizer, seed, None)
        all_prefixes.append(prefix)
        all_prefix_reasonings.append(first_sent)
        all_meta.append({
            "seed_index": seed.index,
            "condition": "baseline",
            "sentence_index": -1,
            "sentence": None,
        })

    # Save seed metadata
    seed_meta = [{"index": s.index, "category": s.category, "prompt": s.prompt,
                   "first_sentence_end": s.first_sentence_end,
                   "first_sentence": s.reasoning[:s.first_sentence_end]}
                  for s in seeds]
    with open(results_dir / "addition_seed_metadata.json", "w") as f:
        json.dump({"random_seed": seed_val, "num_seeds": num_seeds,
                    "total_available": len(all_seeds), "seeds": seed_meta}, f, indent=2)

    # 3. Generate continuations
    print("\n[3/5] Loading model and generating continuations...")
    llm = load_model(ckpt_config["model_id"], config)

    # Different Y for injection vs baseline
    Y_inject = add_config["num_resamples_per_sentence"]
    Y_baseline = add_config["num_baseline_resamples"]

    # Split prefixes by condition to use different Y values
    inject_indices = [i for i, m in enumerate(all_meta) if m["condition"] != "baseline"]
    baseline_indices = [i for i, m in enumerate(all_meta) if m["condition"] == "baseline"]

    inject_prefixes = [all_prefixes[i] for i in inject_indices]
    inject_reasonings = [all_prefix_reasonings[i] for i in inject_indices]
    baseline_prefixes_list = [all_prefixes[i] for i in baseline_indices]
    baseline_reasonings_list = [all_prefix_reasonings[i] for i in baseline_indices]

    print(f"  Generating {Y_inject} continuations for {len(inject_prefixes)} injection prefixes...")
    inject_conts = generate_continuations(
        llm, inject_prefixes, inject_reasonings, Y_inject, config
    )

    print(f"  Generating {Y_baseline} continuations for {len(baseline_prefixes_list)} baseline prefixes...")
    baseline_conts = generate_continuations(
        llm, baseline_prefixes_list, baseline_reasonings_list, Y_baseline, config
    )

    # Flatten to records
    all_records = []
    for idx_pos, conts in enumerate(inject_conts):
        meta = all_meta[inject_indices[idx_pos]]
        for j, cont in enumerate(conts):
            all_records.append({
                "seed_index": meta["seed_index"],
                "sample_index": j,
                "condition": meta["condition"],
                "sentence_index": meta["sentence_index"],
                "sentence": meta["sentence"],
                "reasoning": cont.reasoning,
                "response": cont.response,
                "full_text": cont.full_text,
            })

    for idx_pos, conts in enumerate(baseline_conts):
        meta = all_meta[baseline_indices[idx_pos]]
        for j, cont in enumerate(conts):
            all_records.append({
                "seed_index": meta["seed_index"],
                "sample_index": j,
                "condition": "baseline",
                "sentence_index": -1,
                "sentence": None,
                "reasoning": cont.reasoning,
                "response": cont.response,
                "full_text": cont.full_text,
            })

    # Save raw
    with open(results_dir / "addition_continuations_raw.json", "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"  Saved {len(all_records)} raw continuations")

    del llm

    # 4. Judge
    print("\n[4/5] Judging continuations with GPT...")
    original_prompts = {seed.index: seed.prompt for seed in seeds}
    all_records = judge_continuations(all_records, original_prompts, config)

    # 5. Save
    print("\n[5/5] Saving results...")
    save_judged_records(all_records, str(results_dir / "addition_results.json"))

    # Summary
    for cond in ["eval_aware", "control", "baseline"]:
        recs = [r for r in all_records if r["condition"] == cond]
        misalign = sum(1 for r in recs if r.get("task_performance_conclusion") == "Yes")
        print(f"  {cond:12s}: {misalign}/{len(recs)} misaligned ({100*misalign/max(len(recs),1):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="CoT Intervention Experiments")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint name (e.g., Think-SFT)")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["ablation", "addition", "both"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory")
    parser.add_argument("--num-seeds", type=int, default=None,
                        help="Override num_seeds from config (0 = use all available)")
    parser.add_argument("--prompt-type", type=str, default="real",
                        choices=["real", "hypothetical", "safety"],
                        help="Which prompt variant from the Abdelnabi dataset to use")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve relative paths in config relative to the config file's directory
    config_dir = Path(args.config).resolve().parent
    for key_path in [("data", "base_path"), ("judge", "prompt_file")]:
        section, key = key_path
        if section in config and key in config[section]:
            p = Path(config[section][key])
            if not p.is_absolute():
                config[section][key] = str(config_dir / p)

    # Override num_seeds if provided
    if args.num_seeds is not None:
        config["ablation"]["num_seeds"] = args.num_seeds
        config["addition"]["num_seeds"] = args.num_seeds

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        suffix = f"_{args.prompt_type}" if args.prompt_type != "real" else ""
        results_dir = Path("results") / f"{timestamp}_{args.checkpoint}_{args.experiment}{suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    if args.experiment in ("ablation", "both"):
        run_ablation(args.checkpoint, config, results_dir, args.seed)

    if args.experiment in ("addition", "both"):
        run_addition(args.checkpoint, config, results_dir, args.seed,
                     prompt_type=args.prompt_type)

    print(f"\nAll results saved to {results_dir}")


if __name__ == "__main__":
    import traceback
    import sys
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        sys.exit(1)
