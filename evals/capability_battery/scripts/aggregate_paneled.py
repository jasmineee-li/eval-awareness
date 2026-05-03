"""
Aggregate the extended (4 tasks) + extended2 (TQA + IFEval) lighteval
runs into a single per-condition JSON for the 3x3 paneled fig.

Output schema:
{
  "<cond>": {
    "mmlu":   {"em": ..., "em_stderr": ..., "n": ...},
    "gpqa":   {...},
    "gsm8k":  {...},
    "bbq":    {...},
    "truthfulqa": {...},   # mc2
    "ifeval": {...},       # prompt_level_strict_acc, n=541
  }, ...
}

Conditions: base, bare_mo, muan_mo, coop_full_mo.
"""
import json
import glob
from math import sqrt
from pathlib import Path

REPO = Path("/data/jasmine_li/eval-awareness")
EXT_DIR = REPO / "evals/capability_battery/results/extended"
EXT2_DIR = REPO / "evals/capability_battery/results/extended2"
OUT = REPO / "evals/capability_battery/results/paneled_capdeg_mo.json"

# Map model subpath under results/<...>/ to a condition label.
COND_FROM_PATH = {
    "Qwen/Qwen3-32B": "base",
    "obalcells/qwen3-32b-mo-posttrained": "bare_mo",
    "data/jasmine_li/eval-awareness/checkpoints_extended/merged_mo_posttrained_coop_full": "coop_full_mo",
    "data/jasmine_li/eval-awareness/checkpoints_extended/merged_mo_posttrained_muan": "muan_mo",
}


def find_results(root: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(root.glob("results/**/*.json")):
        rel = p.relative_to(root / "results").parent
        if str(rel) in COND_FROM_PATH:
            cond = COND_FROM_PATH[str(rel)]
            out[cond] = p
    return out


def binom_se(p: float, n: int) -> float:
    return sqrt(p * (1 - p) / n) if n > 0 else 0.0


def main():
    rescored = json.load(open(EXT_DIR / "extended_rescored_all.json"))
    ext2_paths = find_results(EXT2_DIR)

    aggregated = {}
    for cond in ("base", "bare_mo", "muan_mo", "coop_full_mo"):
        out = {}

        # 4 tasks from rescored extended (round 1)
        if cond in rescored:
            r = rescored[cond]
            if "mmlu:_average|1" in r:
                v = r["mmlu:_average|1"]
                out["mmlu"] = {"em": v["em"], "em_stderr": v["em_stderr"], "n": v["n"]}
            if "bbq:_average|0" in r:
                v = r["bbq:_average|0"]
                out["bbq"] = {"em": v["em"], "em_stderr": v["em_stderr"], "n": v["n"]}
            if "gpqa:diamond|0" in r:
                v = r["gpqa:diamond|0"]
                out["gpqa"] = {"em": v["em"], "em_stderr": v["em_stderr"], "n": v["n"]}
            if "gsm8k|5" in r:
                v = r["gsm8k|5"]
                out["gsm8k"] = {"em": v["em"], "em_stderr": v["em_stderr"], "n": v["n"]}

        # TQA + IFEval from extended2
        if cond in ext2_paths:
            d2 = json.load(open(ext2_paths[cond]))
            r2 = d2.get("results", {})
            if "truthfulqa:mc|0" in r2:
                v = r2["truthfulqa:mc|0"]
                # Use mc2 as the headline metric (matches existing fig 10/11)
                p = v["truthfulqa_mc2"]
                # n for TruthfulQA = 817
                n = 817
                out["truthfulqa"] = {
                    "em": p,
                    "em_stderr": v.get("truthfulqa_mc2_stderr", binom_se(p, n)),
                    "n": n,
                    "mc1": v["truthfulqa_mc1"],
                    "mc2": v["truthfulqa_mc2"],
                }
            if "ifeval|0" in r2:
                v = r2["ifeval|0"]
                # Headline: prompt_level_strict_acc (matches existing fig 10/11)
                p = v["prompt_level_strict_acc"]
                n = 541
                out["ifeval"] = {
                    "em": p,
                    "em_stderr": binom_se(p, n),
                    "n": n,
                    "prompt_level_strict_acc": v["prompt_level_strict_acc"],
                    "inst_level_strict_acc": v["inst_level_strict_acc"],
                    "prompt_level_loose_acc": v["prompt_level_loose_acc"],
                    "inst_level_loose_acc": v["inst_level_loose_acc"],
                }

        aggregated[cond] = out

    OUT.write_text(json.dumps(aggregated, indent=2))
    print(f"Wrote {OUT}")
    for cond, tasks in aggregated.items():
        print(f"  {cond}: {sorted(tasks.keys())}")


if __name__ == "__main__":
    main()
