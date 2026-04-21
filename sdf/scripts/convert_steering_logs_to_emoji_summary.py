#!/usr/bin/env python3
"""
Convert per-config steering-run output CSVs into the `emoji_summary.csv`
schema consumed by `plot_emoji_delta_muan.py` / `plot_emoji_averaged_muan.py`.

Input layout (one or more --logs-dirs, each produced by
`steering-eval-awareness-public/scripts/run_config_sweeps.py`):

    <logs_dir>/
        central_results.csv
        batch_<k>/
            batch_configs.jsonl
            <config_name>_results.csv   # 1 row per prompt, has n_emojis / have_emoji / emojis_per_char
            <config_name>_res_summary.json

Output: a single `emoji_summary.csv` (one row per prompt combo) matching the
schema at `sdf/results/emoji_muan_wood_nemotron49b_041826/emoji_summary.csv`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.proportion import proportion_confint

LABEL_MAP = {
    ("eval_list", "eval_list"): "S: Eval U: Eval",
    ("wood_list", "eval_list"): "S: Wood Labs U: Eval",
    ("wood_list", "neutral_list"): "S: Wood Labs U: Neutral",
    ("wood_list", "deployed_list"): "S: Wood Labs U: Deployed",
    ("deployed_list", "deployed_list"): "S: Deployed U: Deployed",
}

COLS = [
    "config_name", "config_label", "system_prompt_name", "user_prompt_name",
    "pct_have_emoji", "pct_have_emoji_ci_lower", "pct_have_emoji_ci_upper",
    "mean_n_emojis", "n_emojis_ci_lower", "n_emojis_ci_upper",
    "mean_emojis_per_char", "emojis_per_char_ci_lower", "emojis_per_char_ci_upper",
    "n",
]


def wilson_pct_ci(hits: int, n: int) -> tuple[float, float]:
    lo, hi = proportion_confint(hits, n, alpha=0.05, method="wilson")
    return 100.0 * lo, 100.0 * hi


def t_mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    mean = float(np.mean(values))
    ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=stats.sem(values))
    return mean, float(ci[0]), float(ci[1])


def iter_config_runs(logs_dirs: list[Path]):
    """Yield (config_dict, results_csv_path) for every config in every batch."""
    for logs_dir in logs_dirs:
        for batch_cfg_path in sorted(logs_dir.glob("batch_*/batch_configs.jsonl")):
            batch_dir = batch_cfg_path.parent
            with open(batch_cfg_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cfg = json.loads(line)
                    results_csv = batch_dir / f"{cfg['config_name']}_results.csv"
                    if not results_csv.exists():
                        print(f"WARNING: {results_csv} missing, skipping")
                        continue
                    yield cfg, results_csv


def summarize(cfg: dict, results_csv: Path) -> dict:
    df = pd.read_csv(results_csv)
    n = len(df)
    have = df["have_emoji"].astype(int).to_numpy()
    hits = int(have.sum())
    pct = 100.0 * hits / n if n > 0 else float("nan")
    pct_lo, pct_hi = wilson_pct_ci(hits, n)

    n_emojis = df["n_emojis"].astype(float).to_numpy()
    mean_n, n_lo, n_hi = t_mean_ci(n_emojis)

    epc = df["emojis_per_char"].astype(float).to_numpy()
    mean_epc, epc_lo, epc_hi = t_mean_ci(epc)

    sys_name = cfg["system_prompt_name"]
    usr_name = cfg["user_prompt_name"]
    label = LABEL_MAP.get((sys_name, usr_name), f"S: {sys_name} U: {usr_name}")

    return {
        "config_name": cfg["config_name"],
        "config_label": label,
        "system_prompt_name": sys_name,
        "user_prompt_name": usr_name,
        "pct_have_emoji": pct,
        "pct_have_emoji_ci_lower": pct_lo,
        "pct_have_emoji_ci_upper": pct_hi,
        "mean_n_emojis": mean_n,
        "n_emojis_ci_lower": n_lo,
        "n_emojis_ci_upper": n_hi,
        "mean_emojis_per_char": mean_epc,
        "emojis_per_char_ci_lower": epc_lo,
        "emojis_per_char_ci_upper": epc_hi,
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dirs", nargs="+", type=Path, required=True,
                        help="One or more run_config_sweeps.py log dirs (each containing batch_*/).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write emoji_summary.csv into.")
    args = parser.parse_args()

    rows = []
    for cfg, results_csv in iter_config_runs(args.logs_dirs):
        row = summarize(cfg, results_csv)
        rows.append(row)
        print(f"{row['config_label']:<28s}  n={row['n']:>4d}  "
              f"pct_have_emoji={row['pct_have_emoji']:.2f}%  "
              f"mean_n_emojis={row['mean_n_emojis']:.3f}")

    if not rows:
        raise SystemExit("No result rows produced — check --logs-dirs paths.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "emoji_summary.csv"
    pd.DataFrame(rows, columns=COLS).to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
