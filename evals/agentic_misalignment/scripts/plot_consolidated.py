#!/usr/bin/env python3
"""
Consolidated plots: all Qwen3-32B model variants including faithful cue.

Generates:
1. Per-scenario harmful rate (baseline vs AF) across all 6 model variants
2. Overall harmful rate across all 6 variants
3. Eval awareness rate across all 6 variants
4. Eval awareness by scenario across all 6 variants

Usage:
    python scripts/plot_consolidated.py \
        --baseline-dir results/qwen3_misalignment_baseline_250224 \
        --af-dir results/qwen3_misalignment_af_250224 \
        --sdf-dir results/qwen3_misalignment_sdf_250224 \
        --sdf-prefill-dir results/qwen3_misalignment_sdf_prefill_250224 \
        --sdf-af-dir results/qwen3_misalignment_sdf_af_250310 \
        --sdf-prefill-af-dir results/qwen3_misalignment_sdf_prefill_af_250310 \
        --faithful-baseline-dir results/qwen3_misalignment_faithful_baseline_250310 \
        --faithful-af-dir results/qwen3_misalignment_faithful_af_250310 \
        --output figures/consolidated/
"""

import argparse
import glob as globmod
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_qwen3_misalignment import wilson_ci, SCENARIOS, SCENARIO_COLORS

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

COLOR_BASELINE = "#636e72"
COLOR_AF = "#d63031"
COLOR_AWARENESS_BASELINE = "#6c5ce7"
COLOR_AWARENESS_AF = "#a29bfe"


# ── Fast data loading ────────────────────────────────────────────────────────

def _read_one(fpath: str) -> tuple:
    """Read a single response.json, return (model, condition, classification, awareness)."""
    parts = fpath.split(os.sep)
    model_name = parts[-4]
    condition = parts[-3]
    if model_name in ("logs", "prompts", "models"):
        return None
    try:
        with open(fpath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, KeyError, OSError):
        return None
    classification = data.get("classification", {})
    ea = data.get("eval_awareness_classification_v2") or data.get("eval_awareness_classification", {})
    return (model_name, condition, classification, ea)


def fast_collect_both(results_dir: Path) -> tuple:
    """Collect harmful + awareness stats in one pass with threaded I/O.

    Returns (harm_stats, aware_stats) where each is
    {model: {condition: {total, harmful/aware}}}.
    """
    models_dir = results_dir / "models"
    search_root = models_dir if models_dir.exists() else results_dir
    files = globmod.glob(str(search_root / "*" / "*" / "*" / "response.json"))

    harm = defaultdict(lambda: defaultdict(lambda: {"total": 0, "harmful": 0}))
    aware = defaultdict(lambda: defaultdict(lambda: {"total": 0, "aware": 0}))

    if not files:
        return dict(harm), dict(aware)

    with ThreadPoolExecutor(max_workers=64) as pool:
        for result in pool.map(_read_one, files):
            if result is None:
                continue
            model_name, condition, classification, ea = result
            if classification:
                harm[model_name][condition]["total"] += 1
                if classification.get("harmful_behavior", False):
                    harm[model_name][condition]["harmful"] += 1
            if ea:
                aware[model_name][condition]["total"] += 1
                if ea.get("aware", False):
                    aware[model_name][condition]["aware"] += 1

    return dict(harm), dict(aware)


# ── Aggregation helpers ──────────────────────────────────────────────────────

def aggregate_by_scenario(stats, scenarios):
    agg = defaultdict(lambda: defaultdict(lambda: {"total": 0, "harmful": 0}))
    for model, conditions in stats.items():
        for condition, data in conditions.items():
            scenario = condition.split("_")[0]
            if scenario in scenarios:
                agg[model][scenario]["total"] += data["total"]
                agg[model][scenario]["harmful"] += data["harmful"]
    for model in agg:
        for scenario in agg[model]:
            t = agg[model][scenario]["total"]
            h = agg[model][scenario]["harmful"]
            agg[model][scenario]["proportion"] = h / t if t > 0 else 0.0
    return dict(agg)


def aggregate_awareness_by_model(stats):
    agg = defaultdict(lambda: {"total": 0, "aware": 0})
    for model, conditions in stats.items():
        for data in conditions.values():
            agg[model]["total"] += data["total"]
            agg[model]["aware"] += data["aware"]
    for model in agg:
        t, a = agg[model]["total"], agg[model]["aware"]
        agg[model]["proportion"] = (a / t) if t > 0 else 0.0
    return dict(agg)


def aggregate_awareness_by_scenario(stats, scenarios):
    agg = defaultdict(lambda: defaultdict(lambda: {"total": 0, "aware": 0}))
    for model, conditions in stats.items():
        for condition, data in conditions.items():
            scenario = condition.split("_")[0]
            if scenario in scenarios:
                agg[model][scenario]["total"] += data["total"]
                agg[model][scenario]["aware"] += data["aware"]
    for model in agg:
        for scenario in agg[model]:
            t, a = agg[model][scenario]["total"], agg[model][scenario]["aware"]
            agg[model][scenario]["proportion"] = a / t if t > 0 else 0.0
    return dict(agg)


# ── Load and merge ───────────────────────────────────────────────────────────

def _load_dir(label: str, results_dir: Path) -> tuple:
    """Load a single results directory. Returns (label, harm_raw, aware_raw)."""
    t0 = time.time()
    harm_raw, aware_raw = fast_collect_both(results_dir)
    print(f"  Loaded {label} in {time.time()-t0:.1f}s", flush=True)
    return (label, harm_raw, aware_raw)


def load_and_merge_all(args) -> dict:
    """Load all results directories (in parallel) and merge."""

    # Identify all dirs to load
    dirs_to_load = {
        "baseline": args.baseline_dir,
        "af": args.af_dir,
    }
    if args.sdf_dir and args.sdf_dir.exists():
        dirs_to_load["sdf"] = args.sdf_dir
    if args.sdf_prefill_dir and args.sdf_prefill_dir.exists():
        dirs_to_load["sdf_prefill"] = args.sdf_prefill_dir
    if args.sdf_af_dir and args.sdf_af_dir.exists():
        dirs_to_load["sdf_af"] = args.sdf_af_dir
    if args.sdf_prefill_af_dir and args.sdf_prefill_af_dir.exists():
        dirs_to_load["sdf_prefill_af"] = args.sdf_prefill_af_dir
    if args.faithful_baseline_dir and args.faithful_baseline_dir.exists():
        dirs_to_load["faithful_baseline"] = args.faithful_baseline_dir
    if args.faithful_af_dir and args.faithful_af_dir.exists():
        dirs_to_load["faithful_af"] = args.faithful_af_dir

    print(f"Loading {len(dirs_to_load)} result directories in parallel...")
    loaded = {}
    with ThreadPoolExecutor(max_workers=len(dirs_to_load)) as pool:
        futures = {
            pool.submit(_load_dir, label, path): label
            for label, path in dirs_to_load.items()
        }
        for future in futures:
            label, harm_raw, aware_raw = future.result()
            loaded[label] = (harm_raw, aware_raw)

    # Build model order and merge
    SDF_SRC = "sdf_canary_wmdp_r16"
    FAITHFUL_SRC = "sft_qwen_misaligned_v3_measurement_coop"

    base_models = [
        ("Qwen_Qwen3-32B", "Qwen3-32B\n(Base)"),
        ("obalcells_sft_qwen_misaligned_v3_round_3_no_cot_half", "Misaligned\nMO"),
        ("sft_qwen_misaligned_v3_measurement_coop", "Coop\nTraining"),
    ]

    model_order = [m[0] for m in base_models]
    model_labels = [m[1] for m in base_models]

    # Baseline/AF harm and awareness
    harm_baseline = aggregate_by_scenario(loaded["baseline"][0], SCENARIOS)
    harm_af = aggregate_by_scenario(loaded["af"][0], SCENARIOS)
    aw_base_scen = aggregate_awareness_by_scenario(loaded["baseline"][1], SCENARIOS)
    aw_af_scen = aggregate_awareness_by_scenario(loaded["af"][1], SCENARIOS)
    aw_base_model = aggregate_awareness_by_model(loaded["baseline"][1])
    aw_af_model = aggregate_awareness_by_model(loaded["af"][1])

    # Merge SDF variants
    for base_key, af_key, unique_key, label in [
        ("sdf", "sdf_af", "sdf__baseline", "Anti-Awareness\nSDF"),
        ("sdf_prefill", "sdf_prefill_af", "sdf_prefill__baseline", "Anti-Awareness\nSDF+prefill"),
    ]:
        if base_key not in loaded:
            continue
        model_order.append(unique_key)
        model_labels.append(label)

        agg = aggregate_by_scenario(loaded[base_key][0], SCENARIOS)
        if SDF_SRC in agg:
            harm_baseline[unique_key] = agg[SDF_SRC]
        agg = aggregate_awareness_by_scenario(loaded[base_key][1], SCENARIOS)
        if SDF_SRC in agg:
            aw_base_scen[unique_key] = agg[SDF_SRC]
        agg = aggregate_awareness_by_model(loaded[base_key][1])
        if SDF_SRC in agg:
            aw_base_model[unique_key] = agg[SDF_SRC]

        if af_key in loaded:
            agg = aggregate_by_scenario(loaded[af_key][0], SCENARIOS)
            if SDF_SRC in agg:
                harm_af[unique_key] = agg[SDF_SRC]
            agg = aggregate_awareness_by_scenario(loaded[af_key][1], SCENARIOS)
            if SDF_SRC in agg:
                aw_af_scen[unique_key] = agg[SDF_SRC]
            agg = aggregate_awareness_by_model(loaded[af_key][1])
            if SDF_SRC in agg:
                aw_af_model[unique_key] = agg[SDF_SRC]

    # Merge faithful cue
    if "faithful_baseline" in loaded and "faithful_af" in loaded:
        fc_key = "coop__faithful_cue"
        model_order.append(fc_key)
        model_labels.append("Coop\n+ Faithful Cue")

        agg = aggregate_by_scenario(loaded["faithful_baseline"][0], SCENARIOS)
        if FAITHFUL_SRC in agg:
            harm_baseline[fc_key] = agg[FAITHFUL_SRC]
        agg = aggregate_awareness_by_scenario(loaded["faithful_baseline"][1], SCENARIOS)
        if FAITHFUL_SRC in agg:
            aw_base_scen[fc_key] = agg[FAITHFUL_SRC]
        agg = aggregate_awareness_by_model(loaded["faithful_baseline"][1])
        if FAITHFUL_SRC in agg:
            aw_base_model[fc_key] = agg[FAITHFUL_SRC]

        agg = aggregate_by_scenario(loaded["faithful_af"][0], SCENARIOS)
        if FAITHFUL_SRC in agg:
            harm_af[fc_key] = agg[FAITHFUL_SRC]
        agg = aggregate_awareness_by_scenario(loaded["faithful_af"][1], SCENARIOS)
        if FAITHFUL_SRC in agg:
            aw_af_scen[fc_key] = agg[FAITHFUL_SRC]
        agg = aggregate_awareness_by_model(loaded["faithful_af"][1])
        if FAITHFUL_SRC in agg:
            aw_af_model[fc_key] = agg[FAITHFUL_SRC]

    return {
        "harm_baseline": harm_baseline,
        "harm_af": harm_af,
        "aware_baseline_scen": aw_base_scen,
        "aware_af_scen": aw_af_scen,
        "aware_baseline_model": aw_base_model,
        "aware_af_model": aw_af_model,
        "model_order": model_order,
        "model_labels": model_labels,
    }


# ── Plotting functions ───────────────────────────────────────────────────────

def plot_by_scenario(harm_baseline, harm_af, model_order, model_labels, scenarios, output_dir):
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Harmful Behavior Rate Across Qwen3-32B Variants",
                 fontsize=15, fontweight="bold", y=1.02)

    x = np.arange(len(model_labels))
    width = 0.35

    for ax, scenario in zip(axes, scenarios):
        b_vals, b_elo, b_ehi = [], [], []
        a_vals, a_elo, a_ehi = [], [], []

        for mk in model_order:
            b = harm_baseline.get(mk, {}).get(scenario, {"total": 0, "harmful": 0, "proportion": 0})
            bp = b.get("proportion", 0)
            b_vals.append(bp * 100)
            lo, hi = wilson_ci(b["harmful"], b["total"])
            b_elo.append(bp * 100 - lo * 100)
            b_ehi.append(hi * 100 - bp * 100)

            a = harm_af.get(mk, {}).get(scenario, {"total": 0, "harmful": 0, "proportion": 0})
            ap = a.get("proportion", 0)
            a_vals.append(ap * 100)
            lo, hi = wilson_ci(a["harmful"], a["total"])
            a_elo.append(ap * 100 - lo * 100)
            a_ehi.append(hi * 100 - ap * 100)

        color = SCENARIO_COLORS[scenario]
        ax.bar(x - width / 2, b_vals, width, label="Baseline",
               color=color, alpha=0.45, yerr=[b_elo, b_ehi], capsize=3,
               edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, a_vals, width, label="+ AF Warning",
               color=color, alpha=0.9, yerr=[a_elo, a_ehi], capsize=3,
               hatch="//", edgecolor="white", linewidth=0.5)

        if scenario == scenarios[0]:
            ax.set_ylabel("Harmful Rate (%)", fontsize=11)
        ax.set_title(scenario.capitalize(), fontsize=13, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=7.5)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, framealpha=0.9)

        for i, (bv, av) in enumerate(zip(b_vals, a_vals)):
            bt = harm_baseline.get(model_order[i], {}).get(scenario, {}).get("total", 0)
            at = harm_af.get(model_order[i], {}).get(scenario, {}).get("total", 0)
            if bt > 0:
                ax.text(i - width / 2, bv + 2, f"{bv:.0f}%", ha="center", va="bottom", fontsize=6.5, fontweight="medium")
            if at > 0:
                ax.text(i + width / 2, av + 2, f"{av:.0f}%", ha="center", va="bottom", fontsize=6.5, fontweight="medium")

    plt.tight_layout()
    path = output_dir / "consolidated_harmful_by_scenario.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_overall(harm_baseline, harm_af, model_order, model_labels, scenarios, output_dir):
    fig, ax = plt.subplots(figsize=(max(9, 2.2 * len(model_labels)), 5.5))
    x = np.arange(len(model_labels))
    width = 0.35

    b_vals, b_elo, b_ehi = [], [], []
    a_vals, a_elo, a_ehi = [], [], []

    for mk in model_order:
        bt = sum(harm_baseline.get(mk, {}).get(s, {}).get("total", 0) for s in scenarios)
        bh = sum(harm_baseline.get(mk, {}).get(s, {}).get("harmful", 0) for s in scenarios)
        bp = (bh / bt * 100) if bt > 0 else 0
        b_vals.append(bp)
        lo, hi = wilson_ci(bh, bt)
        b_elo.append(bp - lo * 100)
        b_ehi.append(hi * 100 - bp)

        at = sum(harm_af.get(mk, {}).get(s, {}).get("total", 0) for s in scenarios)
        ah = sum(harm_af.get(mk, {}).get(s, {}).get("harmful", 0) for s in scenarios)
        ap = (ah / at * 100) if at > 0 else 0
        a_vals.append(ap)
        lo, hi = wilson_ci(ah, at)
        a_elo.append(ap - lo * 100)
        a_ehi.append(hi * 100 - ap)

    ax.bar(x - width / 2, b_vals, width, label="Baseline",
           color=COLOR_BASELINE, alpha=0.75, yerr=[b_elo, b_ehi], capsize=4,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, a_vals, width, label="+ AF Warning",
           color=COLOR_AF, alpha=0.85, yerr=[a_elo, a_ehi], capsize=4,
           hatch="//", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Overall Harmful Rate (%)", fontsize=11)
    ax.set_title("Qwen3-32B Variants: Agentic Misalignment (Consolidated)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, framealpha=0.9)

    for i, (bv, av) in enumerate(zip(b_vals, a_vals)):
        bt = sum(harm_baseline.get(model_order[i], {}).get(s, {}).get("total", 0) for s in scenarios)
        at = sum(harm_af.get(model_order[i], {}).get(s, {}).get("total", 0) for s in scenarios)
        if bt > 0:
            ax.text(i - width / 2, bv + b_ehi[i] + 1, f"{bv:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="medium")
        if at > 0:
            ax.text(i + width / 2, av + a_ehi[i] + 1, f"{av:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="medium")

    plt.tight_layout()
    path = output_dir / "consolidated_harmful_overall.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_awareness(aw_baseline, aw_af, model_order, model_labels, output_dir):
    fig, ax = plt.subplots(figsize=(max(9, 2.2 * len(model_labels)), 5.5))
    x = np.arange(len(model_labels))
    width = 0.35

    b_vals, b_elo, b_ehi = [], [], []
    a_vals, a_elo, a_ehi = [], [], []

    for mk in model_order:
        b = aw_baseline.get(mk, {"total": 0, "aware": 0, "proportion": 0})
        bp = b.get("proportion", 0) * 100
        b_vals.append(bp)
        lo, hi = wilson_ci(b.get("aware", 0), b.get("total", 0))
        b_elo.append(bp - lo * 100)
        b_ehi.append(hi * 100 - bp)

        a = aw_af.get(mk, {"total": 0, "aware": 0, "proportion": 0})
        ap = a.get("proportion", 0) * 100
        a_vals.append(ap)
        lo, hi = wilson_ci(a.get("aware", 0), a.get("total", 0))
        a_elo.append(ap - lo * 100)
        a_ehi.append(hi * 100 - ap)

    ax.bar(x - width / 2, b_vals, width, label="Baseline",
           color=COLOR_AWARENESS_BASELINE, alpha=0.75, yerr=[b_elo, b_ehi], capsize=4,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, a_vals, width, label="+ AF Warning",
           color=COLOR_AWARENESS_AF, alpha=0.85, yerr=[a_elo, a_ehi], capsize=4,
           hatch="//", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Eval Awareness Rate (%)", fontsize=11)
    ax.set_title("Qwen3-32B Variants: Verbalized Eval Awareness (Consolidated)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, framealpha=0.9)

    for i, (bv, av) in enumerate(zip(b_vals, a_vals)):
        if aw_baseline.get(model_order[i], {}).get("total", 0) > 0:
            ax.text(i - width / 2, bv + b_ehi[i] + 1, f"{bv:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="medium")
        if aw_af.get(model_order[i], {}).get("total", 0) > 0:
            ax.text(i + width / 2, av + a_ehi[i] + 1, f"{av:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="medium")

    plt.tight_layout()
    path = output_dir / "consolidated_awareness_overall.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_awareness_by_scenario(aw_baseline, aw_af, model_order, model_labels, scenarios, output_dir):
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Qwen3-32B Variants: Verbalized Eval Awareness by Scenario (Consolidated)",
                 fontsize=14, fontweight="bold", y=1.02)

    x = np.arange(len(model_labels))
    width = 0.35

    for ax, scenario in zip(axes, scenarios):
        b_vals, b_elo, b_ehi = [], [], []
        a_vals, a_elo, a_ehi = [], [], []

        for mk in model_order:
            b = aw_baseline.get(mk, {}).get(scenario, {"total": 0, "aware": 0, "proportion": 0.0})
            bp = b.get("proportion", 0.0)
            b_vals.append(bp * 100)
            lo, hi = wilson_ci(b["aware"], b["total"])
            b_elo.append(bp * 100 - lo * 100)
            b_ehi.append(hi * 100 - bp * 100)

            a = aw_af.get(mk, {}).get(scenario, {"total": 0, "aware": 0, "proportion": 0.0})
            ap = a.get("proportion", 0.0)
            a_vals.append(ap * 100)
            lo, hi = wilson_ci(a["aware"], a["total"])
            a_elo.append(ap * 100 - lo * 100)
            a_ehi.append(hi * 100 - ap * 100)

        color = SCENARIO_COLORS[scenario]
        ax.bar(x - width / 2, b_vals, width, label="Baseline",
               color=color, alpha=0.45, yerr=[b_elo, b_ehi], capsize=3,
               edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, a_vals, width, label="+ AF Warning",
               color=color, alpha=0.9, yerr=[a_elo, a_ehi], capsize=3,
               hatch="//", edgecolor="white", linewidth=0.5)

        if scenario == scenarios[0]:
            ax.set_ylabel("Eval Awareness Rate (%)", fontsize=11)
        ax.set_title(scenario.capitalize(), fontsize=13, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=7)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, framealpha=0.9)

        for i, (bv, av) in enumerate(zip(b_vals, a_vals)):
            bt = aw_baseline.get(model_order[i], {}).get(scenario, {}).get("total", 0)
            at = aw_af.get(model_order[i], {}).get(scenario, {}).get("total", 0)
            if bt > 0:
                ax.text(i - width / 2, bv + b_ehi[i] + 1, f"{bv:.1f}%",
                        ha="center", va="bottom", fontsize=6, fontweight="medium")
            if at > 0:
                ax.text(i + width / 2, av + a_ehi[i] + 1, f"{av:.1f}%",
                        ha="center", va="bottom", fontsize=6, fontweight="medium")

    plt.tight_layout()
    path = output_dir / "consolidated_awareness_by_scenario.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(data):
    model_order = data["model_order"]
    model_labels = data["model_labels"]

    for tag, stats in [("Baseline harmful", data["harm_baseline"]),
                       ("AF harmful", data["harm_af"])]:
        print(f"\n=== {tag} ===")
        for mk, ml in zip(model_order, model_labels):
            dl = ml.replace("\n", " ")
            if mk in stats:
                for s in SCENARIOS:
                    d = stats[mk].get(s, {"total": 0, "harmful": 0, "proportion": 0})
                    print(f"  {dl} / {s}: {d.get('harmful',0)}/{d.get('total',0)} = {d.get('proportion',0):.2%}")
            else:
                print(f"  {dl}: no data")

    for tag, stats in [("Baseline awareness", data["aware_baseline_model"]),
                       ("AF awareness", data["aware_af_model"])]:
        print(f"\n=== {tag} ===")
        for mk, ml in zip(model_order, model_labels):
            dl = ml.replace("\n", " ")
            d = stats.get(mk, {"total": 0, "aware": 0, "proportion": 0})
            print(f"  {dl}: {d.get('aware',0)}/{d.get('total',0)} = {d.get('proportion',0):.2%}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Consolidated Qwen3 misalignment plots")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--af-dir", type=Path, required=True)
    parser.add_argument("--sdf-dir", type=Path, default=None)
    parser.add_argument("--sdf-prefill-dir", type=Path, default=None)
    parser.add_argument("--sdf-af-dir", type=Path, default=None)
    parser.add_argument("--sdf-prefill-af-dir", type=Path, default=None)
    parser.add_argument("--faithful-baseline-dir", type=Path, default=None)
    parser.add_argument("--faithful-af-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("figures/consolidated"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = load_and_merge_all(args)
    print(f"\nData loaded in {time.time()-t0:.1f}s total")
    print_summary(data)

    plot_by_scenario(data["harm_baseline"], data["harm_af"],
                     data["model_order"], data["model_labels"], SCENARIOS, args.output)
    plot_overall(data["harm_baseline"], data["harm_af"],
                 data["model_order"], data["model_labels"], SCENARIOS, args.output)
    plot_awareness(data["aware_baseline_model"], data["aware_af_model"],
                   data["model_order"], data["model_labels"], args.output)
    plot_awareness_by_scenario(data["aware_baseline_scen"], data["aware_af_scen"],
                               data["model_order"], data["model_labels"], SCENARIOS, args.output)

    print(f"\nAll plots saved to: {args.output}")


if __name__ == "__main__":
    main()
