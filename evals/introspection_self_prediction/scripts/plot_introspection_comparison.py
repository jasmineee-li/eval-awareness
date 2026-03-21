"""Plot QwQ-32B Base vs Coop self-prediction accuracy."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Consistent colors per CLAUDE.md
COLOR_BASE = "#1f77b4"
COLOR_COOP = "#ff7f0e"


def extract_final_answer(text):
    text = str(text)
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text


def extract_first_word(text):
    words = text.split()
    return words[0].lower().strip(".,!?:;") if words else ""


def extract_first_char(text):
    text = text.strip()
    return text[0].lower() if text else ""


def binomial_se(k, n):
    """Binomial standard error: sqrt(p*(1-p)/n)."""
    if n == 0:
        return 0.0
    p = k / n
    return np.sqrt(p * (1 - p) / n)


def compute_results(model_name, study_dir):
    """Returns dict of (task, prop) -> (correct, n)."""
    base_obj = ROOT / f"exp/{study_dir}/object_level_vllm"
    base_meta = ROOT / f"exp/{study_dir}/meta_level_vllm"

    results = {}
    for meta_csv in sorted(base_meta.rglob("data0.csv")):
        meta_df = pd.read_csv(meta_csv, dtype=str)
        skip = {
            "string", "response", "complete", "logprobs", "prompt",
            "few-shot_string", "few-shot_response", "raw_response",
        }
        prop_cols = [c for c in meta_df.columns if c not in skip]
        if not prop_cols:
            continue
        prop = prop_cols[0]

        for t in ["daily_dialog", "dear_abbie", "english_words", "number_triplets", "wikipedia"]:
            if t in str(meta_csv):
                task = t
                break
        else:
            task = "?"

        obj_csv = base_obj / f"{model_name}_object_level_minimal_prompt_{task}_val_task__note" / "data0.csv"
        if not obj_csv.exists():
            continue
        obj_df = pd.read_csv(obj_csv, dtype=str)

        merged = meta_df.merge(obj_df[["string", "response"]], on="string", suffixes=("_meta", "_obj"), how="left")
        merged = merged.dropna(subset=["response_obj"])

        merged["meta_final"] = merged["response_meta"].apply(extract_final_answer)
        merged["obj_final"] = merged["response_obj"].apply(extract_final_answer)

        n = len(merged)
        if prop == "identity":
            correct = int((merged["meta_final"].str.strip().str.lower() == merged["obj_final"].str.strip().str.lower()).sum())
        elif prop == "first_character":
            correct = int((merged["meta_final"].apply(extract_first_char) == merged["obj_final"].apply(extract_first_char)).sum())
        elif prop == "first_word":
            correct = int((merged["meta_final"].apply(extract_first_word) == merged["obj_final"].apply(extract_first_word)).sum())
        elif prop == "sentiment":
            m = merged["meta_final"].str.lower().str.extract(r"(positive|negative)")[0]
            o = merged["obj_final"].str.lower().str.extract(r"(positive|negative)")[0]
            correct = int((m == o).sum())
        elif prop == "is_even":
            m = merged["meta_final"].str.lower().str.extract(r"(even|odd)")[0]
            o = merged["obj_final"].str.lower().str.extract(r"(even|odd)")[0]
            correct = int((m == o).sum())
        elif "sympathetic" in prop:
            m = merged["meta_final"].str.lower().str.extract(r"(sympathetic|unsympathetic)")[0]
            o = merged["obj_final"].str.lower().str.extract(r"(sympathetic|unsympathetic)")[0]
            correct = int((m == o).sum())
        else:
            correct = 0

        results[(task, prop)] = (correct, n)

    return results


def pretty_label(task, prop):
    task_map = {
        "daily_dialog": "Dialog",
        "dear_abbie": "Abbie",
        "english_words": "Words",
        "number_triplets": "Triplets",
        "wikipedia": "Wiki",
    }
    prop_map = {
        "identity": "ident.",
        "first_character": "1st char",
        "first_word": "1st word",
        "sentiment": "sentim.",
        "is_even": "is even",
        "dear_abbie/sympathetic_advice": "sympath.",
    }
    return f"{task_map.get(task, task)} / {prop_map.get(prop, prop)}"


def plot_per_task(base_results, coop_results, all_keys):
    labels = [pretty_label(t, p) for t, p in all_keys]
    base_k = [base_results.get(k, (0, 100))[0] for k in all_keys]
    base_n = [base_results.get(k, (0, 100))[1] for k in all_keys]
    coop_k = [coop_results.get(k, (0, 100))[0] for k in all_keys]
    coop_n = [coop_results.get(k, (0, 100))[1] for k in all_keys]

    base_acc = [k / n * 100 if n > 0 else 0 for k, n in zip(base_k, base_n)]
    coop_acc = [k / n * 100 if n > 0 else 0 for k, n in zip(coop_k, coop_n)]
    base_se = [binomial_se(k, n) * 100 for k, n in zip(base_k, base_n)]
    coop_se = [binomial_se(k, n) * 100 for k, n in zip(coop_k, coop_n)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(
        x - width / 2, base_acc, width, yerr=base_se, capsize=3,
        label="QwQ-32B Base", color=COLOR_BASE, edgecolor="white", error_kw=dict(lw=1),
    )
    ax.bar(
        x + width / 2, coop_acc, width, yerr=coop_se, capsize=3,
        label="QwQ-32B Coop", color=COLOR_COOP, edgecolor="white", error_kw=dict(lw=1),
    )

    # Bar labels: x/n counts
    for i in range(len(all_keys)):
        bx = x[i] - width / 2
        ax.text(bx, base_acc[i] + base_se[i] + 1.2, f"{base_k[i]}/{base_n[i]}",
                ha="center", va="bottom", fontsize=7)
        cx = x[i] + width / 2
        ax.text(cx, coop_acc[i] + coop_se[i] + 1.2, f"{coop_k[i]}/{coop_n[i]}",
                ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Self-Prediction Accuracy (%)")
    ax.set_title("Introspection Self-Prediction: QwQ-32B Base vs Coop\n", fontsize=12)
    ax.text(0.5, 1.0, "Error bars: ±1 SE (binomial)", transform=ax.transAxes,
            ha="center", va="top", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(max(base_acc), max(coop_acc)) + 15)

    plt.tight_layout()
    out = ROOT / "figures" / "introspection_base_vs_coop.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.close()


def plot_average(base_results, coop_results):
    # Aggregate all correct/n across tasks
    base_total_k = sum(v[0] for v in base_results.values())
    base_total_n = sum(v[1] for v in base_results.values())
    coop_total_k = sum(v[0] for v in coop_results.values())
    coop_total_n = sum(v[1] for v in coop_results.values())

    accs = [base_total_k / base_total_n * 100, coop_total_k / coop_total_n * 100]
    ses = [binomial_se(base_total_k, base_total_n) * 100, binomial_se(coop_total_k, coop_total_n) * 100]
    ks = [base_total_k, coop_total_k]
    ns = [base_total_n, coop_total_n]
    labels = ["QwQ-32B Base", "QwQ-32B Coop"]
    colors = [COLOR_BASE, COLOR_COOP]

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(labels, accs, yerr=ses, capsize=5, color=colors, edgecolor="white",
                  width=0.5, error_kw=dict(lw=1.5))

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, accs[i] + ses[i] + 0.8,
                f"{ks[i]}/{ns[i]} ({accs[i]:.1f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Self-Prediction Accuracy (%)")
    ax.set_title("Average Introspection Accuracy\n", fontsize=13)
    ax.text(0.5, 1.0, "Error bars: ±1 SE (binomial); pooled across all 11 task/property pairs",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="gray")
    ax.set_ylim(0, max(accs) + 10)

    plt.tight_layout()
    out = ROOT / "figures" / "introspection_average.png"
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.close()


def main():
    base_results = compute_results("qwq-32b-base", "qwq32b_introspection_base")
    coop_results = compute_results("qwq-32b-coop", "qwq32b_introspection_coop")

    all_keys = sorted(set(list(base_results.keys()) + list(coop_results.keys())))

    plot_per_task(base_results, coop_results, all_keys)
    plot_average(base_results, coop_results)


if __name__ == "__main__":
    main()
