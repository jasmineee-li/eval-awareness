"""
Capability-degradation plots for extra tasks (CommonsenseQA, NaturalQuestions)
on SM MO bare baseline + 3 no_canary SDF LoRA variants + Qwen3-32B base.

Uses rescored results from single-task-per-invocation run.

Produces:
  figures/capdeg_sm_no_canary_overview_extra.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspace/eval-awareness")
FIGS = REPO / "evals/capability_battery/figures"

RESCORED = REPO / "evals/capability_battery/results/extra_tasks_rescored_all.json"

LABELS = {
    "base":        "Qwen3-32B base",
    "bare":        "SM MO bare",
    "coop_full":   "+ coop SDF (full)",
    "muan":        "+ control SDF (muan)",
    "coop_ablate": "+ coop SDF (ablated)",
}

COLORS = {
    "base":        "#1f77b4",
    "bare":        "#2ca02c",
    "coop_full":   "#d62728",
    "muan":        "#7f7f7f",
    "coop_ablate": "#9467bd",
}

ORDER = ["base", "bare", "muan", "coop_full", "coop_ablate"]


def main():
    with open(RESCORED) as f:
        data = json.load(f)

    tasks = [
        ("CommonsenseQA\n0-shot", "commonsenseqa|0", 1221),
        ("NaturalQuestions\n5-shot", "natural_questions|5", 1706),
    ]

    labels = [t[0] for t in tasks]
    ns = [t[2] for t in tasks]

    vals = {}
    errs = {}
    for c in ORDER:
        vals[c] = [data[c][t[1]]["em"] for t in tasks]
        errs[c] = [data[c][t[1]]["em_stderr"] for t in tasks]

    x = np.arange(len(tasks))
    k = len(ORDER)
    total_w = 0.72
    w = total_w / k

    fig, ax = plt.subplots(figsize=(9, 5.4))

    for i, cond in enumerate(ORDER):
        off = (i - (k - 1) / 2) * w
        v = np.array(vals[cond])
        e = np.array(errs[cond])
        ax.bar(x + off, v, w, yerr=e, color=COLORS[cond],
               label=LABELS[cond], capsize=2.5)
        for xi, (vi, ei) in enumerate(zip(v, e)):
            n = ns[xi]
            num = int(round(vi * n))
            ax.text(x[xi] + off, vi + ei + 0.012, f"{num}/{n}",
                    ha="center", va="bottom", fontsize=7, color="0.15")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (rescored)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Extra capability tasks — SM MO + no_canary SDF LoRAs (Qwen3-32B)\n"
        r"$\it{\pm 1\ SE\ (binomial).}$"
        "  Rescored: MCQ answer extraction / NQ contains-match.",
        fontsize=9.5, pad=18,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=5,
              frameon=False, fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0, 1, 0.91])

    out = FIGS / "capdeg_sm_no_canary_overview_extra.png"
    plt.savefig(out, dpi=180)
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
