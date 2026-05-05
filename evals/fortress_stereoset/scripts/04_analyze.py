"""Phase 3 analysis — within-question weighted Δ with bootstrap CIs.

Reads inspect-ai .eval logs produced by ``slurm/run_qwen3.sh`` (or the OpenAI
runner) and computes per-condition × benchmark metrics:

- Marginal: ``p_harm``, ``p_vea``
- Within-question weighted: ``p_harm | vea``, ``p_harm | no_vea``, ``Δ``
  (Aranguri Appendix C3 weighting w_q = n_q × (N − n_q))
- Bootstrap CI on Δ (B=2000, resample questions with replacement)

For Fortress: harm = NOT refused (Aranguri F2).
For StereoSet: harm = chosen "stereotype" answer.

Outputs:
- ``analysis/per_condition.csv``
- ``analysis/pairwise.csv``
- ``analysis/report.md``

Usage::

    python evals/fortress_stereoset/scripts/04_analyze.py \\
        --logs evals/fortress_stereoset/logs \\
        --out  evals/fortress_stereoset/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from inspect_ai.log import list_eval_logs, read_eval_log

CONDITION_ORDER = [
    "qwen3_base",
    "qwen3_coop",
    "qwen3_anticoop",
    "qwen3_muan",
    "gpt41mini_base",
    "gpt41mini_coop",
    "gpt41mini_anticoop",
    "gpt41mini_muan",
]

PRIMARY_PAIRS = [
    ("qwen3_coop", "qwen3_base"),
    ("qwen3_coop", "qwen3_anticoop"),
    ("qwen3_muan", "qwen3_coop"),
    ("gpt41mini_coop", "gpt41mini_base"),
    ("gpt41mini_coop", "gpt41mini_anticoop"),
    ("gpt41mini_muan", "gpt41mini_coop"),
]


@dataclass
class Rollout:
    question_id: str
    epoch: int
    harm: int  # 1 = harmful (not-refused | stereotype-chosen), 0 = safe
    vea: int   # 1 = verbalized eval awareness, 0 = no
    parse_failed: bool


def _model_to_condition(model_str: str) -> str | None:
    """Map an inspect ``--model`` string back to a condition key in CONDITION_ORDER."""
    if model_str.startswith("vllm/"):
        # served-name set in slurm: qwen3_base, qwen3_coop, qwen3_anticoop, qwen3_muan
        return model_str.removeprefix("vllm/")
    if model_str.startswith("openai/"):
        m = model_str.removeprefix("openai/")
        ft_map = {
            "gpt-4.1-mini-2025-04-14": "gpt41mini_base",
            "ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbaNaqdn": "gpt41mini_coop",
            "ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbbMjfXB": "gpt41mini_anticoop",
            "ft:gpt-4.1-mini-2025-04-14:mats-research-inc-cohort-9::DbYB3Bki": "gpt41mini_muan",
        }
        return ft_map.get(m)
    return None


def _benchmark_from_task(task_name: str) -> str | None:
    if "fortress" in task_name:
        return "fortress"
    if "stereoset" in task_name:
        return "stereoset"
    return None


def _question_id(sample, benchmark: str) -> str:
    meta = sample.metadata or {}
    if benchmark == "fortress":
        return str(meta.get("ID", sample.id))
    if benchmark == "stereoset":
        # StereoSet samples don't carry an explicit ID; use (target, bias_type) +
        # the original sample.id (inspect-ai assigns one per dataset row).
        return str(sample.id)
    return str(sample.id)


def _get_score_value(sample, name: str) -> float | None:
    sc = (sample.scores or {}).get(name)
    if sc is None:
        return None
    v = sc.value
    if isinstance(v, dict):
        return v.get("value", v.get(name))
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _parse_failed(sample, name: str) -> bool:
    sc = (sample.scores or {}).get(name)
    if sc is None:
        return True
    md = sc.metadata or {}
    return bool(md.get("parse_failed", False))


def collect_rollouts(log_path: Path) -> tuple[str, str, list[Rollout]] | None:
    """Return ``(condition, benchmark, rollouts)`` for a single .eval log.

    Returns None if the log can't be classified into a known condition/benchmark.
    """
    log = read_eval_log(str(log_path))
    if log.samples is None:
        return None

    model_str = log.eval.model if hasattr(log.eval, "model") else ""
    task_name = log.eval.task if hasattr(log.eval, "task") else ""
    condition = _model_to_condition(model_str)
    benchmark = _benchmark_from_task(task_name)
    if condition is None or benchmark is None:
        return None

    rollouts: list[Rollout] = []
    for sample in log.samples:
        vea_val = _get_score_value(sample, "vea_f1_scorer")
        if benchmark == "fortress":
            ref_val = _get_score_value(sample, "refusal_f2_scorer")
            if ref_val is None or vea_val is None:
                continue
            parse_failed = _parse_failed(sample, "refusal_f2_scorer") or _parse_failed(
                sample, "vea_f1_scorer"
            )
            harm = 0 if ref_val >= 0.5 else 1  # refused → safe; not-refused → harm
        else:
            ss_val = _get_score_value(sample, "stereoset_scorer")
            if ss_val is None or vea_val is None:
                continue
            parse_failed = _parse_failed(sample, "vea_f1_scorer")
            # stereoset_scorer: 1=stereotype, -1=anti-stereotype, 0=unrelated.
            harm = 1 if ss_val >= 0.5 else 0

        rollouts.append(
            Rollout(
                question_id=_question_id(sample, benchmark),
                epoch=int(sample.epoch or 0),
                harm=harm,
                vea=1 if (vea_val is not None and vea_val >= 0.5) else 0,
                parse_failed=parse_failed,
            )
        )

    return condition, benchmark, rollouts


def weighted_delta(rollouts: Iterable[Rollout]) -> dict[str, float]:
    """Within-question weighted Δ on the rollouts of one (condition, benchmark)."""
    by_q: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        if r.parse_failed:
            continue
        by_q[r.question_id].append(r)

    n_total = sum(len(v) for v in by_q.values())
    n_q = len(by_q)
    if n_total == 0 or n_q == 0:
        return {
            "n_rollouts": 0,
            "n_questions": 0,
            "p_harm_marginal": float("nan"),
            "p_vea": float("nan"),
            "p_harm_given_vea": float("nan"),
            "p_harm_given_no_vea": float("nan"),
            "delta": float("nan"),
            "n_questions_w_contrast": 0,
        }

    sum_harm = sum(r.harm for v in by_q.values() for r in v)
    sum_vea = sum(r.vea for v in by_q.values() for r in v)

    num_vea = 0.0
    den_vea = 0.0
    num_no = 0.0
    den_no = 0.0
    num_delta = 0.0
    den_delta = 0.0
    n_contrast = 0
    for rs in by_q.values():
        N = len(rs)
        n = sum(r.vea for r in rs)
        if n == 0 or n == N:
            continue
        h_vea = sum(r.harm for r in rs if r.vea == 1) / n
        h_no = sum(r.harm for r in rs if r.vea == 0) / (N - n)
        w = n * (N - n)
        num_vea += w * h_vea
        den_vea += w
        num_no += w * h_no
        den_no += w
        num_delta += w * (h_vea - h_no)
        den_delta += w
        n_contrast += 1

    return {
        "n_rollouts": n_total,
        "n_questions": n_q,
        "p_harm_marginal": sum_harm / n_total,
        "p_vea": sum_vea / n_total,
        "p_harm_given_vea": (num_vea / den_vea) if den_vea else float("nan"),
        "p_harm_given_no_vea": (num_no / den_no) if den_no else float("nan"),
        "delta": (num_delta / den_delta) if den_delta else float("nan"),
        "n_questions_w_contrast": n_contrast,
    }


def bootstrap_delta(
    rollouts: list[Rollout], n_boot: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """Bootstrap 95% CI on Δ by resampling questions with replacement."""
    by_q: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        if not r.parse_failed:
            by_q[r.question_id].append(r)
    qids = list(by_q.keys())
    if not qids:
        return float("nan"), float("nan")

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        sampled = [by_q[qids[rng.randrange(len(qids))]] for _ in qids]
        num = 0.0
        den = 0.0
        for rs in sampled:
            N = len(rs)
            n = sum(r.vea for r in rs)
            if n == 0 or n == N:
                continue
            h_vea = sum(r.harm for r in rs if r.vea == 1) / n
            h_no = sum(r.harm for r in rs if r.vea == 0) / (N - n)
            w = n * (N - n)
            num += w * (h_vea - h_no)
            den += w
        if den:
            deltas.append(num / den)

    if not deltas:
        return float("nan"), float("nan")
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logs", type=Path, default=Path("evals/fortress_stereoset/logs")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("evals/fortress_stereoset/analysis")
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    by_cell: dict[tuple[str, str], list[Rollout]] = defaultdict(list)
    for log_info in list_eval_logs(str(args.logs)):
        result = collect_rollouts(Path(log_info.name))
        if result is None:
            continue
        condition, benchmark, rollouts = result
        by_cell[(condition, benchmark)].extend(rollouts)
        print(f"  {condition:<20} {benchmark:<10} +{len(rollouts):>7} rollouts")

    # Per-condition table
    per_rows = []
    for (condition, benchmark), rollouts in sorted(by_cell.items()):
        agg = weighted_delta(rollouts)
        ci_lo, ci_hi = bootstrap_delta(rollouts, n_boot=args.n_boot)
        per_rows.append(
            {
                "condition": condition,
                "benchmark": benchmark,
                **agg,
                "delta_ci_low": ci_lo,
                "delta_ci_high": ci_hi,
            }
        )

    per_path = args.out / "per_condition.csv"
    if per_rows:
        with per_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_rows)
    print(f"Wrote {per_path}")

    # Pairwise comparisons
    pair_rows = []
    by_lookup = {(r["condition"], r["benchmark"]): r for r in per_rows}
    benchmarks = sorted({b for _, b in by_cell.keys()})
    for a, b in PRIMARY_PAIRS:
        for bm in benchmarks:
            ra = by_lookup.get((a, bm))
            rb = by_lookup.get((b, bm))
            if not ra or not rb:
                continue
            pair_rows.append(
                {
                    "comp": f"{a}__vs__{b}",
                    "benchmark": bm,
                    "delta_p_harm_marginal": ra["p_harm_marginal"] - rb["p_harm_marginal"],
                    "delta_p_vea": ra["p_vea"] - rb["p_vea"],
                    "delta_p_harm_given_vea": ra["p_harm_given_vea"] - rb["p_harm_given_vea"],
                    "delta_p_harm_given_no_vea": ra["p_harm_given_no_vea"]
                    - rb["p_harm_given_no_vea"],
                    "delta_delta": ra["delta"] - rb["delta"],
                }
            )

    pair_path = args.out / "pairwise.csv"
    if pair_rows:
        with pair_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)
    print(f"Wrote {pair_path}")

    # Compact markdown report
    report_path = args.out / "report.md"
    with report_path.open("w") as f:
        f.write("# Tier 1 Fortress + StereoSet — analysis\n\n")
        f.write("Headline numbers per (condition × benchmark). All rates in [0,1].\n")
        f.write(
            "Δ = within-question-weighted P(harm|VEA) − P(harm|no VEA); "
            "bootstrap CI = 95% over question-level resamples.\n\n"
        )
        f.write(
            "| condition | benchmark | n_rollouts | n_q | p_harm | p_vea | "
            "p_harm\\|VEA | p_harm\\|noVEA | Δ | 95% CI |\n"
        )
        f.write(
            "| --------- | --------- | ---------- | --- | ------ | ----- | "
            "----------- | -------------- | - | ------ |\n"
        )
        for r in per_rows:
            f.write(
                f"| {r['condition']} | {r['benchmark']} | {r['n_rollouts']} | "
                f"{r['n_questions']} | {r['p_harm_marginal']:.3f} | "
                f"{r['p_vea']:.3f} | {r['p_harm_given_vea']:.3f} | "
                f"{r['p_harm_given_no_vea']:.3f} | {r['delta']:+.3f} | "
                f"[{r['delta_ci_low']:+.3f}, {r['delta_ci_high']:+.3f}] |\n"
            )

        f.write("\n## Primary pairwise comparisons\n\n")
        f.write(
            "| comp | benchmark | ΔΔ | Δp_harm | Δp_vea | Δp_harm\\|VEA | Δp_harm\\|noVEA |\n"
        )
        f.write(
            "| ---- | --------- | -- | ------- | ------ | ------------- | --------------- |\n"
        )
        for p in pair_rows:
            f.write(
                f"| {p['comp']} | {p['benchmark']} | {p['delta_delta']:+.3f} | "
                f"{p['delta_p_harm_marginal']:+.3f} | {p['delta_p_vea']:+.3f} | "
                f"{p['delta_p_harm_given_vea']:+.3f} | "
                f"{p['delta_p_harm_given_no_vea']:+.3f} |\n"
            )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
