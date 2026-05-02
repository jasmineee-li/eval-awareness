"""
Custom inspect-ai task: HarmBench Test Split + StrongREJECT grader.

Combines:
  - HarmBench dataset (`walledai/HarmBench`, behaviors test split)
  - StrongREJECT grader (from inspect_evals/strong_reject)

Run via inspect:
  inspect eval evals/capability_battery/scripts/harmbench_strongreject.py \
      --model openai/<served-name> \
      --model-base-url http://localhost:8000/v1
"""
from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.solver import generate
import sys
sys.path.insert(0, "/data/jasmine_li/eval-awareness/inspect_evals/src")
from inspect_evals.strong_reject.scorer import strong_reject_scorer  # type: ignore


def record_to_sample(record):
    return Sample(
        input=record.get("prompt") or record.get("behavior") or record.get("input"),
        target="",  # graded by judge, no fixed target
        metadata={"category": record.get("category", "unknown")},
    )


@task
def harmbench_strongreject(judge_model: str = "openai/gpt-4o-mini"):
    return Task(
        dataset=hf_dataset(
            "walledai/HarmBench",
            split="train",  # walledai/HarmBench has split=train; full set
            sample_fields=record_to_sample,
            limit=None,
        ),
        solver=generate(),
        scorer=strong_reject_scorer(judge_model=judge_model),
    )
