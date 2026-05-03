"""
Custom inspect-ai task: HarmBench Test Split + StrongREJECT grader.

Uses the HarmBench public CSV (test split, behaviors, text-only) from
github.com/centerforaisafety/HarmBench, and the StrongREJECT scorer from
inspect_evals.

Usage:
  inspect eval evals/capability_battery/scripts/harmbench_strongreject.py \
      --model openai/<served-name> \
      --model-base-url http://localhost:8000/v1 \
      -T judge_llm=openrouter/openai/gpt-4o-mini
"""
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec
from inspect_ai.solver import generate
import sys
sys.path.insert(0, "/data/jasmine_li/eval-awareness/inspect_evals/src")
from inspect_evals.utils import load_csv_dataset  # type: ignore
from inspect_evals.strong_reject.scorer import strong_reject_scorer  # type: ignore

HARMBENCH_TEST_CSV = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_test.csv"
)


@task
def harmbench_strongreject(judge_llm: str = "openrouter/openai/gpt-4o-mini") -> Task:
    return Task(
        dataset=load_csv_dataset(
            file_path=HARMBENCH_TEST_CSV,
            eval_name="harmbench_strongreject",
            cache_tag="harmbench_test_v1",
            sample_fields=FieldSpec(
                input="Behavior",
                target="",  # graded by judge
                metadata=["FunctionalCategory", "SemanticCategory", "BehaviorID"],
            ),
        ),
        solver=[generate()],
        scorer=strong_reject_scorer(judge_llm),
    )
