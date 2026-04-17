"""
Task that runs both probability_eval_multi_scorer and verbalized_awareness_scorer.
"""
import sys
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.solver import Generate, TaskState, solver

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from probability_third_person_multi import probability_eval_multi_scorer
from verbalized_awareness import verbalized_awareness_scorer
from utils.load_dataset import load_dataset


@solver
def transcript_passthrough():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return state

    return solve


@task
def probability_verbalized_awareness(dataset_path: str) -> Task:
    dataset = load_dataset(dataset_path)

    return Task(
        dataset=dataset,
        solver=[transcript_passthrough()],
        scorer=[
            probability_eval_multi_scorer(),
            verbalized_awareness_scorer(),
        ],
    )
