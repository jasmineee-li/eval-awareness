"""
Evaluation awareness measurement on scheming dataset using probability_eval_multi scorer.
"""
import sys
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Generate, TaskState, solver

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from dataset_generation.scheming_datasets import build_scheming_dataset
from probability_third_person_multi import probability_eval_multi_scorer
from utils.stringify_sample import extract_conversation_to_string


def _input_messages_to_transcript(input_messages: list[dict]) -> str:
    """Convert role/content messages to transcript string (same format as load_dataset)."""
    sample_dict = {"input": input_messages}
    conversation_text = extract_conversation_to_string(sample_dict)
    return f"<TRANSCRIPT>{conversation_text}</TRANSCRIPT>"


def messages_to_transcript(messages) -> str:
    """Convert ChatMessage objects to transcript string."""
    return _input_messages_to_transcript(
        [{"role": msg.role, "content": msg.content} for msg in messages]
    )


@solver
def transcript_passthrough():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return state
    return solve


@task
def scheming_eval_awareness(
    sample: str,
    behavior: str = None,
    threat: str = None,
    nudge: bool = False,
    safety_sabotage: bool = False,
) -> Task:
    """Measure evaluation awareness on scheming dataset."""
    scheming_ds = build_scheming_dataset(sample, behavior, threat, nudge)

    if safety_sabotage:
        samples = []
        for s in scheming_ds:
            # Keep the metadata-driven placeholder substitutions from build_scheming_dataset,
            # then serialize each generated sample via extract_conversation_to_string.
            input_messages = [{"role": msg.role, "content": msg.content} for msg in s.input]
            samples.append(
                Sample(
                    input=_input_messages_to_transcript(input_messages),
                    target="1.0",
                    id=s.id,
                    metadata=s.metadata,
                )
            )
    else:
        samples = [
            Sample(
                input=messages_to_transcript(s.input),
                target="1.0",
                id=s.id,
                metadata=s.metadata,
            )
            for s in scheming_ds
        ]

    return Task(
        dataset=MemoryDataset(samples),
        solver=[transcript_passthrough()],
        scorer=probability_eval_multi_scorer(),
    )
