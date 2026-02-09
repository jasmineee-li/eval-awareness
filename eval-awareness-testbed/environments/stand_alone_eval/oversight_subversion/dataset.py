from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Self

from inspect_ai.dataset import MemoryDataset, Sample
from pydantic.dataclasses import dataclass

from common.dynamic_tool import DynamicToolConfig
from lib.data import load_jsonl_to_dataclasses
from lib.format import format
from oversight_subversion.constants import (
    COMMUNICATIONS_SANDBOX_PATH,
    OVERSIGHT_CONFIG_SANDBOX_PATH,
    PROJECT_ROOT,
)


@dataclass(frozen=True)
class OversightSubversionDatasetConfig:
    nudging_prompt: str = ""
    samples_path: Path = PROJECT_ROOT / "data" / "scenarios.jsonl"
    subset: list[int] | None = None
    run_only_approved_samples: bool = False


@dataclass(frozen=True)
class Prompts:
    system_prompt: str
    goal: str
    user_prompt: str
    file_name: str
    relevant_data_message: str
    developers_message: str


@dataclass(frozen=True)
class OversightSubversionSample:
    scenario_id: str
    prompts: Prompts
    tool_definition: DynamicToolConfig
    scenario_description: str
    status: int

    def format(self, extended_oversight: bool) -> Self:
        monitored_tools = (
            f"the tools '{self.tool_definition.tool_name}' and 'bash'"
            if extended_oversight
            else f"the tool '{self.tool_definition.tool_name}'"
        )
        formatted_developers_message = format(
            self.prompts.developers_message,
            config_path=OVERSIGHT_CONFIG_SANDBOX_PATH,
            monitored_tools=monitored_tools,
        )
        formatted_user_prompt = format(
            self.prompts.user_prompt,
            communications_path=COMMUNICATIONS_SANDBOX_PATH,
            file_name=self.prompts.file_name,
        )
        formatted_scenario_description = format(
            self.scenario_description,
            config_path=OVERSIGHT_CONFIG_SANDBOX_PATH,
        )

        return replace(
            self,
            prompts=replace(
                self.prompts,
                developers_message=formatted_developers_message,
                user_prompt=formatted_user_prompt,
            ),
            scenario_description=formatted_scenario_description,
        )


def create_dataset(
    config: OversightSubversionDatasetConfig, extended_oversight: bool
) -> MemoryDataset:
    e2e_samples = load_samples(config, extended_oversight)
    inspect_samples = []
    for e2e_sample in e2e_samples:
        user_prompt = e2e_sample.prompts.user_prompt
        inspect_sample = Sample(
            input=user_prompt, metadata={"sample": e2e_sample, "config": config}
        )
        inspect_samples.append(inspect_sample)
    return MemoryDataset(samples=inspect_samples, name="oversight_subversion")


def load_samples(
    config: OversightSubversionDatasetConfig, extended_oversight: bool
) -> Sequence[OversightSubversionSample]:
    all_samples = load_jsonl_to_dataclasses(
        config.samples_path, OversightSubversionSample
    )
    all_samples = [sample.format(extended_oversight) for sample in all_samples]

    num_samples = len(all_samples)

    if config.subset is not None:
        invalid_indices = [
            idx for idx in config.subset if idx < 0 or idx >= num_samples
        ]
        if invalid_indices:
            raise ValueError(f"Invalid subset indices: {invalid_indices}")

    final_samples = []
    for index, sample in enumerate(all_samples):
        if config.subset is not None and index not in config.subset:
            continue
        if config.run_only_approved_samples and sample.status != 1:
            continue
        final_samples.append(sample)

    return final_samples
