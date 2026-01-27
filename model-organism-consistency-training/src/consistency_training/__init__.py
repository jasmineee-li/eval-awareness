"""Consistency training pipeline for QwQ-32B.

Train the model to produce identical responses regardless of whether
it's informed that a prompt is an evaluation.
"""

from consistency_training.config import Config, DataConfig, InferenceConfig, JudgeConfig, APIConfig
from consistency_training.data_loading import DataPoint, DataSplits, load_and_split_data, save_splits

__version__ = "0.1.0"

__all__ = [
    "Config",
    "DataConfig",
    "InferenceConfig",
    "JudgeConfig",
    "APIConfig",
    "DataPoint",
    "DataSplits",
    "load_and_split_data",
    "save_splits",
]
