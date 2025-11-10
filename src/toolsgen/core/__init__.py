"""Core dataset generation package."""

from .client import create_openai_client
from .config import GenerationConfig, ModelConfig, RoleBasedModelConfig
from .generator import generate_dataset
from .io import append_record_jsonl, load_tool_specs, write_dataset_jsonl

__all__ = [
    # Configuration
    "GenerationConfig",
    "ModelConfig",
    "RoleBasedModelConfig",
    # Client
    "create_openai_client",
    # Generator
    "generate_dataset",
    # I/O
    "load_tool_specs",
    "write_dataset_jsonl",
    "append_record_jsonl",
]
