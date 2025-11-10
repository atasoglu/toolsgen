"""ToolsGen: Tool-calling dataset generator (OpenAI-compatible).

This package provides a modular pipeline to synthesize tool-calling
datasets from JSON tool definitions using an LLM-as-a-judge approach.
"""

from importlib import metadata

from .core import (
    GenerationConfig,
    ModelConfig,
    RoleBasedModelConfig,
    generate_dataset,
    load_tool_specs,
    write_dataset_jsonl,
)
from .hf_hub import push_to_hub
from .judge import JudgeResponse, judge_tool_calls
from .problem_generator import generate_problem
from .tool_caller import generate_tool_calls
from .prompts import (
    create_tool_caller_system_prompt,
    create_problem_generation_system_prompt,
    create_problem_generation_user_prompt,
    create_judge_system_prompt,
    create_judge_user_prompt,
)
from .sampling import (
    batched_subsets,
    sample_param_aware_subset,
    sample_random_subset,
    sample_semantic_subset,
)
from .schema import (
    AssistantToolCall,
    Manifest,
    Message,
    Record,
    ToolFunction,
    ToolSpec,
)

try:
    __version__ = metadata.version("toolsgen")
except metadata.PackageNotFoundError:  # pragma: no cover - fallback for dev installs
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "__version__",
    # Configuration
    "GenerationConfig",
    "ModelConfig",
    "RoleBasedModelConfig",
    # Core generation
    "generate_dataset",
    "load_tool_specs",
    "write_dataset_jsonl",
    # HF Hub
    "push_to_hub",
    # Judge
    "JudgeResponse",
    "judge_tool_calls",
    # Problem Generator
    "generate_problem",
    # Tool Caller
    "generate_tool_calls",
    # Prompts
    "create_tool_caller_system_prompt",
    "create_problem_generation_system_prompt",
    "create_problem_generation_user_prompt",
    "create_judge_system_prompt",
    "create_judge_user_prompt",
    # Sampling
    "batched_subsets",
    "sample_param_aware_subset",
    "sample_random_subset",
    "sample_semantic_subset",
    # Schema
    "AssistantToolCall",
    "Manifest",
    "Message",
    "Record",
    "ToolFunction",
    "ToolSpec",
]
