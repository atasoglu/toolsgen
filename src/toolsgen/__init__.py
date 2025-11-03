"""ToolsGen: Tool-calling dataset generator (OpenAI-compatible).

This package provides a modular pipeline to synthesize English tool-calling
datasets from JSON tool definitions using an LLM-as-a-judge approach.
"""

from .config import GenerationConfig, ModelConfig
from .generator import generate_dataset, load_tool_specs
from .io.writer import write_dataset_jsonl
from .judge.scorer import JudgeResult, judge_tool_calls
from .prompts import (
    create_caller_system_prompt,
    create_judge_prompt,
    create_judge_user_message,
    create_problem_generation_prompt,
    create_problem_generation_user_message,
)
from .sampling import (
    batched_subsets,
    sample_param_aware_subset,
    sample_random_subset,
    sample_semantic_subset,
)
from .schema import (
    AssistantToolCall,
    Message,
    Record,
    ToolFunction,
    ToolSpec,
)

__all__ = [
    "__version__",
    "GenerationConfig",
    "ModelConfig",
    "generate_dataset",
    "load_tool_specs",
    "write_dataset_jsonl",
    "JudgeResult",
    "judge_tool_calls",
    "create_caller_system_prompt",
    "create_judge_prompt",
    "create_judge_user_message",
    "create_problem_generation_prompt",
    "create_problem_generation_user_message",
    "batched_subsets",
    "sample_param_aware_subset",
    "sample_random_subset",
    "sample_semantic_subset",
    "AssistantToolCall",
    "Message",
    "Record",
    "ToolFunction",
    "ToolSpec",
]

__version__ = "0.1.0"
