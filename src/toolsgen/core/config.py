"""Configuration classes for dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class GenerationConfig:
    """Configuration for dataset generation runtime.

    Attributes:
        num_samples: Number of samples to generate.
        strategy: Sampling strategy name ("random", "param_aware", or "semantic").
        seed: Optional random seed for determinism.
        train_split: Fraction of records for training split (0.0-1.0). Default 1.0 (no split).
        language: Language name for user requests (e.g., "english", "turkish", "spanish"). Default "english".
        max_attempts: Maximum retry attempts per sample. Default 3.
        k_min: Minimum number of tools per sample. Default 1.
        k_max: Maximum number of tools per sample. Default None (uses all available tools).
        batch_size: Optional chunk size for tool batching. Default None (single batch).
        shuffle_tools: Whether to shuffle tools before batching. Default False.
        num_workers: Number of concurrent worker processes. Default 1 (sequential).
        worker_batch_size: Samples processed per worker task submission. Default 1.
    """

    num_samples: int = 10
    strategy: Literal["random", "param_aware", "semantic"] = "random"
    seed: Optional[int] = None
    train_split: float = 1.0
    language: str = "english"
    max_attempts: int = 3
    k_min: int = 1
    k_max: Optional[int] = None
    batch_size: Optional[int] = None
    shuffle_tools: bool = False
    num_workers: int = 1
    worker_batch_size: int = 1


@dataclass
class ModelConfig:
    """Model configuration for an OpenAI-compatible endpoint.

    Attributes:
        model: Model name to use.
        base_url: Optional custom base URL for API.
        api_key_env: Environment variable name for API key. Default: "OPENAI_API_KEY".
        temperature: Sampling temperature (0.0-2.0). Default: 0.7.
        max_tokens: Maximum tokens to generate.
        openai_params: Additional parameters to pass to OpenAI client.
            Examples: {"organization": "org-xxx", "timeout": 120.0, "default_headers": {...}}
    """

    model: str
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    openai_params: Optional[dict] = None


@dataclass
class RoleBasedModelConfig:
    """Configuration for different LLM roles in dataset generation.

    Attributes:
        problem_generator: Config for generating user requests.
        tool_caller: Config for generating tool calls.
        judge: Config for evaluating tool calls.
    """

    problem_generator: ModelConfig
    tool_caller: ModelConfig
    judge: ModelConfig

    @classmethod
    def from_single_config(cls, config: ModelConfig) -> "RoleBasedModelConfig":
        """Create role-based config from a single ModelConfig."""
        return cls(
            problem_generator=config,
            tool_caller=config,
            judge=config,
        )
