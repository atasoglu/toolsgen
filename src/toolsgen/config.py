from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    """Configuration for dataset generation runtime.

    Attributes:
        num_samples: Number of samples to generate.
        strategy: Sampling strategy name ("random" or "param_aware").
        seed: Optional random seed for determinism.
        train_split: Fraction of records for training split (0.0-1.0). Default 1.0 (no split).
        language: Language name for user requests (e.g., "english", "turkish", "spanish"). Default "english".
    """

    num_samples: int = 10
    strategy: str = "random"
    seed: Optional[int] = None
    train_split: float = 1.0
    language: str = "english"


@dataclass
class ModelConfig:
    """Model configuration for an OpenAI-compatible endpoint."""

    model: str
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: Optional[int] = None


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
