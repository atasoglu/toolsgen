from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    """Configuration for dataset generation runtime.

    Attributes:
        num_samples: Number of samples to generate.
        strategy: Sampling strategy name (e.g., "random", "semantic").
        seed: Optional random seed for determinism.
    """

    num_samples: int = 10
    strategy: str = "random"
    seed: Optional[int] = None


@dataclass
class ModelConfig:
    """Model configuration for an OpenAI-compatible endpoint."""

    model: str
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
