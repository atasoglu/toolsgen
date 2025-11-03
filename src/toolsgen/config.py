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
    """

    num_samples: int = 10
    strategy: str = "random"
    seed: Optional[int] = None
    train_split: float = 1.0


@dataclass
class ModelConfig:
    """Model configuration for an OpenAI-compatible endpoint."""

    model: str
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
