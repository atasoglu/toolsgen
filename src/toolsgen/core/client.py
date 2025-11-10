"""OpenAI client helpers and utilities."""

from __future__ import annotations

import os
from openai import OpenAI
from .config import ModelConfig


def create_openai_client(model_config: ModelConfig) -> OpenAI:
    """Create an OpenAI client from model configuration.

    Args:
        model_config: Model configuration with API settings.

    Returns:
        Configured OpenAI client.

    Raises:
        ValueError: If API key is not found.
    """
    api_key = os.environ.get(model_config.api_key_env)
    if not api_key:
        raise ValueError(
            f"API key not found. Set {model_config.api_key_env} environment variable."
        )

    # Base parameters
    params = {
        "api_key": api_key,
        "base_url": model_config.base_url,
    }

    # Merge with additional OpenAI parameters
    if model_config.openai_params:
        params.update(model_config.openai_params)

    return OpenAI(**params)
