"""OpenAI client helpers and utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel

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


def create_structured_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    response_model: type[BaseModel],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> BaseModel:
    """Create a chat completion with structured output.

    Args:
        client: OpenAI client instance.
        model: Model name to use.
        messages: List of message dictionaries.
        response_model: Pydantic model class for response structure.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        Instance of response_model populated with API response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
                "strict": True,
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty content")

    return response_model.model_validate_json(content)
