"""Problem generation for tool-calling datasets."""

from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from .prompts import (
    create_problem_generation_system_prompt,
    create_problem_generation_user_prompt,
)
from .schema import ToolSpec


def generate_problem(
    client: OpenAI,
    model: str,
    tools: List[ToolSpec],
    language: str = "english",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str | None:
    """Generate a natural language user request that requires tool usage.

    Args:
        client: OpenAI client for generation.
        model: Model name to use.
        tools: Available tools for the request.
        language: Language name for the request (e.g., "english", "turkish").
        temperature: Sampling temperature (higher for more creative requests).
        max_tokens: Optional maximum tokens to generate.

    Returns:
        Generated user request string, or None if generation fails.
    """
    system_prompt = create_problem_generation_system_prompt(tools, language)
    user_prompt = create_problem_generation_user_prompt()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    user_request = response.choices[0].message.content
    if not user_request or not user_request.strip():
        return None

    return user_request.strip()
