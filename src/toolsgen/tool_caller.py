"""Tool call generation for datasets."""

from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from .prompts import create_tool_caller_system_prompt
from .schema import AssistantToolCall, ToolSpec


def generate_tool_calls(
    client: OpenAI,
    model: str,
    user_request: str,
    tools: List[ToolSpec],
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> List[AssistantToolCall]:
    """Generate tool calls for a given user request.

    Args:
        client: OpenAI client for generation.
        model: Model name to use.
        user_request: The user's request to fulfill.
        tools: Available tool specifications.
        temperature: Sampling temperature (lower for more consistent calls).
        max_tokens: Optional maximum tokens to generate.

    Returns:
        List of generated tool calls. Empty list if no valid calls generated.
    """
    tools_dict = [tool.model_dump() for tool in tools]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": create_tool_caller_system_prompt()},
            {"role": "user", "content": user_request},
        ],
        tools=tools_dict,
        tool_choice="auto",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Extract and validate tool calls
    message = response.choices[0].message
    tool_calls_data = message.tool_calls or []
    tool_calls = []

    for tc in tool_calls_data:
        try:
            tool_calls.append(
                AssistantToolCall.model_validate(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
            )
        except Exception:
            continue

    return tool_calls
