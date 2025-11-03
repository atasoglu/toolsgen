"""Prompt templates for problem generation, tool calling, and judging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schema import AssistantToolCall, ToolSpec

PROMPTS_DIR = Path(__file__).parent / "prompts"


def create_problem_generation_prompt(
    tools: List[ToolSpec], language: str = "english"
) -> str:
    """Create a prompt for generating natural language user requests.

    Args:
        tools: List of available tools.
        language: Language name for user requests (e.g., "english", "turkish", "spanish").

    Returns:
        System prompt for generating user requests.
    """
    tools_desc = [
        f"- {t.function.name}: {t.function.description or 'No description'}"
        for t in tools
    ]
    tools_list = "\n".join(tools_desc)

    template = (PROMPTS_DIR / "problem_generation.txt").read_text(encoding="utf-8")
    return template.format(tools_list=tools_list, language=language)


def create_problem_generation_user_message() -> str:
    """User message for problem generation prompt.

    Returns:
        User message content.
    """
    return "Generate a realistic user request that would require using these tools."


def create_caller_system_prompt() -> str:
    """System prompt for tool-calling assistant generation.

    Returns:
        System prompt for generating tool calls.
    """
    return "You are a helpful assistant that uses tools to answer user requests. Generate appropriate tool calls based on the user's request."


def create_judge_prompt(
    user_request: str,
    tools: List[ToolSpec],
    tool_calls: List[AssistantToolCall],
) -> str:
    """Create the judge prompt for evaluating tool calls.

    Args:
        user_request: The original user request.
        tools: Available tool specifications.
        tool_calls: Generated tool calls to evaluate.

    Returns:
        System prompt for the judge LLM.
    """
    tools_desc = [
        f"- {t.function.name}: {t.function.description or 'No description'} (params: {json.dumps(t.function.parameters, indent=2)})"
        for t in tools
    ]
    tools_list = "\n".join(tools_desc)

    calls_list = (
        "\n".join(
            f"- {tc.function.get('name', 'unknown')}({tc.function.get('arguments', '{}')})"
            for tc in tool_calls
        )
        or "None"
    )

    template = (PROMPTS_DIR / "judge.txt").read_text(encoding="utf-8")
    return template.format(
        user_request=user_request,
        tools_list=tools_list,
        calls_list=calls_list,
    )


def create_judge_user_message() -> str:
    """User message for judge prompt.

    Returns:
        User message content.
    """
    return "Evaluate the tool calls according to the rubric and return the JSON score."
