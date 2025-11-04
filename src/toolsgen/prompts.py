"""Prompt templates for problem generation, tool calling, and judging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schema import AssistantToolCall, ToolSpec

PROMPTS_DIR = Path(__file__).parent / "prompts"


def create_problem_generation_system_prompt(
    tools: List[ToolSpec], language: str = "english"
) -> str:
    """Create system prompt for generating natural language user requests.

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

    template = (PROMPTS_DIR / "problem_generation_system.txt").read_text(
        encoding="utf-8"
    )
    return template.format(tools_list=tools_list, language=language)


def create_problem_generation_user_prompt() -> str:
    """Create user prompt for problem generation.

    Returns:
        User prompt content.
    """
    return (
        (PROMPTS_DIR / "problem_generation_user.txt")
        .read_text(encoding="utf-8")
        .strip()
    )


def create_tool_caller_system_prompt() -> str:
    """System prompt for tool-calling assistant generation.

    Returns:
        System prompt for generating tool calls.
    """
    return (PROMPTS_DIR / "tool_caller_system.txt").read_text(encoding="utf-8").strip()


def create_judge_system_prompt(
    user_request: str,
    tools: List[ToolSpec],
    tool_calls: List[AssistantToolCall],
) -> str:
    """Create system prompt for evaluating tool calls.

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

    template = (PROMPTS_DIR / "judge_system.txt").read_text(encoding="utf-8")
    return template.format(
        user_request=user_request,
        tools_list=tools_list,
        calls_list=calls_list,
    )


def create_judge_user_prompt() -> str:
    """Create user prompt for judge evaluation.

    Returns:
        User prompt content.
    """
    return (PROMPTS_DIR / "judge_user.txt").read_text(encoding="utf-8").strip()
