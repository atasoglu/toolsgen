"""Tests for prompt templates."""

from __future__ import annotations

from toolsgen.prompts import (
    create_judge_system_prompt,
    create_judge_user_prompt,
    create_problem_generation_system_prompt,
    create_problem_generation_user_prompt,
    create_tool_caller_system_prompt,
)
from toolsgen.schema import AssistantToolCall, ToolFunction, ToolSpec


def test_create_problem_generation_system_prompt() -> None:
    """Test problem generation system prompt creation."""
    tools = [
        ToolSpec(
            function=ToolFunction(
                name="send_email", description="Send an email message"
            )
        ),
        ToolSpec(
            function=ToolFunction(name="read_file", description="Read a file from disk")
        ),
    ]

    prompt = create_problem_generation_system_prompt(tools, language="english")

    assert "send_email" in prompt
    assert "read_file" in prompt
    assert "Send an email message" in prompt
    assert "Read a file from disk" in prompt
    assert "english" in prompt.lower()


def test_create_problem_generation_prompts_edge_cases() -> None:
    """Test problem generation prompts with various configurations."""
    # No description
    tools_no_desc = [ToolSpec(function=ToolFunction(name="test_tool"))]
    prompt = create_problem_generation_system_prompt(tools_no_desc)
    assert "test_tool" in prompt
    assert "No description" in prompt

    # Custom language
    tools = [ToolSpec(function=ToolFunction(name="test_func", description="Test"))]
    prompt = create_problem_generation_system_prompt(tools, language="spanish")
    assert "spanish" in prompt.lower()


def test_create_simple_prompts() -> None:
    """Test simple prompt creation functions return non-empty strings."""
    assert len(create_problem_generation_user_prompt()) > 0
    assert len(create_tool_caller_system_prompt()) > 0
    assert len(create_judge_user_prompt()) > 0


def test_create_judge_system_prompt() -> None:
    """Test judge system prompt creation."""
    user_request = "Send an email to john@example.com"
    tools = [
        ToolSpec(
            function=ToolFunction(
                name="send_email",
                description="Send an email",
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                },
            )
        ),
    ]
    tool_calls = [
        AssistantToolCall(
            id="call_1",
            function={
                "name": "send_email",
                "arguments": '{"to": "john@example.com", "subject": "Hello", "body": "Test"}',
            },
        )
    ]

    prompt = create_judge_system_prompt(user_request, tools, tool_calls)

    assert user_request in prompt
    assert "send_email" in prompt
    assert "Send an email" in prompt
    assert "call_1" not in prompt  # IDs should not be in prompt


def test_create_judge_system_prompt_no_calls() -> None:
    """Test judge system prompt with no tool calls."""
    user_request = "Test request"
    tools = [ToolSpec(function=ToolFunction(name="test_func"))]
    tool_calls: list[AssistantToolCall] = []

    prompt = create_judge_system_prompt(user_request, tools, tool_calls)

    assert user_request in prompt
    assert "None" in prompt  # Should show "None" for empty calls


def test_create_judge_system_prompt_multiple_tools() -> None:
    """Test judge system prompt with multiple tools and calls."""
    user_request = "Send email and save to file"
    tools = [
        ToolSpec(
            function=ToolFunction(
                name="send_email",
                description="Send email",
                parameters={"type": "object", "properties": {}},
            )
        ),
        ToolSpec(
            function=ToolFunction(
                name="save_file",
                description="Save file",
                parameters={"type": "object", "properties": {}},
            )
        ),
    ]
    tool_calls = [
        AssistantToolCall(
            id="call_1", function={"name": "send_email", "arguments": "{}"}
        ),
        AssistantToolCall(
            id="call_2", function={"name": "save_file", "arguments": "{}"}
        ),
    ]

    prompt = create_judge_system_prompt(user_request, tools, tool_calls)

    assert "send_email" in prompt
    assert "save_file" in prompt
    assert user_request in prompt
