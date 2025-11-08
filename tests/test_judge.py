"""Tests for LLM-as-a-judge scoring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from toolsgen.judge import JudgeResponse, judge_tool_calls
from toolsgen.schema import AssistantToolCall, ToolFunction, ToolSpec


def test_judge_response_creation() -> None:
    """Test creating a JudgeResponse."""
    response = JudgeResponse(
        tool_relevance=0.4,
        argument_quality=0.35,
        clarity=0.15,
        score=0.9,
        verdict="accept",
        rationale="Excellent tool usage",
    )

    assert response.tool_relevance == 0.4
    assert response.argument_quality == 0.35
    assert response.clarity == 0.15
    assert response.score == 0.9
    assert response.verdict == "accept"
    assert response.rationale == "Excellent tool usage"


def test_judge_response_to_dict() -> None:
    """Test converting JudgeResponse to dict."""
    response = JudgeResponse(
        tool_relevance=0.3,
        argument_quality=0.3,
        clarity=0.1,
        score=0.7,
        verdict="accept",
        rationale="Good",
    )

    result = response.to_dict()

    assert result["tool_relevance"] == 0.3
    assert result["argument_quality"] == 0.3
    assert result["clarity"] == 0.1
    assert result["score"] == 0.7
    assert result["verdict"] == "accept"
    assert result["rationale"] == "Good"
    assert result["rubric_version"] == "0.1.0"


@patch("toolsgen.judge.OpenAI")
def test_judge_tool_calls_success(mock_openai_class: MagicMock) -> None:
    """Test successful tool call judging."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = """
    {
        "tool_relevance": 0.4,
        "argument_quality": 0.35,
        "clarity": 0.15,
        "score": 0.9,
        "verdict": "accept",
        "rationale": "Excellent tool usage with proper arguments"
    }
    """
    mock_client.chat.completions.create.return_value = mock_response

    tools = [
        ToolSpec(
            function=ToolFunction(
                name="send_email",
                description="Send email",
                parameters={"type": "object", "properties": {"to": {"type": "string"}}},
            )
        )
    ]
    tool_calls = [
        AssistantToolCall(
            id="call_1",
            function={"name": "send_email", "arguments": '{"to": "user@example.com"}'},
        )
    ]

    result = judge_tool_calls(
        client=mock_client,
        model="gpt-4",
        user_request="Send an email to user@example.com",
        tools=tools,
        tool_calls=tool_calls,
        temperature=0.3,
    )

    assert isinstance(result, JudgeResponse)
    assert result.score == 0.9
    assert result.verdict == "accept"
    assert result.tool_relevance == 0.4

    # Verify API call
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4"
    assert call_kwargs["temperature"] == 0.3
    assert call_kwargs["max_tokens"] is None
    assert call_kwargs["response_format"]["type"] == "json_schema"


@patch("toolsgen.judge.OpenAI")
def test_judge_tool_calls_empty_calls(mock_openai_class: MagicMock) -> None:
    """Test judging with no tool calls."""
    mock_client = MagicMock()

    tools = [ToolSpec(function=ToolFunction(name="test"))]
    tool_calls: list[AssistantToolCall] = []

    result = judge_tool_calls(
        client=mock_client,
        model="gpt-4",
        user_request="Test",
        tools=tools,
        tool_calls=tool_calls,
    )

    # Should return reject verdict without API call
    assert result.verdict == "reject"
    assert result.score == 0.0
    assert result.tool_relevance == 0.0
    assert result.argument_quality == 0.0
    assert result.clarity == 0.0
    assert "No tool calls" in result.rationale

    # Should not call API
    mock_client.chat.completions.create.assert_not_called()


@patch("toolsgen.judge.OpenAI")
def test_judge_tool_calls_reject_verdict(mock_openai_class: MagicMock) -> None:
    """Test judging with reject verdict."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = """
    {
        "tool_relevance": 0.2,
        "argument_quality": 0.15,
        "clarity": 0.1,
        "score": 0.45,
        "verdict": "reject",
        "rationale": "Incorrect tool usage"
    }
    """
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="send_email"))]
    tool_calls = [
        AssistantToolCall(
            id="call_1", function={"name": "send_email", "arguments": "{}"}
        )
    ]

    result = judge_tool_calls(
        client=mock_client,
        model="gpt-4",
        user_request="Test",
        tools=tools,
        tool_calls=tool_calls,
    )

    assert result.verdict == "reject"
    assert result.score == 0.45


@patch("toolsgen.judge.OpenAI")
def test_judge_tool_calls_empty_content_error(mock_openai_class: MagicMock) -> None:
    """Test judging fails with empty content."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="test"))]
    tool_calls = [
        AssistantToolCall(id="call_1", function={"name": "test", "arguments": "{}"})
    ]

    with pytest.raises(ValueError, match="Judge LLM returned empty content"):
        judge_tool_calls(
            client=mock_client,
            model="gpt-4",
            user_request="Test",
            tools=tools,
            tool_calls=tool_calls,
        )


@patch("toolsgen.judge.OpenAI")
def test_judge_tool_calls_with_schema(mock_openai_class: MagicMock) -> None:
    """Test that judging uses correct JSON schema."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = """
    {
        "tool_relevance": 0.3,
        "argument_quality": 0.3,
        "clarity": 0.1,
        "score": 0.7,
        "verdict": "accept",
        "rationale": "Good"
    }
    """
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="test"))]
    tool_calls = [
        AssistantToolCall(id="call_1", function={"name": "test", "arguments": "{}"})
    ]

    judge_tool_calls(
        client=mock_client,
        model="gpt-4",
        user_request="Test",
        tools=tools,
        tool_calls=tool_calls,
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    response_format = call_kwargs["response_format"]

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "JudgeResponse"
    assert response_format["json_schema"]["strict"] is True
    assert "schema" in response_format["json_schema"]


@patch("toolsgen.judge.OpenAI")
def test_judge_tool_calls_multiple_calls(mock_openai_class: MagicMock) -> None:
    """Test judging with multiple tool calls."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = """
    {
        "tool_relevance": 0.4,
        "argument_quality": 0.35,
        "clarity": 0.15,
        "score": 0.9,
        "verdict": "accept",
        "rationale": "Good multi-tool usage"
    }
    """
    mock_client.chat.completions.create.return_value = mock_response

    tools = [
        ToolSpec(function=ToolFunction(name="send_email")),
        ToolSpec(function=ToolFunction(name="save_file")),
    ]
    tool_calls = [
        AssistantToolCall(
            id="call_1", function={"name": "send_email", "arguments": "{}"}
        ),
        AssistantToolCall(
            id="call_2", function={"name": "save_file", "arguments": "{}"}
        ),
    ]

    result = judge_tool_calls(
        client=mock_client,
        model="gpt-4",
        user_request="Send email and save",
        tools=tools,
        tool_calls=tool_calls,
    )

    assert result.verdict == "accept"

    # Verify both tool calls appear in system prompt
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    system_message = call_kwargs["messages"][0]["content"]
    assert "send_email" in system_message
    assert "save_file" in system_message


@patch("toolsgen.judge.OpenAI")
def test_judge_tool_calls_custom_max_tokens(mock_openai_class: MagicMock) -> None:
    """Test that max_tokens parameter is forwarded when provided."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = """
    {
        "tool_relevance": 0.3,
        "argument_quality": 0.3,
        "clarity": 0.1,
        "score": 0.7,
        "verdict": "accept",
        "rationale": "Good"
    }
    """
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="test"))]
    tool_calls = [
        AssistantToolCall(id="call_1", function={"name": "test", "arguments": "{}"})
    ]

    judge_tool_calls(
        client=mock_client,
        model="gpt-4",
        user_request="Test",
        tools=tools,
        tool_calls=tool_calls,
        max_tokens=640,
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 640
