"""Tests for tool call generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from toolsgen.schema import ToolFunction, ToolSpec
from toolsgen.tool_caller import generate_tool_calls


@patch("toolsgen.tool_caller.OpenAI")
def test_generate_tool_calls_success(mock_openai_class: MagicMock) -> None:
    """Test successful tool call generation."""
    # Mock client and response
    mock_client = MagicMock()
    mock_response = MagicMock()

    # Mock tool call
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "send_email"
    mock_tool_call.function.arguments = '{"to": "user@example.com", "subject": "Hello"}'

    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_client.chat.completions.create.return_value = mock_response

    tools = [
        ToolSpec(
            function=ToolFunction(
                name="send_email",
                description="Send email",
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                    },
                },
            )
        )
    ]

    result = generate_tool_calls(
        client=mock_client,
        model="gpt-4",
        user_request="Send an email to user@example.com",
        tools=tools,
        temperature=0.3,
    )

    assert len(result) == 1
    assert result[0].id == "call_123"
    assert result[0].type == "function"
    assert result[0].function["name"] == "send_email"
    assert "user@example.com" in result[0].function["arguments"]

    # Verify API call
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4"
    assert call_kwargs["temperature"] == 0.3
    assert call_kwargs["max_tokens"] is None
    assert call_kwargs["tool_choice"] == "auto"


@patch("toolsgen.tool_caller.OpenAI")
def test_generate_tool_calls_no_calls(mock_openai_class: MagicMock) -> None:
    """Test tool call generation with no tool calls returned."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="test_func"))]

    result = generate_tool_calls(
        client=mock_client, model="gpt-4", user_request="Test", tools=tools
    )

    assert result == []


@patch("toolsgen.tool_caller.OpenAI")
def test_generate_tool_calls_empty_list(mock_openai_class: MagicMock) -> None:
    """Test tool call generation with empty tool calls list."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = []
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="test_func"))]

    result = generate_tool_calls(
        client=mock_client, model="gpt-4", user_request="Test", tools=tools
    )

    assert result == []


@patch("toolsgen.tool_caller.OpenAI")
def test_generate_tool_calls_multiple(mock_openai_class: MagicMock) -> None:
    """Test generating multiple tool calls."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    # Mock multiple tool calls
    mock_call1 = MagicMock()
    mock_call1.id = "call_1"
    mock_call1.type = "function"
    mock_call1.function.name = "send_email"
    mock_call1.function.arguments = "{}"

    mock_call2 = MagicMock()
    mock_call2.id = "call_2"
    mock_call2.type = "function"
    mock_call2.function.name = "save_file"
    mock_call2.function.arguments = "{}"

    mock_response.choices[0].message.tool_calls = [mock_call1, mock_call2]
    mock_client.chat.completions.create.return_value = mock_response

    tools = [
        ToolSpec(function=ToolFunction(name="send_email")),
        ToolSpec(function=ToolFunction(name="save_file")),
    ]

    result = generate_tool_calls(
        client=mock_client,
        model="gpt-4",
        user_request="Send email and save to file",
        tools=tools,
    )

    assert len(result) == 2
    assert result[0].function["name"] == "send_email"
    assert result[1].function["name"] == "save_file"


@patch("toolsgen.tool_caller.OpenAI")
def test_generate_tool_calls_invalid_call_skipped(mock_openai_class: MagicMock) -> None:
    """Test that invalid tool calls are skipped."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    # Mock one valid and one invalid call
    mock_valid = MagicMock()
    mock_valid.id = "call_1"
    mock_valid.type = "function"
    mock_valid.function.name = "valid_func"
    mock_valid.function.arguments = "{}"

    mock_invalid = MagicMock()
    # Make validation fail by raising exception
    mock_invalid.id = None

    mock_response.choices[0].message.tool_calls = [mock_valid, mock_invalid]
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="valid_func"))]

    result = generate_tool_calls(
        client=mock_client, model="gpt-4", user_request="Test", tools=tools
    )

    # Should only return valid call
    assert len(result) == 1
    assert result[0].function["name"] == "valid_func"


@patch("toolsgen.tool_caller.OpenAI")
def test_generate_tool_calls_with_tools_dict(mock_openai_class: MagicMock) -> None:
    """Test that tools are properly converted to dict for API."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = []
    mock_client.chat.completions.create.return_value = mock_response

    tools = [
        ToolSpec(
            function=ToolFunction(
                name="test_func",
                description="Test function",
                parameters={"type": "object", "properties": {"x": {"type": "string"}}},
            )
        )
    ]

    generate_tool_calls(
        client=mock_client, model="gpt-4", user_request="Test", tools=tools
    )

    # Verify tools were converted to dict and passed
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "tools" in call_kwargs
    tools_arg = call_kwargs["tools"]
    assert isinstance(tools_arg, list)
    assert len(tools_arg) == 1
    assert tools_arg[0]["function"]["name"] == "test_func"


@patch("toolsgen.tool_caller.OpenAI")
def test_generate_tool_calls_system_prompt(mock_openai_class: MagicMock) -> None:
    """Test that system prompt is included in messages."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = []
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="test"))]
    user_request = "Test request"

    generate_tool_calls(
        client=mock_client, model="gpt-4", user_request=user_request, tools=tools
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == user_request


@patch("toolsgen.tool_caller.OpenAI")
def test_generate_tool_calls_custom_max_tokens(mock_openai_class: MagicMock) -> None:
    """Test that max_tokens parameter is forwarded when provided."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = []
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="test"))]

    generate_tool_calls(
        client=mock_client,
        model="gpt-4",
        user_request="Test",
        tools=tools,
        max_tokens=1024,
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 1024
