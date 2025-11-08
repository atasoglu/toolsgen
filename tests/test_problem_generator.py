"""Tests for problem generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from toolsgen.problem_generator import generate_problem
from toolsgen.schema import ToolFunction, ToolSpec


@patch("toolsgen.problem_generator.OpenAI")
def test_generate_problem_success(mock_openai_class: MagicMock) -> None:
    """Test successful problem generation."""
    # Mock client and response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Create a new user account with email"
    mock_client.chat.completions.create.return_value = mock_response

    tools = [
        ToolSpec(function=ToolFunction(name="create_user", description="Create user"))
    ]

    result = generate_problem(
        client=mock_client,
        model="gpt-4",
        tools=tools,
        language="english",
        temperature=0.7,
    )

    assert result == "Create a new user account with email"

    # Verify API call
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] is None
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"


@patch("toolsgen.problem_generator.OpenAI")
@pytest.mark.parametrize(
    "content",
    [
        "",  # Empty
        None,  # None
        "   \n\t  ",  # Whitespace only
    ],
)
def test_generate_problem_invalid_responses(
    mock_openai_class: MagicMock, content: str | None
) -> None:
    """Test problem generation with invalid responses returns None."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="test"))]
    result = generate_problem(client=mock_client, model="gpt-4", tools=tools)

    assert result is None


@patch("toolsgen.problem_generator.OpenAI")
def test_generate_problem_strips_whitespace(mock_openai_class: MagicMock) -> None:
    """Test that problem generation strips leading/trailing whitespace."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "  \n  Test request  \n  "
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="test"))]

    result = generate_problem(client=mock_client, model="gpt-4", tools=tools)

    assert result == "Test request"


@patch("toolsgen.problem_generator.OpenAI")
def test_generate_problem_custom_language(mock_openai_class: MagicMock) -> None:
    """Test problem generation with custom language."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Crear una cuenta de usuario"
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="create_user"))]

    result = generate_problem(
        client=mock_client, model="gpt-4", tools=tools, language="spanish"
    )

    assert result == "Crear una cuenta de usuario"

    # Verify language was passed to prompt
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    system_message = call_kwargs["messages"][0]["content"]
    assert "spanish" in system_message.lower()


@patch("toolsgen.problem_generator.OpenAI")
def test_generate_problem_multiple_tools(mock_openai_class: MagicMock) -> None:
    """Test problem generation with multiple tools."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Send an email and save to database"
    mock_client.chat.completions.create.return_value = mock_response

    tools = [
        ToolSpec(function=ToolFunction(name="send_email", description="Send email")),
        ToolSpec(function=ToolFunction(name="save_db", description="Save to database")),
    ]

    result = generate_problem(client=mock_client, model="gpt-4", tools=tools)

    assert result == "Send an email and save to database"

    # Verify all tools are in system prompt
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    system_message = call_kwargs["messages"][0]["content"]
    assert "send_email" in system_message
    assert "save_db" in system_message


@patch("toolsgen.problem_generator.OpenAI")
def test_generate_problem_custom_max_tokens(mock_openai_class: MagicMock) -> None:
    """Test that max_tokens parameter is forwarded when provided."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Request"
    mock_client.chat.completions.create.return_value = mock_response

    tools = [ToolSpec(function=ToolFunction(name="test"))]

    generate_problem(
        client=mock_client,
        model="gpt-4",
        tools=tools,
        max_tokens=256,
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 256
