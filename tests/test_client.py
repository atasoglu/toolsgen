"""Tests for OpenAI client utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from toolsgen.core.client import create_openai_client, create_structured_completion
from toolsgen.core.config import ModelConfig


def test_create_openai_client_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful OpenAI client creation."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    config = ModelConfig(model="gpt-4")
    client = create_openai_client(config)

    assert client is not None
    assert client.api_key == "test-api-key"


def test_create_openai_client_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test client creation fails without API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = ModelConfig(model="gpt-4")

    with pytest.raises(ValueError, match="API key not found"):
        create_openai_client(config)


def test_create_openai_client_custom_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test client creation with custom API key environment variable."""
    monkeypatch.setenv("CUSTOM_API_KEY", "custom-key")

    config = ModelConfig(model="gpt-4", api_key_env="CUSTOM_API_KEY")
    client = create_openai_client(config)

    assert client.api_key == "custom-key"


def test_create_openai_client_with_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test client creation with custom base URL."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = ModelConfig(model="gpt-4", base_url="https://custom.api.com/v1")
    client = create_openai_client(config)

    # OpenAI client normalizes URLs by adding trailing slash
    assert "https://custom.api.com/v1" in str(client.base_url)


def test_create_openai_client_with_openai_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test client creation with additional OpenAI parameters."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = ModelConfig(
        model="gpt-4",
        openai_params={
            "timeout": 120.0,
            "max_retries": 3,
        },
    )
    client = create_openai_client(config)

    assert client.timeout == 120.0
    assert client.max_retries == 3


@patch("toolsgen.core.client.OpenAI")
def test_create_structured_completion_success(mock_openai_class: MagicMock) -> None:
    """Test successful structured completion."""

    # Define a test response model
    class TestResponse(BaseModel):
        message: str
        count: int

    # Mock OpenAI client and response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"message": "hello", "count": 42}'
    mock_client.chat.completions.create.return_value = mock_response

    messages = [{"role": "user", "content": "Test"}]
    result = create_structured_completion(
        client=mock_client,
        model="gpt-4",
        messages=messages,
        response_model=TestResponse,
        temperature=0.7,
        max_tokens=100,
    )

    assert isinstance(result, TestResponse)
    assert result.message == "hello"
    assert result.count == 42

    # Verify the API was called correctly
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 100
    assert call_kwargs["response_format"]["type"] == "json_schema"


@patch("toolsgen.core.client.OpenAI")
def test_create_structured_completion_empty_content(
    mock_openai_class: MagicMock,
) -> None:
    """Test structured completion fails with empty content."""

    class TestResponse(BaseModel):
        message: str

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None
    mock_client.chat.completions.create.return_value = mock_response

    messages = [{"role": "user", "content": "Test"}]

    with pytest.raises(ValueError, match="LLM returned empty content"):
        create_structured_completion(
            client=mock_client,
            model="gpt-4",
            messages=messages,
            response_model=TestResponse,
        )


@patch("toolsgen.core.client.OpenAI")
def test_create_structured_completion_with_schema(mock_openai_class: MagicMock) -> None:
    """Test that structured completion uses correct JSON schema."""

    class TestResponse(BaseModel):
        value: str

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"value": "test"}'
    mock_client.chat.completions.create.return_value = mock_response

    messages = [{"role": "user", "content": "Test"}]
    create_structured_completion(
        client=mock_client,
        model="gpt-4",
        messages=messages,
        response_model=TestResponse,
    )

    # Verify schema was passed correctly
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "response_format" in call_kwargs
    response_format = call_kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "TestResponse"
    assert response_format["json_schema"]["strict"] is True
    assert "schema" in response_format["json_schema"]
