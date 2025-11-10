"""Tests for OpenAI client utilities."""

from __future__ import annotations

import pytest

from toolsgen.core.client import create_openai_client
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
