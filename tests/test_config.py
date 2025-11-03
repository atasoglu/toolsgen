"""Tests for configuration classes."""

from toolsgen.config import GenerationConfig, ModelConfig


def test_generation_config_defaults() -> None:
    """Test GenerationConfig default values."""
    config = GenerationConfig()
    assert config.num_samples == 10
    assert config.strategy == "random"
    assert config.seed is None


def test_generation_config_custom() -> None:
    """Test GenerationConfig with custom values."""
    config = GenerationConfig(num_samples=100, strategy="param_aware", seed=42)
    assert config.num_samples == 100
    assert config.strategy == "param_aware"
    assert config.seed == 42


def test_model_config_required() -> None:
    """Test ModelConfig requires model name."""
    config = ModelConfig(model="gpt-4")
    assert config.model == "gpt-4"
    assert config.base_url is None
    assert config.api_key_env == "OPENAI_API_KEY"
    assert config.temperature == 0.7


def test_model_config_custom() -> None:
    """Test ModelConfig with custom values."""
    config = ModelConfig(
        model="gpt-4o-mini",
        base_url="https://api.example.com",
        temperature=0.5,
        max_tokens=1000,
    )
    assert config.model == "gpt-4o-mini"
    assert config.base_url == "https://api.example.com"
    assert config.temperature == 0.5
    assert config.max_tokens == 1000
