"""Tests for configuration classes."""

from toolsgen.core import GenerationConfig, ModelConfig


def test_generation_config_defaults() -> None:
    """Test GenerationConfig default values."""
    config = GenerationConfig()
    assert config.num_samples == 10
    assert config.strategy == "random"
    assert config.seed is None
    assert config.batch_size is None
    assert config.shuffle_tools is False
    assert config.num_workers == 1
    assert config.worker_batch_size == 1


def test_generation_config_custom() -> None:
    """Test GenerationConfig with custom values."""
    config = GenerationConfig(
        num_samples=100,
        strategy="param_aware",
        seed=42,
        batch_size=4,
        shuffle_tools=True,
        num_workers=4,
        worker_batch_size=3,
    )
    assert config.num_samples == 100
    assert config.strategy == "param_aware"
    assert config.seed == 42
    assert config.batch_size == 4
    assert config.shuffle_tools is True
    assert config.num_workers == 4
    assert config.worker_batch_size == 3


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
