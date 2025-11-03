"""Tests for sampling functions."""

from toolsgen.schema import ToolFunction, ToolSpec
from toolsgen.sampling import (
    batched_subsets,
    sample_param_aware_subset,
    sample_random_subset,
)


def _create_tool(name: str, param_count: int) -> ToolSpec:
    """Helper to create a tool with specified parameter count."""
    props = {f"param_{i}": {"type": "string"} for i in range(param_count)}
    func = ToolFunction(
        name=name,
        description=f"Tool {name}",
        parameters={
            "type": "object",
            "properties": props,
        },
    )
    return ToolSpec(function=func)


def test_sample_random_subset() -> None:
    """Test random subset sampling."""
    tools = [
        _create_tool("tool1", 1),
        _create_tool("tool2", 2),
        _create_tool("tool3", 3),
    ]

    result = sample_random_subset(tools, k=2, seed=42)
    assert len(result) == 2
    assert all(t in tools for t in result)

    # Deterministic with same seed
    result2 = sample_random_subset(tools, k=2, seed=42)
    assert [t.function.name for t in result] == [t.function.name for t in result2]


def test_sample_random_subset_empty() -> None:
    """Test random subset with empty tools."""
    result = sample_random_subset([], k=5)
    assert result == []


def test_sample_random_subset_k_clamping() -> None:
    """Test that k is clamped to valid range."""
    tools = [_create_tool("tool1", 1)]
    result = sample_random_subset(tools, k=10, seed=42)
    assert len(result) == 1

    result = sample_random_subset(tools, k=0, seed=42)
    assert len(result) == 1  # Min 1


def test_sample_param_aware_subset() -> None:
    """Test parameter-aware subset sampling."""
    tools = [
        _create_tool("tool1", 1),
        _create_tool("tool2", 5),
        _create_tool("tool3", 3),
    ]

    result = sample_param_aware_subset(tools, k=2, seed=42)
    assert len(result) == 2
    # Should prefer tools with more parameters
    assert result[0].function.name == "tool2"  # 5 params
    assert result[1].function.name in ["tool1", "tool3"]


def test_batched_subsets() -> None:
    """Test batched subset generation."""
    tools = [
        _create_tool("tool1", 1),
        _create_tool("tool2", 2),
        _create_tool("tool3", 3),
    ]

    batches = batched_subsets(tools, batch_size=2, total=3, strategy="random", seed=42)

    assert len(batches) == 3
    for batch in batches:
        assert len(batch) == 2
        assert all(t in tools for t in batch)


def test_batched_subsets_empty() -> None:
    """Test batched subsets with empty tools."""
    batches = batched_subsets([], batch_size=2, total=3, strategy="random")
    assert batches == []
