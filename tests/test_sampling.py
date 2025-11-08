"""Tests for sampling functions."""

from __future__ import annotations

import random

import pytest

from toolsgen.schema import ToolFunction, ToolSpec
from toolsgen.sampling import (
    extract_keywords,
    tool_param_count,
    tool_semantic_similarity,
    batched_subsets,
    sample_param_aware_subset,
    sample_random_subset,
    sample_semantic_subset,
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

    batches = batched_subsets(
        tools, total=3, strategy="random", seed=42, k_min=2, k_max=2
    )

    assert len(batches) == 3
    for batch in batches:
        assert len(batch) == 2
        assert all(t in tools for t in batch)


def test_batched_subsets_empty() -> None:
    """Test batched subsets with empty tools."""
    batches = batched_subsets([], total=3, strategy="random", k_min=2, k_max=2)
    assert batches == []


@pytest.mark.parametrize(
    "strategy,expected_tool",
    [
        ("param_aware", "tool2"),  # Should prefer tool with most params (5)
        ("semantic", None),  # No specific expectation
        ("random", None),  # No specific expectation
    ],
)
def test_batched_subsets_strategies(strategy: str, expected_tool: str | None) -> None:
    """Test batched subsets with different strategies."""
    tools = [
        _create_tool("tool1", 1),
        _create_tool("tool2", 5),
        _create_tool("tool3", 3),
    ]

    batches = batched_subsets(
        tools, total=3, strategy=strategy, seed=42, k_min=2, k_max=2
    )

    assert len(batches) == 3
    for batch in batches:
        assert len(batch) == 2
        if expected_tool:
            assert any(t.function.name == expected_tool for t in batch)


def test_batched_subsets_with_chunking_and_wraparound() -> None:
    """Ensure batching respects chunk sizes and wraps deterministically."""
    tools = [
        _create_tool(f"tool{i}", i) for i in range(1, 6)
    ]  # 5 tools -> chunks of [2,2,1]

    batches = batched_subsets(
        tools,
        total=4,
        strategy="random",
        seed=7,
        k_min=1,
        k_max=2,
        batch_size=2,
        shuffle=False,
    )

    assert len(batches) == 4
    assert {t.function.name for t in batches[0]} == {"tool1", "tool2"}
    assert {t.function.name for t in batches[1]} == {"tool3", "tool4"}
    assert len(batches[2]) == 1
    assert batches[2][0].function.name == "tool5"
    assert {t.function.name for t in batches[3]} == {"tool1", "tool2"}


def test_batched_subsets_shuffle_applies_before_chunking() -> None:
    """Verify shuffle uses the provided seed before chunking."""
    tools = [_create_tool(f"tool{i}", i) for i in range(1, 5)]
    rng = random.Random(42)
    shuffled = tools.copy()
    rng.shuffle(shuffled)
    expected_names = {t.function.name for t in shuffled[:2]}

    batches = batched_subsets(
        tools,
        total=2,
        strategy="random",
        seed=42,
        k_min=2,
        k_max=2,
        batch_size=2,
        shuffle=True,
    )

    assert {t.function.name for t in batches[0]} == expected_names


def test_sample_semantic_subset() -> None:
    """Test semantic subset sampling."""
    tools = [
        ToolSpec(
            function=ToolFunction(
                name="email_send", description="Send an email message"
            )
        ),
        ToolSpec(
            function=ToolFunction(name="email_read", description="Read email messages")
        ),
        ToolSpec(
            function=ToolFunction(
                name="database_query", description="Query the database"
            )
        ),
        ToolSpec(
            function=ToolFunction(
                name="database_insert", description="Insert into database"
            )
        ),
    ]

    result = sample_semantic_subset(tools, k=3, seed=42)

    assert len(result) == 3
    assert all(t in tools for t in result)


def test_sample_semantic_subset_edge_cases() -> None:
    """Test semantic subset edge cases (empty, k equals/greater than len)."""
    # Empty tools
    assert sample_semantic_subset([], k=5) == []

    # k equals len
    tools = [_create_tool("tool1", 1), _create_tool("tool2", 2)]
    result = sample_semantic_subset(tools, k=2, seed=42)
    assert len(result) == 2

    # k greater than len
    result = sample_semantic_subset([_create_tool("tool1", 1)], k=5, seed=42)
    assert len(result) == 1


def test_tool_param_count() -> None:
    """Test parameter counting function."""
    # Tool with parameters
    tool1 = _create_tool("tool1", 3)
    assert tool_param_count(tool1) == 3

    # Tool with no parameters
    tool2 = ToolSpec(function=ToolFunction(name="tool2", parameters={}))
    assert tool_param_count(tool2) == 0

    # Tool with None parameters
    tool3 = ToolSpec(function=ToolFunction(name="tool3"))
    assert tool_param_count(tool3) == 0


def test_extract_keywords() -> None:
    """Test keyword extraction from text."""
    # Normal text
    keywords = extract_keywords("Send an email to the recipient")
    assert "send" in keywords
    assert "email" in keywords
    assert "recipient" in keywords
    assert "the" not in keywords  # Stop word
    assert "an" not in keywords  # Stop word

    # Empty text
    assert extract_keywords("") == set()
    # Note: extract_keywords expects str, but handles None gracefully in implementation

    # Text with special characters - note that underscores are treated as part of word
    keywords = extract_keywords("user-authentication & data_validation!")
    assert "user" in keywords
    assert "authentication" in keywords
    # data_validation is extracted as single word due to underscore
    assert "data_validation" in keywords or "validation" in keywords


def test_tool_semantic_similarity() -> None:
    """Test semantic similarity between tools."""
    tool1 = ToolSpec(
        function=ToolFunction(name="send_email", description="Send an email message")
    )
    tool2 = ToolSpec(
        function=ToolFunction(name="read_email", description="Read email messages")
    )
    tool3 = ToolSpec(
        function=ToolFunction(name="query_database", description="Query the database")
    )

    # Similar tools (both about email)
    sim_email = tool_semantic_similarity(tool1, tool2)
    assert sim_email > 0

    # Dissimilar tools
    sim_diff = tool_semantic_similarity(tool1, tool3)
    assert sim_diff < sim_email

    # Tool with no description
    tool_no_desc = ToolSpec(function=ToolFunction(name="test"))
    sim_no_desc = tool_semantic_similarity(tool1, tool_no_desc)
    assert sim_no_desc == 0.0
