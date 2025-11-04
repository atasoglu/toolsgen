from __future__ import annotations

import random
import re
from typing import List, Sequence

from .schema import ToolSpec


def _tool_param_count(tool: ToolSpec) -> int:
    params = tool.function.parameters or {}
    props = params.get("properties") if isinstance(params, dict) else None
    if isinstance(props, dict):
        return len(props)
    return 0


def sample_random_subset(
    tools: Sequence[ToolSpec], *, k: int, seed: int | None = None
) -> List[ToolSpec]:
    """Sample k tools uniformly at random without replacement.

    Ensures 1 <= k <= len(tools). Deterministic given a seed.
    """

    if not tools:
        return []
    k = max(1, min(k, len(tools)))
    rng = random.Random(seed)
    indices = list(range(len(tools)))
    rng.shuffle(indices)
    chosen = indices[:k]
    return [tools[i] for i in chosen]


def sample_param_aware_subset(
    tools: Sequence[ToolSpec], *, k: int, seed: int | None = None
) -> List[ToolSpec]:
    """Prefer tools with more parameters to encourage richer arguments.

    Strategy: sort by parameter count desc; break ties with a seeded shuffle;
    then pick top-k.
    """

    if not tools:
        return []
    k = max(1, min(k, len(tools)))
    rng = random.Random(seed)
    # Pair tools with param count
    scored = [(t, _tool_param_count(t)) for t in tools]
    # Shuffle for tie-breaking
    rng.shuffle(scored)
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:k]]


def _extract_keywords(text: str) -> set[str]:
    """Extract keywords from text for semantic matching.

    Args:
        text: Text to extract keywords from.

    Returns:
        Set of lowercase keywords.
    """
    if not text:
        return set()
    words = re.findall(r"\b\w+\b", text.lower())
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
    return {w for w in words if len(w) > 2 and w not in stop_words}


def _tool_semantic_similarity(tool1: ToolSpec, tool2: ToolSpec) -> float:
    """Calculate semantic similarity between two tools using Jaccard similarity.

    Args:
        tool1: First tool specification.
        tool2: Second tool specification.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    keywords1 = _extract_keywords(tool1.function.name) | _extract_keywords(
        tool1.function.description or ""
    )
    keywords2 = _extract_keywords(tool2.function.name) | _extract_keywords(
        tool2.function.description or ""
    )

    if not keywords1 or not keywords2:
        return 0.0

    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)
    return intersection / union if union > 0 else 0.0


def sample_semantic_subset(
    tools: Sequence[ToolSpec], *, k: int, seed: int | None = None
) -> List[ToolSpec]:
    """Sample tools with semantic similarity preference.

    Strategy: Start with a random tool, then iteratively add tools that have
    moderate similarity to already selected tools (encourages related tools
    while avoiding duplicates).

    Args:
        tools: Sequence of tools to sample from.
        k: Number of tools to sample.
        seed: Optional random seed for determinism.

    Returns:
        List of sampled tools.
    """
    if not tools:
        return []
    k = max(1, min(k, len(tools)))
    rng = random.Random(seed)

    if len(tools) <= k:
        return list(tools)

    # Start with a random tool
    remaining = list(tools)
    rng.shuffle(remaining)
    selected = [remaining.pop(0)]

    # Add tools with moderate similarity to selected ones
    while len(selected) < k and remaining:
        best_tool = None
        best_score = -1.0

        for tool in remaining:
            avg_sim = sum(_tool_semantic_similarity(tool, s) for s in selected) / len(
                selected
            )
            # Prefer moderate similarity (0.2-0.6), penalize very high or very low
            score = avg_sim if 0.2 <= avg_sim <= 0.6 else avg_sim * 0.5
            if score > best_score:
                best_score = score
                best_tool = tool

        if best_tool:
            selected.append(best_tool)
            remaining.remove(best_tool)
        else:
            # Fallback to random if no good match
            selected.append(remaining.pop(0))

    return selected


def batched_subsets(
    tools: Sequence[ToolSpec],
    *,
    batch_size: int,
    total: int,
    strategy: str = "random",
    seed: int | None = None,
    k_min: int = 1,
    k_max: int | None = None,
) -> List[List[ToolSpec]]:
    """Produce multiple subsets according to a strategy.

    Notes:
    - `batch_size` determines the size (k) of each sampled subset.
    - `total` determines how many subsets to produce.
    - `strategy`: "random", "param_aware", or "semantic".
    - If you want variable subset sizes, adjust `k_min`/`k_max` and set
      `batch_size` to a value outside [1, len(tools)] to fallback to range.
    """

    if not tools:
        return []

    # Clamp batch_size to valid k; if invalid, use the k_min..k_max range uniformly
    max_allowed = len(tools)
    fixed_k = (
        max(1, min(batch_size, max_allowed)) if 1 <= batch_size <= max_allowed else None
    )

    k_max = k_max or len(tools)
    k_max = max(1, min(k_max, len(tools)))
    k_min = max(1, min(k_min, k_max))

    rng = random.Random(seed)

    # Choose sampling function based on strategy
    if strategy == "param_aware":
        chooser = sample_param_aware_subset
    elif strategy == "semantic":
        chooser = sample_semantic_subset
    else:
        chooser = sample_random_subset

    subsets: List[List[ToolSpec]] = []
    for i in range(total):
        if fixed_k is not None:
            k = fixed_k
        else:
            k = rng.randint(k_min, k_max)
        subset_seed = None if seed is None else (seed + i)
        subsets.append(chooser(tools, k=k, seed=subset_seed))
    return subsets
