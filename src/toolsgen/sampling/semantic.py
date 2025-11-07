from __future__ import annotations

import random
import re
from typing import List, Sequence

from ..schema import ToolSpec


def extract_keywords(text: str) -> set[str]:
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


def tool_semantic_similarity(tool1: ToolSpec, tool2: ToolSpec) -> float:
    """Calculate semantic similarity between two tools using Jaccard similarity.

    Args:
        tool1: First tool specification.
        tool2: Second tool specification.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    keywords1 = extract_keywords(tool1.function.name) | extract_keywords(
        tool1.function.description or ""
    )
    keywords2 = extract_keywords(tool2.function.name) | extract_keywords(
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
            avg_sim = sum(tool_semantic_similarity(tool, s) for s in selected) / len(
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
