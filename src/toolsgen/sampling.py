from __future__ import annotations

import random
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
    - `strategy`: "random" or "param_aware".
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
    chooser = (
        sample_random_subset if strategy == "random" else sample_param_aware_subset
    )

    subsets: List[List[ToolSpec]] = []
    for i in range(total):
        if fixed_k is not None:
            k = fixed_k
        else:
            k = rng.randint(k_min, k_max)
        subset_seed = None if seed is None else (seed + i)
        subsets.append(chooser(tools, k=k, seed=subset_seed))
    return subsets
