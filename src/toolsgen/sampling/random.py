from __future__ import annotations

import random
from typing import List, Sequence

from ..schema import ToolSpec


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
