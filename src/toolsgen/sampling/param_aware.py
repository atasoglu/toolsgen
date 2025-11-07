from __future__ import annotations

import random
from typing import List, Sequence

from ..schema import ToolSpec


def tool_param_count(tool: ToolSpec) -> int:
    params = tool.function.parameters or {}
    props = params.get("properties") if isinstance(params, dict) else None
    if isinstance(props, dict):
        return len(props)
    return 0


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
    scored = [(t, tool_param_count(t)) for t in tools]
    # Shuffle for tie-breaking
    rng.shuffle(scored)
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:k]]
