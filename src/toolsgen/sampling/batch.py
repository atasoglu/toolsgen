import random
from typing import List, Optional, Sequence

from .random import sample_random_subset
from .param_aware import sample_param_aware_subset
from .semantic import sample_semantic_subset
from ..schema import ToolSpec


def batched_subsets(
    tools: Sequence[ToolSpec],
    *,
    total: int,
    strategy: str = "random",
    seed: Optional[int] = None,
    k_min: int = 1,
    k_max: Optional[int] = None,
) -> List[List[ToolSpec]]:
    """Produce multiple subsets according to a strategy.

    Notes:
    - `total` determines how many subsets to produce.
    - `k_min` and `k_max` determine the range of subset sizes.
    - If `k_min == k_max`, all subsets will have the same size.
    - `strategy`: "random", "param_aware", or "semantic".
    """

    if not tools:
        return []

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
        k = rng.randint(k_min, k_max)
        subset_seed = None if seed is None else (seed + i)
        subsets.append(chooser(tools, k=k, seed=subset_seed))
    return subsets
