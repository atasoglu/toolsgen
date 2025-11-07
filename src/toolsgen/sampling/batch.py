import random
from typing import List, Optional, Sequence

from .random import sample_random_subset
from .param_aware import sample_param_aware_subset
from .semantic import sample_semantic_subset
from ..schema import ToolSpec


def batched_subsets(
    tools: Sequence[ToolSpec],
    *,
    batch_size: int,
    total: int,
    strategy: str = "random",
    seed: Optional[int] = None,
    k_min: int = 1,
    k_max: Optional[int] = None,
) -> List[List[ToolSpec]]:
    """Produce multiWple subsets according to a strategy.

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
