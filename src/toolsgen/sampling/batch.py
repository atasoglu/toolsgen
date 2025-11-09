import random
from typing import Callable, List, Optional, Sequence

from tqdm import tqdm

from .param_aware import sample_param_aware_subset
from .random import sample_random_subset
from .semantic import sample_semantic_subset
from ..schema import ToolSpec


def _resolve_sampler(
    strategy: str,
) -> Callable[[Sequence[ToolSpec], int, Optional[int]], List[ToolSpec]]:
    """Return the subset sampler associated with a strategy."""
    lookup = {
        "random": sample_random_subset,
        "param_aware": sample_param_aware_subset,
        "semantic": sample_semantic_subset,
    }
    return lookup.get(strategy, sample_random_subset)


def _chunk_tools(
    tools: Sequence[ToolSpec],
    *,
    batch_size: Optional[int],
    shuffle: bool,
    rng: random.Random,
) -> List[List[ToolSpec]]:
    """Split tools into deterministic batches."""
    if not tools:
        return []

    if batch_size is None or batch_size <= 0 or batch_size >= len(tools):
        arranged = list(tools)
        if shuffle:
            rng.shuffle(arranged)
        return [arranged]

    arranged = list(tools)
    if shuffle:
        rng.shuffle(arranged)
    batches = []
    for idx in range(0, len(arranged), batch_size):
        batches.append(arranged[idx : idx + batch_size])
    return batches


def batched_subsets(
    tools: Sequence[ToolSpec],
    *,
    total: int,
    strategy: str = "random",
    seed: Optional[int] = None,
    k_min: int = 1,
    k_max: Optional[int] = None,
    batch_size: Optional[int] = None,
    shuffle: bool = False,
) -> List[List[ToolSpec]]:
    """Produce multiple subsets according to a strategy.

    Notes:
    - `total` determines how many subsets to produce.
    - `k_min` and `k_max` determine the range of subset sizes.
    - If `k_min == k_max`, all subsets will have the same size.
    - `strategy`: "random", "param_aware", or "semantic".
    - `batch_size` and `shuffle` allow iterating tools in deterministic chunks.
    """

    if not tools:
        return []

    rng = random.Random(seed)

    batches = _chunk_tools(tools, batch_size=batch_size, shuffle=shuffle, rng=rng)
    if not batches:
        return []

    max_batch_size = max(len(batch) for batch in batches)
    k_max = k_max or max_batch_size
    k_max = max(1, min(k_max, max_batch_size))
    k_min = max(1, min(k_min, k_max))

    chooser = _resolve_sampler(strategy)
    using_chunks = batch_size is not None and batch_size > 0

    subsets: List[List[ToolSpec]] = []
    for i in tqdm(range(total), desc="Preparing tool subsets", total=total):
        batch = batches[i % len(batches)]
        if using_chunks:
            k = len(batch)
        else:
            local_k_max = max(1, min(k_max, len(batch)))
            local_k_min = max(1, min(k_min, local_k_max))
            k = rng.randint(local_k_min, local_k_max)
        subset_seed = None if seed is None else (seed + i)
        subsets.append(chooser(batch, k=k, seed=subset_seed))
    return subsets
