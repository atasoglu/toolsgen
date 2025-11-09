"""Public entry point for dataset generation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..sampling import batched_subsets
from ..schema import Manifest, Record, ToolSpec
from .config import GenerationConfig, ModelConfig, RoleBasedModelConfig
from .io import load_tool_specs, write_dataset_jsonl
from .parallel import generate_records_parallel
from .sequential import generate_records_sequential


def _resolve_role_config(
    model_config: ModelConfig | RoleBasedModelConfig,
) -> RoleBasedModelConfig:
    if isinstance(model_config, ModelConfig):
        return RoleBasedModelConfig.from_single_config(model_config)
    return model_config


def _prepare_tool_subsets(
    tools: List[ToolSpec], gen_config: GenerationConfig
) -> List[List[ToolSpec]]:
    strategy = gen_config.strategy
    sampling_strategy = (
        strategy if strategy in ("random", "param_aware", "semantic") else "random"
    )

    return batched_subsets(
        tools,
        total=gen_config.num_samples,
        strategy=sampling_strategy,
        seed=gen_config.seed,
        k_min=gen_config.k_min,
        k_max=gen_config.k_max,
        batch_size=gen_config.batch_size,
        shuffle=gen_config.shuffle_tools,
    )


def _split_records(
    records: List[Record],
    gen_config: GenerationConfig,
    output_dir: Path,
) -> Dict[str, List[Record]]:
    if gen_config.train_split < 1.0 and records:
        rng = random.Random(gen_config.seed)
        shuffled = records.copy()
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * gen_config.train_split)
        splits: Dict[str, List[Record]] = {
            "train": shuffled[:split_idx],
            "val": shuffled[split_idx:],
        }

        temp_train = output_dir / "train.jsonl"
        if temp_train.exists():
            temp_train.unlink()

        for split_name, split_records in splits.items():
            if split_records:
                split_path = output_dir / f"{split_name}.jsonl"
                write_dataset_jsonl(split_records, split_path)

        return splits

    return {"train": records}


def _write_manifest(
    output_dir: Path,
    gen_config: GenerationConfig,
    role_config: RoleBasedModelConfig,
    all_tools: List[ToolSpec],
    all_records: List[Record],
    failed: int,
    splits: Dict[str, List[Record]],
) -> Dict[str, Any]:
    manifest = Manifest(
        num_requested=gen_config.num_samples,
        num_generated=len(all_records),
        num_failed=failed,
        strategy=gen_config.strategy,
        seed=gen_config.seed,
        train_split=gen_config.train_split,
        tools_count=len(all_tools),
        models={
            "problem_generator": role_config.problem_generator.model,
            "tool_caller": role_config.tool_caller.model,
            "judge": role_config.judge.model,
        },
        splits={name: len(records) for name, records in splits.items()},
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    return manifest.model_dump()


def generate_dataset(
    output_dir: Path,
    gen_config: GenerationConfig,
    model_config: ModelConfig | RoleBasedModelConfig,
    tools_path: Optional[Path] = None,
    tools: Optional[List[ToolSpec]] = None,
) -> Dict[str, Any]:
    """Generate a tool-calling dataset from tool specifications."""

    if tools is not None:
        all_tools = tools
    elif tools_path is not None:
        all_tools = load_tool_specs(tools_path)
    else:
        raise ValueError("Either tools_path or tools must be provided")

    role_config = _resolve_role_config(model_config)
    tool_subsets = _prepare_tool_subsets(all_tools, gen_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "train.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    if gen_config.num_workers <= 1:
        all_records, failed = generate_records_sequential(
            tool_subsets, role_config, gen_config, jsonl_path
        )
    else:
        all_records, failed = generate_records_parallel(
            tool_subsets, role_config, gen_config, jsonl_path
        )

    splits = _split_records(all_records, gen_config, output_dir)
    return _write_manifest(
        output_dir,
        gen_config,
        role_config,
        all_tools,
        all_records,
        failed,
        splits,
    )
