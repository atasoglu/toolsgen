"""Core dataset generation logic."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

from ..judge import judge_tool_calls
from ..problem_generator import generate_problem
from ..tool_caller import generate_tool_calls
from ..sampling import batched_subsets
from ..schema import Manifest, Message, Record, ToolSpec
from .client import create_openai_client
from .config import GenerationConfig, ModelConfig, RoleBasedModelConfig
from .io import append_record_jsonl, load_tool_specs, write_dataset_jsonl


def _generate_sample(
    problem_client: OpenAI,
    caller_client: OpenAI,
    judge_client: OpenAI,
    record_id: str,
    tools: List[ToolSpec],
    role_config: RoleBasedModelConfig,
    language: str = "english",
) -> Optional[Record]:
    """Generate a complete sample (request + tool calls + record).

    Args:
        problem_client: OpenAI client for problem generation.
        caller_client: OpenAI client for tool calling.
        judge_client: OpenAI client for judging.
        record_id: Unique identifier for the record.
        tools: Available tools for this sample.
        role_config: Role-based model configuration.
        language: Language name for user requests.

    Returns:
        Record object or None if generation fails.
    """
    # 1. Generate user request
    user_request = generate_problem(
        client=problem_client,
        model=role_config.problem_generator.model,
        tools=tools,
        language=language,
        temperature=role_config.problem_generator.temperature,
        max_tokens=200,
    )

    if not user_request:
        return None

    # 2. Generate tool calls
    tool_calls = generate_tool_calls(
        client=caller_client,
        model=role_config.tool_caller.model,
        user_request=user_request,
        tools=tools,
        temperature=role_config.tool_caller.temperature,
        max_tokens=500,
    )

    if not tool_calls:
        return None

    # 3. Judge the tool calls
    judge_dict: Dict[str, Any] = {
        "model": role_config.judge.model,
        "temperature": role_config.judge.temperature,
    }
    try:
        judge_result = judge_tool_calls(
            client=judge_client,
            model=role_config.judge.model,
            user_request=user_request,
            tools=tools,
            tool_calls=tool_calls,
            temperature=role_config.judge.temperature,
        )
        judge_dict.update(judge_result.to_dict())
    except Exception:
        pass  # Continue without judge data

    # 4. Create record
    return Record(
        id=record_id,
        language=language,
        tools=tools,
        messages=[Message(role="user", content=user_request)],
        assistant_calls=tool_calls,
        problem_metadata={"generated": True, "user_request": user_request},
        judge=judge_dict,
        tools_metadata={"num_tools": len(tools)},
    )


def generate_dataset(
    output_dir: Path,
    gen_config: GenerationConfig,
    model_config: ModelConfig | RoleBasedModelConfig,
    tools_path: Optional[Path] = None,
    tools: Optional[List[ToolSpec]] = None,
) -> Dict[str, Any]:
    """Generate a tool-calling dataset from tool specifications.

    Args:
        output_dir: Directory to write dataset files.
        gen_config: Generation configuration.
        model_config: Model configuration (single or role-based).
        tools_path: Path to tools.json file (optional if tools is provided).
        tools: List of tool specifications (optional if tools_path is provided).

    Returns:
        Dictionary containing generation statistics.

    Raises:
        ValueError: If neither tools_path nor tools is provided.
    """
    # Load or use provided tool specs
    if tools is not None:
        all_tools = tools
    elif tools_path is not None:
        all_tools = load_tool_specs(tools_path)
    else:
        raise ValueError("Either tools_path or tools must be provided")

    # Convert to role-based config if needed
    if isinstance(model_config, ModelConfig):
        role_config = RoleBasedModelConfig.from_single_config(model_config)
    else:
        role_config = model_config

    # Create clients for each role
    problem_client = create_openai_client(role_config.problem_generator)
    caller_client = create_openai_client(role_config.tool_caller)
    judge_client = create_openai_client(role_config.judge)

    # Sample tool subsets
    strategy = gen_config.strategy
    if strategy not in ("random", "param_aware", "semantic"):
        sampling_strategy = "random"
    else:
        sampling_strategy = strategy

    tool_subsets = batched_subsets(
        all_tools,
        total=gen_config.num_samples,
        strategy=sampling_strategy,
        seed=gen_config.seed,
        k_min=gen_config.k_min,
        k_max=gen_config.k_max,
        batch_size=gen_config.batch_size,
        shuffle=gen_config.shuffle_tools,
    )

    # Generate records
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "train.jsonl"

    # Clear existing file
    if jsonl_path.exists():
        jsonl_path.unlink()

    all_records: List[Record] = []
    failed = 0

    with tqdm(
        total=gen_config.num_samples, desc="Generating samples", unit="sample"
    ) as pbar:
        for i in range(gen_config.num_samples):
            tools_subset = tool_subsets[i % len(tool_subsets)]
            for attempt in range(gen_config.max_attempts):
                try:
                    record_id = f"record_{len(all_records):06d}"
                    record = _generate_sample(
                        problem_client,
                        caller_client,
                        judge_client,
                        record_id,
                        tools_subset,
                        role_config,
                        gen_config.language,
                    )

                    if record:
                        all_records.append(record)
                        append_record_jsonl(record, jsonl_path)
                        pbar.update(1)
                        break
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    tqdm.write(f"Warning: Attempt {attempt + 1} failed: {e}")
            else:
                tqdm.write(
                    f"Warning: Failed to generate sample {i} after {gen_config.max_attempts} attempts"
                )

    # Split records into train/val if configured
    splits: Dict[str, List[Record]] = {}
    if gen_config.train_split < 1.0 and len(all_records) > 0:
        # Shuffle for deterministic split
        rng = random.Random(gen_config.seed)
        shuffled = all_records.copy()
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * gen_config.train_split)
        splits["train"] = shuffled[:split_idx]
        splits["val"] = shuffled[split_idx:]

        # Rewrite files with split data
        for split_name, records in splits.items():
            if records:
                split_path = output_dir / f"{split_name}.jsonl"
                write_dataset_jsonl(records, split_path)
    else:
        splits["train"] = all_records

    # Create manifest
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
