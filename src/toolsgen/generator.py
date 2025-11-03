from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .config import GenerationConfig, ModelConfig, RoleBasedModelConfig
from .io.writer import append_record_jsonl, write_dataset_jsonl
from .judge.scorer import judge_tool_calls
from .prompts import (
    create_caller_system_prompt,
    create_problem_generation_prompt,
    create_problem_generation_user_message,
)
from .providers.openai_compat import ChatModelClient
from .sampling import batched_subsets
from .schema import AssistantToolCall, Message, Record, ToolSpec


def load_tool_specs(tools_path: Path) -> List[ToolSpec]:
    """Load tool specifications from a JSON file.

    Args:
        tools_path: Path to JSON file containing OpenAI-compatible tool definitions.

    Returns:
        List of validated ToolSpec objects.

    Raises:
        ValueError: If the JSON structure is invalid.
    """
    with tools_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("tools.json must contain a list of tool definitions")

    return [ToolSpec.model_validate(tool) for tool in data]


def _generate_sample(
    problem_client: ChatModelClient,
    caller_client: ChatModelClient,
    judge_client: ChatModelClient,
    record_id: str,
    tools: List[ToolSpec],
    role_config: RoleBasedModelConfig,
    language: str = "english",
) -> Optional[Record]:
    """Generate a complete sample (request + tool calls + record).

    Args:
        problem_client: LLM client for problem generation.
        caller_client: LLM client for tool calling.
        judge_client: LLM client for judging.
        record_id: Unique identifier for the record.
        tools: Available tools for this sample.
        role_config: Role-based model configuration.
        language: Language name for user requests.

    Returns:
        Record object or None if generation fails.
    """
    # 1. Generate user request
    prompt = create_problem_generation_prompt(tools, language)
    response = problem_client.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": create_problem_generation_user_message()},
        ],
        temperature=role_config.problem_generator.temperature,
        max_tokens=200,
    )

    choices = response.get("choices", [])
    if not choices:
        return None
    user_request = choices[0].get("message", {}).get("content", "").strip()
    if not user_request:
        return None

    # 2. Generate tool calls
    tools_dict = [tool.model_dump() for tool in tools]
    response = caller_client.create(
        messages=[
            {"role": "system", "content": create_caller_system_prompt()},
            {"role": "user", "content": user_request},
        ],
        tools=tools_dict,
        tool_choice="auto",
        temperature=role_config.tool_caller.temperature,
        max_tokens=500,
    )

    # 3. Extract tool calls
    message = response.get("choices", [{}])[0].get("message", {})
    tool_calls_data = message.get("tool_calls", [])
    tool_calls = []
    for tc_data in tool_calls_data:
        try:
            tool_calls.append(AssistantToolCall.model_validate(tc_data))
        except Exception:
            continue

    if not tool_calls:
        return None

    # 4. Judge the tool calls
    judge_dict: Dict[str, Any] = {
        "model": role_config.judge.model,
        "temperature": role_config.judge.temperature,
    }
    try:
        judge_result = judge_tool_calls(
            client=judge_client,
            user_request=user_request,
            tools=tools,
            tool_calls=tool_calls,
            temperature=role_config.judge.temperature,
        )
        judge_dict.update(judge_result.to_dict())
    except Exception:
        pass  # Continue without judge data

    # 5. Create record
    return Record(
        id=record_id,
        language=language,
        tools=tools,
        messages=[Message(role="user", content=user_request)],
        assistant_calls=tool_calls,
        problem_metadata={"generated": True, "user_request": user_request},
        judge=judge_dict,
    )


def generate_dataset(
    tools_path: Path,
    output_dir: Path,
    gen_config: GenerationConfig,
    model_config: ModelConfig | RoleBasedModelConfig,
) -> Dict[str, Any]:
    """Generate a tool-calling dataset from tool specifications.

    Args:
        tools_path: Path to tools.json file.
        output_dir: Directory to write dataset files.
        gen_config: Generation configuration.
        model_config: Model configuration (single or role-based).

    Returns:
        Dictionary containing generation statistics.
    """
    # Load tool specs
    all_tools = load_tool_specs(tools_path)

    # Convert to role-based config if needed
    if isinstance(model_config, ModelConfig):
        role_config = RoleBasedModelConfig.from_single_config(model_config)
    else:
        role_config = model_config

    # Create clients for each role
    problem_client = ChatModelClient(
        model=role_config.problem_generator.model,
        base_url=role_config.problem_generator.base_url,
    )
    caller_client = ChatModelClient(
        model=role_config.tool_caller.model,
        base_url=role_config.tool_caller.base_url,
    )
    judge_client = ChatModelClient(
        model=role_config.judge.model,
        base_url=role_config.judge.base_url,
    )

    # Sample tool subsets
    strategy = gen_config.strategy
    if strategy not in ("random", "param_aware", "semantic"):
        sampling_strategy = "random"
    else:
        sampling_strategy = strategy

    # Determine batch size (k) - use average of 2 tools per sample if not specified
    batch_size = max(1, min(3, len(all_tools))) if len(all_tools) > 0 else 1

    tool_subsets = batched_subsets(
        all_tools,
        batch_size=batch_size,
        total=gen_config.num_samples,
        strategy=sampling_strategy,
        seed=gen_config.seed,
    )

    # Generate records
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "train.jsonl"

    # Clear existing file
    if jsonl_path.exists():
        jsonl_path.unlink()

    all_records: List[Record] = []
    failed = 0

    for i, tools_subset in enumerate(tqdm(tool_subsets, desc="Generating samples")):
        try:
            record_id = f"record_{i:06d}"
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
            else:
                failed += 1
        except Exception:
            failed += 1

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
                write_dataset_jsonl(records, split_path, split=split_name)
    else:
        splits["train"] = all_records

    # Create manifest
    manifest = {
        "version": "0.1.0",
        "num_requested": gen_config.num_samples,
        "num_generated": len(all_records),
        "num_failed": failed,
        "strategy": gen_config.strategy,
        "seed": gen_config.seed,
        "train_split": gen_config.train_split,
        "tools_count": len(all_tools),
        "models": {
            "problem_generator": role_config.problem_generator.model,
            "tool_caller": role_config.tool_caller.model,
            "judge": role_config.judge.model,
        },
        "splits": {name: len(records) for name, records in splits.items()},
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return manifest
