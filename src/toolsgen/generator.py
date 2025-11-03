from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .config import GenerationConfig, ModelConfig
from .io.writer import write_dataset_jsonl
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
    client: ChatModelClient,
    record_id: str,
    tools: List[ToolSpec],
    model_config: ModelConfig,
) -> Optional[Record]:
    """Generate a complete sample (request + tool calls + record).

    Args:
        client: LLM client for generation.
        record_id: Unique identifier for the record.
        tools: Available tools for this sample.
        model_config: Model configuration.

    Returns:
        Record object or None if generation fails.
    """
    # 1. Generate user request
    prompt = create_problem_generation_prompt(tools)
    response = client.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": create_problem_generation_user_message()},
        ],
        temperature=model_config.temperature,
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
    response = client.create(
        messages=[
            {"role": "system", "content": create_caller_system_prompt()},
            {"role": "user", "content": user_request},
        ],
        tools=tools_dict,
        tool_choice="auto",
        temperature=model_config.temperature,
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
        "model": model_config.model,
        "temperature": model_config.temperature,
    }
    try:
        judge_result = judge_tool_calls(
            client=client,
            user_request=user_request,
            tools=tools,
            tool_calls=tool_calls,
            temperature=model_config.temperature,
        )
        judge_dict.update(judge_result.to_dict())
    except Exception:
        pass  # Continue without judge data

    # 5. Create record
    return Record(
        id=record_id,
        language="en",
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
    model_config: ModelConfig,
) -> Dict[str, Any]:
    """Generate a tool-calling dataset from tool specifications.

    Args:
        tools_path: Path to tools.json file.
        output_dir: Directory to write dataset files.
        gen_config: Generation configuration.
        model_config: Model configuration.

    Returns:
        Dictionary containing generation statistics.
    """
    # Load tool specs
    all_tools = load_tool_specs(tools_path)

    # Create client
    client = ChatModelClient(
        model=model_config.model,
        base_url=model_config.base_url,
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

    all_records: List[Record] = []
    failed = 0

    for i, tools_subset in enumerate(tqdm(tool_subsets, desc="Generating samples")):
        try:
            record_id = f"record_{i:06d}"
            record = _generate_sample(client, record_id, tools_subset, model_config)

            if record:
                all_records.append(record)
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
    else:
        splits["train"] = all_records

    # Write JSONL files for each split
    for split_name, records in splits.items():
        if records:
            jsonl_path = output_dir / f"{split_name}.jsonl"
            write_dataset_jsonl(records, jsonl_path, split=split_name)

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
        "model": model_config.model,
        "temperature": model_config.temperature,
        "splits": {name: len(records) for name, records in splits.items()},
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return manifest
