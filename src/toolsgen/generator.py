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


def _generate_user_request(
    client: ChatModelClient, tools: List[ToolSpec], temperature: float = 0.8
) -> str:
    """Generate a natural language user request using the LLM.

    Args:
        client: LLM client for generation.
        tools: Available tools for the request.
        temperature: Sampling temperature.

    Returns:
        Generated user request text.
    """
    prompt = create_problem_generation_prompt(tools)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": create_problem_generation_user_message()},
    ]

    response = client.create(
        messages=messages,
        temperature=temperature,
        max_tokens=200,
    )

    choices = response.get("choices", [])
    if not choices:
        raise ValueError("LLM returned no choices")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError("LLM returned empty content")
    return content.strip()


def _generate_tool_calls(
    client: ChatModelClient,
    user_request: str,
    tools: List[ToolSpec],
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Generate tool calls for a user request using the LLM.

    Args:
        client: LLM client for generation.
        user_request: The user's natural language request.
        tools: Available tools that can be called.
        temperature: Sampling temperature (lower for more deterministic).

    Returns:
        API response containing tool calls.
    """
    tools_dict = [tool.model_dump() for tool in tools]
    messages = [
        {"role": "system", "content": create_caller_system_prompt()},
        {"role": "user", "content": user_request},
    ]

    response = client.create(
        messages=messages,
        tools=tools_dict,
        tool_choice="auto",
        temperature=temperature,
        max_tokens=500,
    )

    return response


def _extract_tool_calls(response: Dict[str, Any]) -> List[AssistantToolCall]:
    """Extract tool calls from LLM response.

    Args:
        response: API response dictionary.

    Returns:
        List of AssistantToolCall objects.
    """
    choices = response.get("choices", [])
    if not choices:
        return []

    message = choices[0].get("message", {})
    tool_calls_data = message.get("tool_calls", [])

    tool_calls = []
    for tc_data in tool_calls_data:
        try:
            tool_call = AssistantToolCall.model_validate(tc_data)
            tool_calls.append(tool_call)
        except Exception:
            continue

    return tool_calls


def _create_record(
    record_id: str,
    tools: List[ToolSpec],
    user_request: str,
    tool_calls: List[AssistantToolCall],
    model_info: Dict[str, Any],
    judge_result: Optional[Dict[str, Any]] = None,
) -> Record:
    """Create a Record object from generated data.

    Args:
        record_id: Unique identifier for the record.
        tools: Tools available in this record.
        user_request: Generated user request.
        tool_calls: Generated tool calls.
        model_info: Information about the model used.
        judge_result: Optional judge scoring result.

    Returns:
        Record object.
    """
    messages = [
        Message(role="user", content=user_request),
    ]

    judge_dict: Dict[str, Any] = {
        "model": model_info.get("model", "unknown"),
        "temperature": model_info.get("temperature", 0.7),
    }

    if judge_result:
        judge_dict.update(judge_result)

    return Record(
        id=record_id,
        language="en",
        tools=tools,
        messages=messages,
        assistant_calls=tool_calls,
        problem_metadata={
            "generated": True,
            "user_request": user_request,
        },
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
            # Generate user request
            user_request = _generate_user_request(
                client, tools_subset, temperature=model_config.temperature
            )

            # Generate tool calls
            response = _generate_tool_calls(
                client, user_request, tools_subset, temperature=model_config.temperature
            )

            # Extract tool calls
            tool_calls = _extract_tool_calls(response)

            # Only keep records with at least one tool call
            if not tool_calls:
                failed += 1
                continue

            # Judge the tool calls
            try:
                judge_result = judge_tool_calls(
                    client=client,
                    user_request=user_request,
                    tools=tools_subset,
                    tool_calls=tool_calls,
                    temperature=model_config.temperature,
                )
                judge_dict = judge_result.to_dict()
            except Exception as e:
                # If judging fails, continue without judge data
                import sys

                print(
                    f"Warning: Judge failed for record {i}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                judge_dict = None

            # Create record
            record_id = f"record_{i:06d}"
            record = _create_record(
                record_id=record_id,
                tools=tools_subset,
                user_request=user_request,
                tool_calls=tool_calls,
                model_info={
                    "model": model_config.model,
                    "temperature": model_config.temperature,
                },
                judge_result=judge_dict,
            )

            all_records.append(record)

        except Exception as e:
            failed += 1
            # Log the error for debugging
            import sys

            print(
                f"Warning: Failed to generate record {i}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            continue

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
