"""I/O operations for dataset generation (loading and writing)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..schema import Record, ToolSpec


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


def write_dataset_jsonl(
    records: List[Record],
    output_path: Path,
) -> None:
    """Write records to a JSONL file.

    Args:
        records: List of Record objects to write.
        output_path: Path to output JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            record_dict = record.model_dump(exclude_none=True)
            json_line = json.dumps(record_dict, ensure_ascii=False)
            f.write(json_line + "\n")


def append_record_jsonl(record: Record, output_path: Path) -> None:
    """Append a single record to a JSONL file.

    Args:
        record: Record object to append.
        output_path: Path to output JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as f:
        record_dict = record.model_dump(exclude_none=True)
        json_line = json.dumps(record_dict, ensure_ascii=False)
        f.write(json_line + "\n")
