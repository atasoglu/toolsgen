"""Dataset writer for JSONL output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..schema import Record


def write_dataset_jsonl(
    records: List[Record],
    output_path: Path,
    split: Optional[str] = None,
) -> None:
    """Write records to a JSONL file.

    Args:
        records: List of Record objects to write.
        output_path: Path to output JSONL file.
        split: Optional split name (e.g., "train", "val") for filename.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            # Convert record to dict and serialize to JSON
            record_dict = record.model_dump(exclude_none=True)
            json_line = json.dumps(record_dict, ensure_ascii=False)
            f.write(json_line + "\n")
