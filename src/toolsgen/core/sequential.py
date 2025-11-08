"""Sequential record generation engine."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from ..schema import Record, ToolSpec
from .config import GenerationConfig, RoleBasedModelConfig
from .io import append_record_jsonl
from .record_builder import RecordBuilder


def generate_records_sequential(
    tool_subsets: List[List[ToolSpec]],
    role_config: RoleBasedModelConfig,
    gen_config: GenerationConfig,
    jsonl_path: Path,
) -> Tuple[List[Record], int]:
    """Iteratively produce records in a single process."""

    builder = RecordBuilder(role_config=role_config, language=gen_config.language)
    builder.ensure_clients()

    all_records: List[Record] = []
    failed = 0

    with tqdm(
        total=gen_config.num_samples, desc="Generating samples", unit="sample"
    ) as pbar:
        for sample_index in range(gen_config.num_samples):
            tools_subset = tool_subsets[sample_index % len(tool_subsets)]
            for attempt in range(gen_config.max_attempts):
                try:
                    record_id = f"record_{len(all_records):06d}"
                    record = builder.generate_record(record_id, tools_subset)

                    if record:
                        all_records.append(record)
                        append_record_jsonl(record, jsonl_path)
                        pbar.update(1)
                        break

                    failed += 1
                except Exception as exc:
                    failed += 1
                    tqdm.write(f"Warning: Attempt {attempt + 1} failed: {exc}")
            else:
                tqdm.write(
                    f"Warning: Failed to generate sample {sample_index} after {gen_config.max_attempts} attempts"
                )

    return all_records, failed
