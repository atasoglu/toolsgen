"""Multiprocessing-based record generation engine."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..schema import Record, ToolSpec
from .config import GenerationConfig, RoleBasedModelConfig
from .io import append_record_jsonl
from .record_builder import RecordBuilder


@dataclass
class WorkerSampleSpec:
    sample_index: int
    tools: List[Dict[str, Any]]


@dataclass
class WorkerTask:
    batch_id: int
    samples: List[WorkerSampleSpec]


@dataclass
class WorkerSampleResult:
    sample_index: int
    record: Optional[Dict[str, Any]]
    failed_attempts: int
    error: Optional[str] = None


@dataclass
class WorkerBatchResult:
    batch_id: int
    sample_results: List[WorkerSampleResult]


_WORKER_BUILDER: Optional[RecordBuilder] = None
_WORKER_MAX_ATTEMPTS: int = 3


def _init_worker(
    role_config: RoleBasedModelConfig, language: str, max_attempts: int
) -> None:
    global _WORKER_BUILDER, _WORKER_MAX_ATTEMPTS

    _WORKER_BUILDER = RecordBuilder(role_config=role_config, language=language)
    _WORKER_BUILDER.ensure_clients()
    _WORKER_MAX_ATTEMPTS = max_attempts


def _worker_generate_sample(sample_spec: WorkerSampleSpec) -> WorkerSampleResult:
    if _WORKER_BUILDER is None:
        return WorkerSampleResult(
            sample_index=sample_spec.sample_index,
            record=None,
            failed_attempts=_WORKER_MAX_ATTEMPTS,
            error="Worker builder not initialized",
        )

    tools = [ToolSpec.model_validate(tool) for tool in sample_spec.tools]
    failed_attempts = 0
    last_error: Optional[str] = None

    for _ in range(_WORKER_MAX_ATTEMPTS):
        try:
            record = _WORKER_BUILDER.generate_record(
                record_id=f"record_worker_{sample_spec.sample_index:06d}",
                tools=tools,
            )

            if record:
                return WorkerSampleResult(
                    sample_index=sample_spec.sample_index,
                    record=record.model_dump(mode="python"),
                    failed_attempts=failed_attempts,
                )

            failed_attempts += 1
        except Exception as exc:
            failed_attempts += 1
            last_error = str(exc)

    return WorkerSampleResult(
        sample_index=sample_spec.sample_index,
        record=None,
        failed_attempts=failed_attempts,
        error=last_error or "Failed to generate sample",
    )


def _worker_generate_batch(task: WorkerTask) -> WorkerBatchResult:
    results = [_worker_generate_sample(spec) for spec in task.samples]
    return WorkerBatchResult(batch_id=task.batch_id, sample_results=results)


def _build_worker_tasks(
    tool_subsets: List[List[ToolSpec]], gen_config: GenerationConfig
) -> List[WorkerTask]:
    sample_specs: List[WorkerSampleSpec] = []
    for sample_index in range(gen_config.num_samples):
        tools_subset = tool_subsets[sample_index % len(tool_subsets)]
        tools_payload = [tool.model_dump(mode="python") for tool in tools_subset]
        sample_specs.append(
            WorkerSampleSpec(sample_index=sample_index, tools=tools_payload)
        )

    batch_size = max(1, gen_config.worker_batch_size)
    tasks: List[WorkerTask] = []
    for batch_id, start in enumerate(range(0, len(sample_specs), batch_size)):
        batch_samples = sample_specs[start : start + batch_size]
        tasks.append(WorkerTask(batch_id=batch_id, samples=batch_samples))

    return tasks


def generate_records_parallel(
    tool_subsets: List[List[ToolSpec]],
    role_config: RoleBasedModelConfig,
    gen_config: GenerationConfig,
    jsonl_path: Path,
) -> Tuple[List[Record], int]:
    tasks = _build_worker_tasks(tool_subsets, gen_config)
    if not tasks:
        return [], 0

    results_by_index: Dict[int, Record] = {}
    failed_indices: set[int] = set()
    written_records: List[Record] = []
    failed = 0
    next_id_to_write = 0

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=gen_config.num_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(role_config, gen_config.language, gen_config.max_attempts),
    ) as executor:
        future_to_task = {
            executor.submit(_worker_generate_batch, task): task for task in tasks
        }

        with tqdm(
            total=gen_config.num_samples, desc="Generating samples", unit="sample"
        ) as pbar:
            for future in as_completed(future_to_task):
                batch_result = future.result()
                for sample_result in batch_result.sample_results:
                    failed += sample_result.failed_attempts

                    if sample_result.record:
                        record = Record.model_validate(sample_result.record)
                        results_by_index[sample_result.sample_index] = record
                    else:
                        failed_indices.add(sample_result.sample_index)
                        tqdm.write(
                            "Warning: Failed to generate sample "
                            f"{sample_result.sample_index} after {gen_config.max_attempts} attempts"
                            + (
                                f" ({sample_result.error})"
                                if sample_result.error
                                else ""
                            )
                        )

                    while (
                        next_id_to_write in results_by_index
                        or next_id_to_write in failed_indices
                    ):
                        if next_id_to_write in results_by_index:
                            rec = results_by_index[next_id_to_write]
                            rec.id = f"record_{next_id_to_write:06d}"
                            append_record_jsonl(rec, jsonl_path)
                            written_records.append(rec)
                            del results_by_index[next_id_to_write]
                        next_id_to_write += 1

                    pbar.update(1)

    return written_records, failed
