import json

from .schema import ToolFunction
from datasets import load_dataset
from typing import Generator, Optional
from multiprocessing import Pool, cpu_count
from toolsgen import ToolSpec


def _process_sample(sample_data: tuple) -> tuple[list[dict], int]:
    """Process a single sample and extract valid tools."""
    sample, column_id, debug = sample_data
    tools_list = []
    failed_count = 0
    try:
        tools = json.loads(sample[column_id])
        if not isinstance(tools, list):
            tools = [tools]
        for tool in tools:
            try:
                tool_definition = json.dumps(tool)
                model = ToolFunction.model_validate_json(tool_definition)
                tools_list.append(model.model_dump())
            except Exception as e:
                failed_count += 1
                if debug:
                    print(f"Skipping invalid tool definition: {e}")
                    print(f"Tool definition: {json.dumps(tool, indent=2)}")
                continue
    except Exception as e:
        failed_count += 1
        if debug:
            print(f"Skipping invalid sample: {e}")
    return tools_list, failed_count


def stream_tools_from_datasets(
    dataset_ids: list[str] | str,
    split: str = "train",
    column_id: str = "tools",
    debug: bool = False,
    num_workers: Optional[int] = None,
    batch_size: int = 100,
) -> Generator[dict, None, None]:
    """Stream unique tool definitions from multiple datasets with parallel processing.

    **Note:** The datasets are expected to contain tool definitions in JSON format
    within the specified column. Otherwise, those entries will be skipped.

    Args:
        dataset_ids (list[str] | str): The dataset identifier(s). Can be a single string or list of strings.
        split (str, optional): The dataset split to load. Defaults to "train".
        column_id (str, optional): The column containing tool definitions. Defaults to "tools".
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        num_workers (Optional[int], optional): Number of worker processes. Defaults to CPU count.
        batch_size (int, optional): Batch size for parallel processing. Defaults to 100.

    Yields:
        Generator[dict, None, None]: Tool definitions as dictionaries.
    """
    total_failed = 0
    if num_workers is None:
        num_workers = cpu_count()

    # Convert single dataset_id to list for uniform processing
    if isinstance(dataset_ids, str):
        dataset_ids = [dataset_ids]

    seen_tools = set()

    for dataset_id in dataset_ids:
        dataset = load_dataset(dataset_id, split=split, streaming=True)

        if num_workers == 1:
            # Sequential processing
            for sample in dataset:
                tools_list, failed_count = _process_sample((sample, column_id, debug))
                total_failed += failed_count
                for tool in tools_list:
                    tool_name = tool.get("function", {}).get("name")
                    if tool_name and tool_name not in seen_tools:
                        seen_tools.add(tool_name)
                        yield tool
        else:
            # Parallel processing
            batch = []
            with Pool(num_workers) as pool:
                for sample in dataset:
                    batch.append((sample, column_id, debug))
                    if len(batch) >= batch_size:
                        results = pool.map(_process_sample, batch)
                        for tools_list, failed_count in results:
                            total_failed += failed_count
                            for tool in tools_list:
                                tool_name = tool.get("function", {}).get("name")
                                if tool_name and tool_name not in seen_tools:
                                    seen_tools.add(tool_name)
                                    yield tool
                        batch = []

                # Process remaining batch
                if batch:
                    results = pool.map(_process_sample, batch)
                    for tools_list, failed_count in results:
                        total_failed += failed_count
                        for tool in tools_list:
                            tool_name = tool.get("function", {}).get("name")
                            if tool_name and tool_name not in seen_tools:
                                seen_tools.add(tool_name)
                                yield tool

    # After processing each dataset, print the number of failed samples/tools
    print(f"Total failed samples/tools: {total_failed}")


def save_tools_to_file(
    tools: Generator[dict, None, None],
    output_file: str,
) -> None:
    """Save tool definitions to a JSONL file.

    Args:
        tools (Generator[dict, None, None]): Generator of tool definitions.
        output_file (str): Path to the output JSONL file.
    """
    with open(output_file, "a", encoding="utf-8") as f:
        for tool in tools:
            f.write(json.dumps(tool, ensure_ascii=False) + "\n")


def load_tools_from_file(
    input_file: str,
) -> Generator[ToolSpec, None, None]:
    """Load tool definitions from a JSONL file.

    Args:
        input_file (str): Path to the input JSONL file.
    Yields:
        Generator[ToolSpec, None, None]: Generator of ToolSpec instances.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            tool_dict = json.loads(line)
            tool_spec = ToolSpec.model_validate(tool_dict)
            yield tool_spec
