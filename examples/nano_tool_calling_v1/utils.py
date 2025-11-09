import json
from typing import List, Optional

from datasets import load_dataset

from toolsgen import (
    ToolFunction,
    ToolSpec,
)

from validation import validate_json_schema


def dataset_to_tools(
    dataset_id: str, dataset_kwargs: Optional[dict] = None
) -> List[ToolSpec]:
    """Load tools from a Hugging Face dataset.

    Args:
        dataset_id (str): The Hugging Face dataset identifier.
        dataset_kwargs (Optional[dict]): Additional arguments for loading the dataset.
    Returns:
        List[ToolSpec]: A list of ToolSpec objects.
    """
    dataset = load_dataset(dataset_id, **(dataset_kwargs or {}))
    # Each dataset row contains a list of tools in OpenAI format
    # Flatten the nested lists: [[tool1], [tool2, tool3], ...] -> [tool1, tool2, tool3, ...]
    all_tools = []
    for tools_json in dataset["tools"]:
        tools_list = json.loads(tools_json)
        all_tools.extend(tools_list)

    # Convert from OpenAI format to ToolSpec
    # OpenAI format: {'type': 'function', 'function': {'name': ..., 'description': ..., 'parameters': ...}}
    return [
        ToolSpec(
            function=ToolFunction(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            )
        )
        for tool in all_tools
        if validate_json_schema(tool)
    ]
