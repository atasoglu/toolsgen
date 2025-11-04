"""
Hugging Face example - Using a dataset from Hugging Face

Install datasets library by using `pip install datasets` and set the dataset id.
"""

import json
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset
from dotenv import load_dotenv
from validation import validate_json_schema

from toolsgen import (
    GenerationConfig,
    ModelConfig,
    RoleBasedModelConfig,
    ToolFunction,
    ToolSpec,
    generate_dataset,
)

# Load environment variables from .env file
load_dotenv()


def dataset_to_tools(
    dataset_id: str, dataset_kwargs: Optional[dict] = None
) -> List[ToolSpec]:
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


# Load dataset from Hugging Face
dataset_id = "argilla-warehouse/python-seed-tools"
tools = dataset_to_tools(dataset_id, dataset_kwargs={"split": "train"})
output_dir = Path(__file__).parent / "output"

gen_config = GenerationConfig(
    num_samples=20,
    strategy="random",
    seed=42,
    train_split=0.8,
    language="turkish",
    max_attempts=3,
    batch_size=3,
)

# Different models for different roles
role_config = RoleBasedModelConfig(
    problem_generator=ModelConfig(
        model="gpt-4.1-mini",
        temperature=0.9,  # More creative problems
    ),
    tool_caller=ModelConfig(
        model="gpt-4.1-mini",
        temperature=0.3,  # More consistent tool calls
    ),
    judge=ModelConfig(
        model="gpt-4.1",
        temperature=0.0,  # Deterministic evaluation
    ),
)

manifest = generate_dataset(output_dir, gen_config, role_config, tools=tools)

print(f"\n✓ Generated {manifest['num_generated']}/{manifest['num_requested']} records")
if manifest["num_failed"] > 0:
    print(f"  Failed: {manifest['num_failed']} attempts")
print(f"  Problem Generator: {role_config.problem_generator.model}")
print(f"  Tool Caller: {role_config.tool_caller.model}")
print(f"  Judge: {role_config.judge.model}")
print(f"  Output: {output_dir}")
