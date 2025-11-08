"""
Hugging Face example - Using a dataset from Hugging Face

Install datasets library by using `pip install datasets` and set the dataset id.
"""

from pathlib import Path
from dotenv import load_dotenv
from utils import dataset_to_tools
from config import gen_config, role_config
from toolsgen import generate_dataset

# Load environment variables from .env file
load_dotenv()

# Load dataset from Hugging Face
dataset_id = "argilla-warehouse/python-seed-tools"
tools = dataset_to_tools(dataset_id, dataset_kwargs={"split": "train"})
output_dir = Path(__file__).parent / "output"

# Generate dataset
manifest = generate_dataset(output_dir, gen_config, role_config, tools=tools)

# Print summary
print(f"\n✓ Generated {manifest['num_generated']}/{manifest['num_requested']} records")
if manifest["num_failed"] > 0:
    print(f"  Failed: {manifest['num_failed']} attempts")
print(f"  Problem Generator: {role_config.problem_generator.model}")
print(f"  Tool Caller: {role_config.tool_caller.model}")
print(f"  Judge: {role_config.judge.model}")
print(f"  Output: {output_dir}")
