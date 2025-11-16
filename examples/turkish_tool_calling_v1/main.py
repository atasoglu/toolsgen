from pathlib import Path
from dotenv import load_dotenv
from preprocessing import load_tools_from_file
from config import gen_config, role_config
from toolsgen import generate_dataset
from uuid import uuid4

# Load environment variables from .env file
load_dotenv()


def main() -> None:
    # Load tools from file
    tools = list(load_tools_from_file("tools.jsonl"))
    print("Loaded tools from file.")
    print(f"Number of tools loaded: {len(tools)}")

    # Define output directory with timestamp
    output_dir = Path(__file__).parent / f"output_{uuid4().hex}"

    # Generate dataset
    manifest = generate_dataset(output_dir, gen_config, role_config, tools=tools)

    # Print summary
    print(
        f"\n✓ Generated {manifest['num_generated']}/{manifest['num_requested']} records"
    )
    if manifest["num_failed"] > 0:
        print(f"  Failed: {manifest['num_failed']} attempts")
    print(f"  Problem Generator: {role_config.problem_generator.model}")
    print(f"  Tool Caller: {role_config.tool_caller.model}")
    print(f"  Judge: {role_config.judge.model}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
