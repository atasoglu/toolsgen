"""Multi-model example - Using different models for different roles"""

from pathlib import Path
from dotenv import load_dotenv
from toolsgen.config import GenerationConfig, ModelConfig, RoleBasedModelConfig
from toolsgen.generator import generate_dataset

load_dotenv()

tools_path = Path(__file__).parent / "tools.json"
output_dir = Path(__file__).parent / "output"

gen_config = GenerationConfig(
    num_samples=10,
    strategy="random",
    seed=42,
    max_attempts=3,
)

# Different models for different roles
role_config = RoleBasedModelConfig(
    problem_generator=ModelConfig(
        model="gpt-4o-mini",
        temperature=0.9,  # More creative problems
    ),
    tool_caller=ModelConfig(
        model="gpt-4o",
        temperature=0.3,  # More consistent tool calls
    ),
    judge=ModelConfig(
        model="gpt-4o",
        temperature=0.0,  # Deterministic evaluation
    ),
)

manifest = generate_dataset(tools_path, output_dir, gen_config, role_config)

print(f"\n✓ Generated {manifest['num_generated']}/{manifest['num_requested']} records")
if manifest["num_failed"] > 0:
    print(f"  Failed: {manifest['num_failed']} attempts")
print(f"  Problem Generator: {role_config.problem_generator.model}")
print(f"  Tool Caller: {role_config.tool_caller.model}")
print(f"  Judge: {role_config.judge.model}")
print(f"  Output: {output_dir}")
