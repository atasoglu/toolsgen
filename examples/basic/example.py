"""Basic Python example - ToolsGen usage"""

from pathlib import Path

from toolsgen import GenerationConfig, ModelConfig, generate_dataset

# NOTE: Set OPENAI_API_KEY environment variable before running
# Example: export OPENAI_API_KEY="your-api-key-here"

# Configuration
tools_path = Path(__file__).parent / "tools.json"
output_dir = Path(__file__).parent / "output"

gen_config = GenerationConfig(
    num_samples=5,
    strategy="random",
    seed=42,
    train_split=0.8,
)

model_config = ModelConfig(
    model="gpt-4o-mini",
    temperature=0.7,
)

# Generate dataset
manifest = generate_dataset(tools_path, output_dir, gen_config, model_config)

print(f"✓ Generated {manifest['num_generated']} records")
print(f"  Output: {output_dir}")
