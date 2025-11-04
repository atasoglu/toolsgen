"""Basic Python example - ToolsGen usage"""

from pathlib import Path

from dotenv import load_dotenv

from toolsgen import GenerationConfig, ModelConfig, generate_dataset

# Load environment variables from .env file
load_dotenv()

# NOTE: Set OPENAI_API_KEY environment variable before running
# You can either:
# 1. Create a .env file with: OPENAI_API_KEY=your-api-key-here
# 2. Or export in terminal: export OPENAI_API_KEY="your-api-key-here"

# Configuration
tools_path = Path(__file__).parent / "tools.json"
output_dir = Path(__file__).parent / "output"

gen_config = GenerationConfig(
    num_samples=5,
    strategy="random",
    seed=42,
    max_attempts=3,
)

model_config = ModelConfig(
    model="gpt-4o-mini",
    temperature=0.7,
)

# Generate dataset
manifest = generate_dataset(output_dir, gen_config, model_config, tools_path=tools_path)

print(f"✓ Generated {manifest['num_generated']} records")
print(f"  Output: {output_dir}")
