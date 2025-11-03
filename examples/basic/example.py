"""Basic Python example - ToolsGen usage"""

from pathlib import Path
from dotenv import load_dotenv
from toolsgen.config import GenerationConfig, ModelConfig
from toolsgen.generator import generate_dataset

# Load environment variables from .env
load_dotenv()

# Yapılandırma
tools_path = Path(__file__).parent / "tools.json"
output_dir = Path(__file__).parent / "output"

gen_config = GenerationConfig(
    num_samples=5,
    strategy="random",
    seed=42,
)

model_config = ModelConfig(
    model="gpt-4o-mini",
    temperature=0.7,
)

# Dataset oluştur
manifest = generate_dataset(tools_path, output_dir, gen_config, model_config)

print(f"✓ Generated {manifest['num_generated']} records")
print(f"  Output: {output_dir}")
