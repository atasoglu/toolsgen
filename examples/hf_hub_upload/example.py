"""Example: Generate dataset and push to Hugging Face Hub."""

from pathlib import Path

from dotenv import load_dotenv

from toolsgen import GenerationConfig, ModelConfig, generate_dataset, push_to_hub

# Load environment variables from .env file
load_dotenv()

# Configuration
tools_path = Path(__file__).parent.parent / "basic" / "tools.json"
output_dir = Path(__file__).parent / "output"

gen_config = GenerationConfig(
    num_samples=50,
    strategy="random",
    seed=42,
    train_split=0.9,
)

model_config = ModelConfig(
    model="gpt-4o-mini",
    temperature=0.7,
)

# Generate dataset
manifest = generate_dataset(
    output_dir=output_dir,
    gen_config=gen_config,
    model_config=model_config,
    tools_path=tools_path,
)

# Push to Hub
hub_info = push_to_hub(
    output_dir=output_dir,
    repo_id="your-username/your-dataset-name",  # Change this!
    private=False,
)

print("\n✓ Dataset generated and uploaded!")
print(f"  Generated: {manifest['num_generated']} records")
print(f"  Repository: {hub_info['repo_url']}")
