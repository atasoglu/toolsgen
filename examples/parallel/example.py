"""Parallel generation example - accelerate dataset creation with multiprocessing."""

from pathlib import Path

from dotenv import load_dotenv

from toolsgen import GenerationConfig, ModelConfig, generate_dataset

# Automatically load environment variables from .env if present
load_dotenv()

# NOTE: Set OPENAI_API_KEY environment variable before running
# e.g. export OPENAI_API_KEY="your-api-key-here"


def main() -> None:
    tools_path = Path(__file__).parent / "tools.json"
    output_dir = Path(__file__).parent / "output"

    # Configure multiprocessing with 6 workers, each handling 3 samples per task
    # Shuffle tools between batches to mix coverage while the workers run in parallel
    # Increase num_samples to better showcase the throughput benefits of multiprocessing

    gen_config = GenerationConfig(
        num_samples=30,
        strategy="semantic",
        seed=2025,
        max_attempts=3,
        batch_size=2,
        shuffle_tools=True,
        num_workers=6,
        worker_batch_size=3,
    )

    # Single model configuration shared across roles
    model_config = ModelConfig(
        model="gpt-4o-mini",
        temperature=0.6,
    )

    manifest = generate_dataset(
        output_dir, gen_config, model_config, tools_path=tools_path
    )

    print(
        f"\n✓ Parallel run complete: {manifest['num_generated']}/{manifest['num_requested']} records"
    )
    if manifest["num_failed"]:
        print(f"  Failed attempts: {manifest['num_failed']}")
    print(f"  Workers used: {gen_config.num_workers}")
    print(f"  Worker batch size: {gen_config.worker_batch_size}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
