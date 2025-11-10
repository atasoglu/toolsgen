"""Command-line interface for ToolsGen."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Union

from . import __version__
from .core import GenerationConfig, ModelConfig, RoleBasedModelConfig, generate_dataset


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="toolsgen",
        description="ToolsGen - generate tool-calling datasets from tool specs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Version command
    subparsers.add_parser("version", help="Show package version")

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate a tool-calling dataset from tool specifications",
    )

    # Required arguments
    gen_parser.add_argument(
        "--tools",
        type=Path,
        required=True,
        help="Path to tools.json (OpenAI-compatible tools)",
    )
    gen_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for dataset files",
    )

    # Generation config
    gen_parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of samples to generate (default: 10)",
    )
    gen_parser.add_argument(
        "--strategy",
        choices=["random", "param_aware", "semantic"],
        default="random",
        help="Sampling strategy (default: random)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic runs",
    )
    gen_parser.add_argument(
        "--language",
        default="english",
        help="Language name for user requests (default: english)",
    )
    gen_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum retry attempts per sample (default: 3)",
    )
    gen_parser.add_argument(
        "--train-split",
        type=float,
        default=1.0,
        help="Fraction of records for training split 0.0-1.0 (default: 1.0, no split)",
    )
    gen_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional number of tools per batch when sampling (default: all tools)",
    )
    gen_parser.add_argument(
        "--shuffle-tools",
        action="store_true",
        help="Shuffle tool order before batching (default: disabled)",
    )
    gen_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for generation (default: 1)",
    )
    gen_parser.add_argument(
        "--worker-batch-size",
        type=int,
        default=1,
        help="Number of samples each worker processes per task (default: 1)",
    )

    # Model config
    gen_parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)",
    )
    gen_parser.add_argument(
        "--base-url",
        default=None,
        help="Custom base URL for OpenAI-compatible API",
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    gen_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per response",
    )

    # Role-specific model config
    gen_parser.add_argument(
        "--problem-model",
        default=None,
        help="Model for problem generation (defaults to --model)",
    )
    gen_parser.add_argument(
        "--caller-model",
        default=None,
        help="Model for tool calling (defaults to --model)",
    )
    gen_parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for judging (defaults to --model)",
    )
    gen_parser.add_argument(
        "--problem-temp",
        type=float,
        default=None,
        help="Temperature for problem generation (defaults to --temperature)",
    )
    gen_parser.add_argument(
        "--caller-temp",
        type=float,
        default=None,
        help="Temperature for tool calling (defaults to --temperature)",
    )
    gen_parser.add_argument(
        "--judge-temp",
        type=float,
        default=None,
        help="Temperature for judging (defaults to --temperature)",
    )

    # Hugging Face Hub options
    gen_parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to Hugging Face Hub after generation",
    )
    gen_parser.add_argument(
        "--repo-id",
        default=None,
        help="HF Hub repository ID (e.g., 'username/dataset-name')",
    )
    gen_parser.add_argument(
        "--hf-token",
        default=None,
        help="HF API token (defaults to HF_TOKEN env var)",
    )
    gen_parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository on HF Hub",
    )

    return parser


def cmd_version() -> None:
    """Show package version."""
    print(__version__)


def cmd_generate(args: argparse.Namespace) -> None:
    """Run dataset generation.

    Args:
        args: Parsed command-line arguments.
    """
    # Validate inputs
    if not args.tools.exists():
        print(f"Error: Tools file not found: {args.tools}", file=sys.stderr)
        sys.exit(1)

    if args.num < 1:
        print("Error: --num must be at least 1", file=sys.stderr)
        sys.exit(1)

    if not 0.0 <= args.train_split <= 1.0:
        print("Error: --train-split must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    if not 0.0 <= args.temperature <= 2.0:
        print("Error: --temperature must be between 0.0 and 2.0", file=sys.stderr)
        sys.exit(1)

    if args.workers < 1:
        print("Error: --workers must be at least 1", file=sys.stderr)
        sys.exit(1)

    if args.worker_batch_size < 1:
        print("Error: --worker-batch-size must be at least 1", file=sys.stderr)
        sys.exit(1)

    # Create generation config
    gen_config = GenerationConfig(
        num_samples=args.num,
        strategy=args.strategy,
        seed=args.seed,
        train_split=args.train_split,
        language=args.language,
        max_attempts=args.max_attempts,
        batch_size=args.batch_size,
        shuffle_tools=args.shuffle_tools,
        num_workers=args.workers,
        worker_batch_size=args.worker_batch_size,
    )

    # Create model config
    model_config: Union[ModelConfig, RoleBasedModelConfig]
    if (
        args.problem_model
        or args.caller_model
        or args.judge_model
        or args.problem_temp is not None
        or args.caller_temp is not None
        or args.judge_temp is not None
    ):
        # Role-based configuration
        model_config = RoleBasedModelConfig(
            problem_generator=ModelConfig(
                model=args.problem_model or args.model,
                base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
                api_key_env="OPENAI_API_KEY",
                temperature=(
                    args.problem_temp
                    if args.problem_temp is not None
                    else args.temperature
                ),
                max_tokens=args.max_tokens,
            ),
            tool_caller=ModelConfig(
                model=args.caller_model or args.model,
                base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
                api_key_env="OPENAI_API_KEY",
                temperature=(
                    args.caller_temp
                    if args.caller_temp is not None
                    else args.temperature
                ),
                max_tokens=args.max_tokens,
            ),
            judge=ModelConfig(
                model=args.judge_model or args.model,
                base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
                api_key_env="OPENAI_API_KEY",
                temperature=(
                    args.judge_temp if args.judge_temp is not None else args.temperature
                ),
                max_tokens=args.max_tokens,
            ),
        )
    else:
        # Single model configuration
        model_config = ModelConfig(
            model=args.model,
            base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key_env="OPENAI_API_KEY",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    # Validate HF Hub options
    if args.push_to_hub and not args.repo_id:
        print("Error: --repo-id is required when using --push-to-hub", file=sys.stderr)
        sys.exit(1)

    # Generate dataset
    try:
        print(f"Generating {args.num} samples using {args.model}...")

        manifest = generate_dataset(
            args.out, gen_config, model_config, tools_path=args.tools
        )

        print(f"\nGenerated {manifest['num_generated']} records")
        print(f"  - Requested: {manifest['num_requested']}")
        print(f"  - Failed: {manifest['num_failed']}")
        print(f"  - Output directory: {args.out}")

        splits = manifest.get("splits", {})
        if splits:
            print("  - Splits:")
            for split_name, count in splits.items():
                print(f"    * {split_name}.jsonl: {count} records")
        else:
            print(f"  - train.jsonl: {manifest['num_generated']} records")

        print(f"  - Manifest: {args.out / 'manifest.json'}")

        if args.push_to_hub:
            from .hf_hub import push_to_hub

            print("\nPushing to Hugging Face Hub...")
            hub_info = push_to_hub(
                output_dir=args.out,
                repo_id=args.repo_id,
                token=args.hf_token,
                private=args.private,
            )
            print("✓ Pushed to Hugging Face Hub")
            print(f"  - Repository: {hub_info['repo_url']}")
            print(f"  - Files uploaded: {', '.join(hub_info['files_uploaded'])}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "version":
        cmd_version()
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
