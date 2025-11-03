from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import typer

from . import __version__
from .config import GenerationConfig, ModelConfig, RoleBasedModelConfig
from .generator import generate_dataset


app = typer.Typer(help="ToolsGen CLI - generate tool-calling datasets from tool specs")


@app.command()
def version() -> None:
    """Show package version."""

    typer.echo(__version__)


@app.command()
def generate(
    tools: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to tools.json (OpenAI-compatible tools)",
    ),
    out: Path = typer.Option(..., help="Output directory for dataset files"),
    n: int = typer.Option(10, min=1, help="Number of samples to generate"),
    strategy: str = typer.Option(
        "random", help="Sampling strategy: random, param_aware, or semantic"
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed for deterministic runs"),
    model: str = typer.Option(
        "gpt-4o-mini",
        help="Model name (e.g., gpt-4o-mini, gpt-4, claude-3-sonnet)",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        help="Custom base URL for OpenAI-compatible API (overrides OPENAI_BASE_URL)",
    ),
    temperature: float = typer.Option(
        0.7, min=0.0, max=2.0, help="Sampling temperature"
    ),
    max_tokens: Optional[int] = typer.Option(
        None, min=1, help="Maximum tokens per response"
    ),
    language: str = typer.Option(
        "english",
        help="Language name for user requests (e.g., english, turkish, spanish, french, german)",
    ),
    problem_model: Optional[str] = typer.Option(
        None, help="Model for problem generation (defaults to --model)"
    ),
    caller_model: Optional[str] = typer.Option(
        None, help="Model for tool calling (defaults to --model)"
    ),
    judge_model: Optional[str] = typer.Option(
        None, help="Model for judging (defaults to --model)"
    ),
    problem_temp: Optional[float] = typer.Option(
        None,
        min=0.0,
        max=2.0,
        help="Temperature for problem generation (defaults to --temperature)",
    ),
    caller_temp: Optional[float] = typer.Option(
        None,
        min=0.0,
        max=2.0,
        help="Temperature for tool calling (defaults to --temperature)",
    ),
    judge_temp: Optional[float] = typer.Option(
        None,
        min=0.0,
        max=2.0,
        help="Temperature for judging (defaults to --temperature)",
    ),
    max_attempts: int = typer.Option(
        3, min=1, help="Maximum retry attempts per sample"
    ),
) -> None:
    """Generate a tool-calling dataset from tool specifications.

    This command uses an LLM to generate natural language user requests and
    corresponding tool calls, creating a dataset suitable for training or
    evaluating tool-calling models.
    """
    if strategy not in ("random", "param_aware", "semantic"):
        typer.echo(
            f"Error: strategy must be 'random', 'param_aware', or 'semantic', got '{strategy}'",
            err=True,
        )
        raise typer.Exit(1)

    # Load environment variables
    dotenv_path = Path(".env")
    if dotenv_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path)
        except ImportError:
            pass  # python-dotenv not installed, skip

    gen_config = GenerationConfig(
        num_samples=n,
        strategy=strategy,
        seed=seed,
        train_split=1.0,  # Default: no split, can be added as CLI option later
        language=language,
        max_attempts=max_attempts,
    )

    # Create role-based config if any role-specific options are provided
    model_config: Union[ModelConfig, RoleBasedModelConfig]
    if (
        problem_model
        or caller_model
        or judge_model
        or problem_temp is not None
        or caller_temp is not None
        or judge_temp is not None
    ):
        model_config = RoleBasedModelConfig(
            problem_generator=ModelConfig(
                model=problem_model or model,
                base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                api_key_env="OPENAI_API_KEY",
                temperature=problem_temp if problem_temp is not None else temperature,
                max_tokens=max_tokens,
            ),
            tool_caller=ModelConfig(
                model=caller_model or model,
                base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                api_key_env="OPENAI_API_KEY",
                temperature=caller_temp if caller_temp is not None else temperature,
                max_tokens=max_tokens,
            ),
            judge=ModelConfig(
                model=judge_model or model,
                base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                api_key_env="OPENAI_API_KEY",
                temperature=judge_temp if judge_temp is not None else temperature,
                max_tokens=max_tokens,
            ),
        )
    else:
        model_config = ModelConfig(
            model=model,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key_env="OPENAI_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        typer.echo(f"Generating {n} samples using {model}...")
        manifest = generate_dataset(tools, out, gen_config, model_config)

        typer.echo(f"\nGenerated {manifest['num_generated']} records")
        typer.echo(f"  - Requested: {manifest['num_requested']}")
        typer.echo(f"  - Failed: {manifest['num_failed']}")
        typer.echo(f"  - Output directory: {out}")

        splits = manifest.get("splits", {})
        if splits:
            typer.echo("  - Splits:")
            for split_name, count in splits.items():
                typer.echo(f"    * {split_name}.jsonl: {count} records")
        else:
            typer.echo(f"  - train.jsonl: {manifest['num_generated']} records")

        typer.echo(f"  - Manifest: {out / 'manifest.json'}")
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
