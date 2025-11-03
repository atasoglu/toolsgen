from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import __version__


app = typer.Typer(help="ToolsGen CLI - generate tool-calling datasets from tool specs")


def _read_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        "random", help="Sampling strategy: random or semantic"
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed for deterministic runs"),
) -> None:
    """Generate a dataset (stub implementation).

    This skeleton command validates inputs and prepares the output directory.
    Further functionality will be implemented in subsequent steps.
    """

    tools_data = _read_json_file(tools)
    out.mkdir(parents=True, exist_ok=True)

    # Placeholder behavior for skeleton: write a minimal manifest
    manifest = {
        "version": __version__,
        "num_requested": n,
        "strategy": strategy,
        "seed": seed,
        "tools_count": len(tools_data) if isinstance(tools_data, list) else 1,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(f"Initialized generation stub. Wrote: {out / 'manifest.json'}")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
