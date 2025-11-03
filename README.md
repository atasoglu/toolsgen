# ToolsGen

A modular Python library to synthesize English tool-calling datasets from JSON tool definitions using an LLM-as-a-judge pipeline. OpenAI-compatible and Hugging Face friendly.

## Installation

```bash
# Recommended: uv
uv venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

## CLI Usage

```bash
python -m toolsgen.cli version
python -m toolsgen.cli generate --tools tools.json --out out_dir --n 1000 --strategy random --seed 42
```

## Run Tests

```bash
pytest --cov=src
```

## Pre-commit

Install dev tools and enable hooks:

```bash
uv pip install -r requirements-dev.txt
pre-commit install
```

Run hooks on all files:

```bash
pre-commit run --all-files
```

Hooks included:
- trailing whitespace, EOF fixer, YAML/JSON checks, JSON pretty-format
- black, flake8, ruff (auto-fix), mypy, pyupgrade

## License

MIT
