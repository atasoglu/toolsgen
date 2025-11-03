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

## Configuration

Create a `.env` file in the project root with your OpenAI API key:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
# OPENAI_API_KEY=sk-your-actual-api-key-here
```

Or set the environment variable directly:
```bash
export OPENAI_API_KEY="your-api-key-here"  # Linux/macOS
$env:OPENAI_API_KEY="your-api-key-here"    # Windows PowerShell
```

## CLI Usage

```bash
# Check version
python -m toolsgen.cli version

# Generate dataset (requires OPENAI_API_KEY in .env or environment)
python -m toolsgen.cli generate \
  --tools tools.json \
  --out out_dir \
  --n 1000 \
  --strategy random \
  --seed 42 \
  --model gpt-4o-mini
```

### End-to-end example

- See `examples/basic` for a runnable example with a minimal `tools.json` and scripts for Windows PowerShell and bash.
  - Windows:
    ```powershell
    ./examples/basic/run.ps1
    ```
  - macOS/Linux:
    ```bash
    bash ./examples/basic/run.sh
    ```
  The run writes `examples/basic/out/manifest.json` summarizing the generation.

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
