# 🛠️ ToolsGen

[![PyPI version](https://img.shields.io/pypi/v/toolsgen)](https://pypi.org/project/toolsgen/)
[![image](https://img.shields.io/pypi/pyversions/toolsgen.svg)]()
[![CI](https://github.com/atasoglu/toolsgen/actions/workflows/test.yml/badge.svg)](https://github.com/atasoglu/toolsgen/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A modular Python library for synthesizing tool-calling datasets from JSON tool definitions using an LLM-as-a-judge pipeline. Designed for OpenAI-compatible APIs.

> **⚠️ Development Status**: This project is under active development. The API is not yet stable and may undergo significant changes. Breaking changes may occur between versions.

## Overview

ToolsGen automates the creation of tool-calling datasets for training and evaluating language models. It generates realistic user requests, produces corresponding tool calls, and evaluates their quality using a multi-dimensional rubric system.

### Key Features

- **Multi-role LLM Pipeline**: Separate models for problem generation, tool calling, and quality evaluation
- **Flexible Sampling Strategies**: Random, parameter-aware, and semantic clustering approaches
- **LLM-as-a-Judge Scoring**: Rubric-based evaluation with structured outputs
- **OpenAI-Compatible**: Works with OpenAI API and compatible providers (Azure OpenAI, local models via vLLM, etc.)
- **Hugging Face Ready**: JSONL output format compatible with Hugging Face datasets
- **Configurable Quality Control**: Adjustable scoring thresholds and retry mechanisms
- **Train/Val Splitting**: Built-in dataset splitting for model training workflows
- **Parallel Generation**: Multiprocessing pipeline to accelerate dataset creation on multi-core hosts

## Requirements

- Python 3.9+
- OpenAI API key (or compatible API endpoint)

## Installation

```bash
git clone https://github.com/atasoglu/toolsgen.git
cd toolsgen
pip install .
```

## Usage

### CLI Usage

```bash
# Check version
toolsgen version

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Generate dataset with default settings
toolsgen generate \
  --tools tools.json \
  --out output_dir \
  --num 100

# Advanced: Use different models and temperatures for each role
toolsgen generate \
  --tools tools.json \
  --out output_dir \
  --num 1000 \
  --strategy param_aware \
  --seed 42 \
  --train-split 0.9 \
  --workers 4 \
  --worker-batch-size 8 \
  --problem-model gpt-4o-mini --problem-temp 0.9 \
  --caller-model gpt-4o --caller-temp 0.3 \
  --judge-model gpt-4o --judge-temp 0.0

# Parallel generation with 6 workers processing four samples per task
toolsgen generate \
  --tools tools.json \
  --out output_dir \
  --num 500 \
  --workers 6 \
  --worker-batch-size 4

# Generate and push directly to Hugging Face Hub
export HF_TOKEN="your-hf-token-here"
toolsgen generate \
  --tools tools.json \
  --out output_dir \
  --num 100 \
  --push-to-hub \
  --repo-id username/dataset-name
```

### Python API Usage

```python
from pathlib import Path
from dotenv import load_dotenv

from toolsgen.core import GenerationConfig, ModelConfig, generate_dataset

load_dotenv()  # Load from .env file

# Configuration
tools_path = Path("tools.json")
output_dir = Path("output")

gen_config = GenerationConfig(
    num_samples=100,
    strategy="random",
    seed=42,
    train_split=0.9,  # 90% train, 10% validation
    batch_size=10,  # optional: iterate tools in batches
    shuffle_tools=True,  # optional: reshuffle tools between batches
    num_workers=4,  # enable multiprocessing
    worker_batch_size=2,  # samples per worker task
)

model_config = ModelConfig(
    model="gpt-4o-mini",
    temperature=0.7,
)

# Generate dataset from file
manifest = generate_dataset(output_dir, gen_config, model_config, tools_path=tools_path)

# Or use tools list directly (alternative to tools_path)
# from toolsgen.schema import ToolSpec
# tools = [ToolSpec(...), ToolSpec(...)]
# manifest = generate_dataset(output_dir, gen_config, model_config, tools=tools)

print(f"Generated {manifest['num_generated']}/{manifest['num_requested']} records")
print(f"Failed: {manifest['num_failed']} attempts")
```

### Push to Hugging Face Hub

```python
from pathlib import Path
from dotenv import load_dotenv

from toolsgen import GenerationConfig, ModelConfig, generate_dataset, push_to_hub

load_dotenv()  # Load from .env file

tools_path = Path("tools.json")
output_dir = Path("output")

gen_config = GenerationConfig(
    num_samples=100,
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
    repo_id="username/dataset-name",
    private=False,
)

print(f"Generated: {manifest['num_generated']} records")
print(f"Repository: {hub_info['repo_url']}")
```

See `examples/` directory for complete working examples.

**Note**: The examples in `examples/` use `python-dotenv` for convenience (load API keys from `.env` file). Install it with `pip install python-dotenv` if you want to use this approach.

## Output Format

### Dataset Files (JSONL)

Each line in `train.jsonl` (or `val.jsonl`) is a JSON record:

```json
{
  "id": "record_000001",
  "language": "english",
  "tools": [...],
  "messages": [
    {"role": "user", "content": "What's the weather in San Francisco?"}
  ],
  "assistant_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"San Francisco, CA\"}"
      }
    }
  ],
  "problem_metadata": {"generated": true, "user_request": "..."},
  "judge": {
    "tool_relevance": 0.4,
    "argument_quality": 0.38,
    "clarity": 0.2,
    "score": 0.98,
    "verdict": "accept",
    "rationale": "Excellent tool selection and argument quality",
    "rubric_version": "0.1.0",
    "model": "gpt-4o",
    "temperature": 0.0
  },
  "quality_tags": [],
  "tools_metadata": {"num_tools": 5}
}
```

### Manifest File

`manifest.json` contains generation metadata:

```json
{
  "version": "0.1.0",
  "num_requested": 1000,
  "num_generated": 987,
  "num_failed": 13,
  "strategy": "param_aware",
  "seed": 42,
  "train_split": 0.9,
  "tools_count": 15,
  "models": {
    "problem_generator": "gpt-4o-mini",
    "tool_caller": "gpt-4o",
    "judge": "gpt-4o"
  },
  "splits": {
    "train": 888,
    "val": 99
  }
}
```

## Testing

```bash
# Run all tests with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_generator.py

# Run with verbose output
pytest -v
```

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests with coverage
pytest --cov=src

# Run code quality checks
ruff check src tests --fix
ruff format src tests
```

## Architecture

For detailed information about the system architecture, pipeline, and core components, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Roadmap

### Planned Features
- [ ] Multi-turn conversation support
- [ ] Custom prompt template system
- [x] Parallel generation with multiprocessing
- [ ] Additional sampling strategies (coverage-based, difficulty-based)
- [x] Integration with Hugging Face Hub for direct dataset uploads
- [ ] Support for more LLM providers (Anthropic, Cohere, etc.)
- [ ] Web UI for dataset inspection and curation
- [ ] Advanced filtering and deduplication

### Known Limitations
- Single-turn conversations only
- English-focused prompts (multilingual support is experimental)
- No built-in tool execution or validation
- Limited to OpenAI-compatible APIs

## Contributing

Contributions are welcome! Please note that the API is still evolving. Before starting major work, please open an issue to discuss your proposed changes.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use ToolsGen in your research, please cite:

```bibtex
@software{toolsgen2025,
  title = {ToolsGen: Synthetic Tool-Calling Dataset Generator},
  author = {Ataşoğlu, Ahmet},
  year = {2025},
  url = {https://github.com/atasoglu/toolsgen}
}
```
