# Hugging Face Hub Upload Example

This example demonstrates how to generate a dataset and push it directly to Hugging Face Hub.

## Prerequisites

1. OpenAI API key
2. Hugging Face account and token with write access

## Setup

```bash
# Install dependencies
pip install toolsgen huggingface_hub python-dotenv

# Create .env file from example
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your-openai-api-key
# HF_TOKEN=your-huggingface-token
```

## Usage

### Python API

```python
python example.py
```

Make sure to update the `repo_id` in the script to your own repository name.

### CLI

```bash
toolsgen generate \
  --tools ../basic/tools.json \
  --out output \
  --num 50 \
  --push-to-hub \
  --repo-id your-username/your-dataset-name
```

## What Gets Uploaded

The following files are automatically uploaded to your HF Hub repository:

- `train.jsonl` - Training dataset
- `val.jsonl` - Validation dataset (if train_split < 1.0)
- `manifest.json` - Generation metadata
- `README.md` - Auto-generated dataset card

## Repository Visibility

By default, repositories are public. To create a private repository:

**Python API:**
```python
hub_info = push_to_hub(
    output_dir=output_dir,
    repo_id="username/dataset-name",
    private=True,
)
```

**CLI:**
```bash
toolsgen generate ... --push-to-hub --private
```

## Notes

- The HF token can be provided via `--hf-token` flag or `HF_TOKEN` environment variable
- If a repository already exists, it will be updated with new files
- A dataset card (README.md) is automatically generated if not present
