"""Hugging Face Hub integration for dataset uploads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from huggingface_hub import HfApi, create_repo

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


def push_to_hub(
    output_dir: Path,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Push generated dataset to Hugging Face Hub.

    Args:
        output_dir: Directory containing train.jsonl, val.jsonl, and manifest.json
        repo_id: Repository ID on Hugging Face Hub (e.g., "username/dataset-name")
        token: HF API token (if None, uses HF_TOKEN env var or cached token)
        private: Whether to create a private repository
        commit_message: Custom commit message

    Returns:
        Dictionary with upload metadata including repo_url

    Raises:
        ImportError: If huggingface_hub is not installed
        ValueError: If output_dir doesn't contain required files
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for Hub uploads. "
            "Install with: pip install huggingface_hub"
        )

    if not output_dir.exists():
        raise ValueError(f"Output directory not found: {output_dir}")

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"manifest.json not found in {output_dir}")

    api = HfApi(token=token)

    # Create repository
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    # Prepare files to upload
    files_to_upload = [manifest_path]

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    if train_path.exists():
        files_to_upload.append(train_path)
    if val_path.exists():
        files_to_upload.append(val_path)

    # Generate README if not exists
    readme_path = output_dir / "README.md"
    if not readme_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        readme_content = _generate_readme(repo_id, manifest)
        readme_path.write_text(readme_content, encoding="utf-8")
    files_to_upload.append(readme_path)

    # Upload files
    message = commit_message or "Upload ToolsGen dataset"

    for file_path in files_to_upload:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=message,
        )

    repo_url = f"https://huggingface.co/datasets/{repo_id}"

    return {
        "repo_id": repo_id,
        "repo_url": repo_url,
        "files_uploaded": [f.name for f in files_to_upload],
        "private": private,
    }


def _generate_readme(repo_id: str, manifest: Dict[str, Any]) -> str:
    """Generate a README.md for the dataset."""
    splits_info = manifest.get("splits", {})
    train_count = splits_info.get("train", 0)
    val_count = splits_info.get("val", 0)

    splits_section = ""
    if val_count > 0:
        splits_section = f"""
## Dataset Splits

| Split | Records |
|-------|---------|
| train | {train_count} |
| val   | {val_count} |
"""
    else:
        splits_section = f"\n## Dataset Size\n\n{train_count} training records\n"

    readme = f"""# {repo_id.split("/")[-1]}

This dataset was generated using [ToolsGen](https://github.com/atasoglu/toolsgen).

## Dataset Description

- **Generated samples**: {manifest.get("num_generated", 0)}
- **Tools count**: {manifest.get("tools_count", 0)}
- **Sampling strategy**: {manifest.get("strategy", "N/A")}
- **Models used**:
  - Problem generator: {manifest.get("models", {}).get("problem_generator", "N/A")}
  - Tool caller: {manifest.get("models", {}).get("tool_caller", "N/A")}
  - Judge: {manifest.get("models", {}).get("judge", "N/A")}
{splits_section}
## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

## Citation

```bibtex
@software{{toolsgen2025,
  title = {{ToolsGen: Synthetic Tool-Calling Dataset Generator}},
  author = {{Ataşoğlu, Ahmet}},
  year = {{2025}},
  url = {{https://github.com/atasoglu/toolsgen}}
}}
```
"""
    return readme
