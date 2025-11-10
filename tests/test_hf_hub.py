"""Tests for Hugging Face Hub integration."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_output_dir(tmp_path):
    """Create a mock output directory with dataset files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create manifest.json
    manifest = {
        "version": "0.1.0",
        "num_requested": 100,
        "num_generated": 95,
        "num_failed": 5,
        "strategy": "random",
        "seed": 42,
        "train_split": 0.9,
        "tools_count": 10,
        "models": {
            "problem_generator": "gpt-4o-mini",
            "tool_caller": "gpt-4o-mini",
            "judge": "gpt-4o-mini",
        },
        "splits": {"train": 85, "val": 10},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Create train.jsonl
    (output_dir / "train.jsonl").write_text('{"id": "1"}\n', encoding="utf-8")

    # Create val.jsonl
    (output_dir / "val.jsonl").write_text('{"id": "2"}\n', encoding="utf-8")

    return output_dir


def test_push_to_hub_missing_huggingface_hub():
    """Test error when huggingface_hub is not installed."""
    from toolsgen.hf_hub import push_to_hub

    with patch("toolsgen.hf_hub.HF_HUB_AVAILABLE", False):
        with pytest.raises(ImportError, match="huggingface_hub is required"):
            push_to_hub(Path("output"), "user/repo")


def test_push_to_hub_missing_output_dir():
    """Test error when output directory doesn't exist."""
    from toolsgen.hf_hub import push_to_hub

    with patch("toolsgen.hf_hub.HF_HUB_AVAILABLE", True):
        with pytest.raises(ValueError, match="Output directory not found"):
            push_to_hub(Path("nonexistent"), "user/repo")


def test_push_to_hub_missing_manifest(tmp_path):
    """Test error when manifest.json is missing."""
    from toolsgen.hf_hub import push_to_hub

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("toolsgen.hf_hub.HF_HUB_AVAILABLE", True):
        with pytest.raises(ValueError, match="manifest.json not found"):
            push_to_hub(output_dir, "user/repo")


def test_push_to_hub_success(mock_output_dir):
    """Test successful push to Hub."""
    import sys

    # Mock huggingface_hub module
    mock_hf_hub = MagicMock()
    mock_api = MagicMock()
    mock_hf_hub.HfApi.return_value = mock_api
    sys.modules["huggingface_hub"] = mock_hf_hub

    try:
        # Force reimport with mocked module
        import importlib
        import toolsgen.hf_hub

        importlib.reload(toolsgen.hf_hub)
        from toolsgen.hf_hub import push_to_hub

        result = push_to_hub(
            output_dir=mock_output_dir,
            repo_id="user/test-dataset",
            token="test-token",
            private=True,
        )

        # Verify create_repo was called
        mock_hf_hub.create_repo.assert_called_once_with(
            repo_id="user/test-dataset",
            repo_type="dataset",
            private=True,
            exist_ok=True,
            token="test-token",
        )

        # Verify upload_file was called for each file
        assert mock_api.upload_file.call_count == 4  # manifest, train, val, README

        # Verify result
        assert result["repo_id"] == "user/test-dataset"
        assert result["repo_url"] == "https://huggingface.co/datasets/user/test-dataset"
        assert result["private"] is True
        assert len(result["files_uploaded"]) == 4
    finally:
        # Cleanup
        if "huggingface_hub" in sys.modules:
            del sys.modules["huggingface_hub"]
        importlib.reload(toolsgen.hf_hub)


def test_generate_readme():
    """Test README generation."""
    from toolsgen.hf_hub import _generate_readme

    manifest = {
        "num_generated": 100,
        "tools_count": 15,
        "strategy": "random",
        "models": {
            "problem_generator": "gpt-4o-mini",
            "tool_caller": "gpt-4o",
            "judge": "gpt-4o",
        },
        "splits": {"train": 90, "val": 10},
    }

    readme = _generate_readme("user/test-dataset", manifest)

    assert "test-dataset" in readme
    assert "100" in readme
    assert "15" in readme
    assert "random" in readme
    assert "gpt-4o-mini" in readme
    assert "train | 90" in readme
    assert "val   | 10" in readme
    assert "load_dataset" in readme
    assert "user/test-dataset" in readme


def test_generate_readme_no_val_split():
    """Test README generation without validation split."""
    from toolsgen.hf_hub import _generate_readme

    manifest = {
        "num_generated": 100,
        "tools_count": 15,
        "strategy": "random",
        "models": {
            "problem_generator": "gpt-4o-mini",
            "tool_caller": "gpt-4o",
            "judge": "gpt-4o",
        },
        "splits": {"train": 100},
    }

    readme = _generate_readme("user/test-dataset", manifest)

    assert "100 training records" in readme
    assert "| Split |" not in readme
