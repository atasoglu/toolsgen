"""Tests for CLI interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from toolsgen.cli import cmd_generate, cmd_version, create_parser, main


def test_create_parser() -> None:
    """Test that argument parser is created correctly."""
    parser = create_parser()
    assert parser.prog == "toolsgen"

    # Test version subcommand
    args = parser.parse_args(["version"])
    assert args.command == "version"

    # Test generate subcommand with minimal args
    args = parser.parse_args(["generate", "--tools", "tools.json", "--out", "output"])
    assert args.command == "generate"
    assert args.tools == Path("tools.json")
    assert args.out == Path("output")
    assert args.num == 10  # default
    assert args.workers == 1
    assert args.worker_batch_size == 1


def test_cmd_version(capsys: pytest.CaptureFixture) -> None:
    """Test version command output."""
    cmd_version()
    captured = capsys.readouterr()
    assert captured.out.strip()  # Should print non-empty version


def test_cmd_generate_missing_tools_file(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test generate command fails when tools file doesn't exist."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "generate",
            "--tools",
            str(tmp_path / "nonexistent.json"),
            "--out",
            str(tmp_path / "out"),
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        cmd_generate(args)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Tools file not found" in captured.err


@pytest.mark.parametrize(
    "flag,value,error_msg",
    [
        ("--num", "0", "--num must be at least 1"),
        ("--train-split", "1.5", "--train-split must be between 0.0 and 1.0"),
        ("--temperature", "3.0", "--temperature must be between 0.0 and 2.0"),
        ("--workers", "0", "--workers must be at least 1"),
        (
            "--worker-batch-size",
            "0",
            "--worker-batch-size must be at least 1",
        ),
    ],
)
def test_cmd_generate_invalid_params(
    flag: str, value: str, error_msg: str, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test generate command fails with invalid parameters."""
    tools_path = tmp_path / "tools.json"
    tools_path.write_text("[]", encoding="utf-8")

    parser = create_parser()
    args = parser.parse_args(
        [
            "generate",
            "--tools",
            str(tools_path),
            "--out",
            str(tmp_path / "out"),
            flag,
            value,
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        cmd_generate(args)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert error_msg in captured.err


@patch("toolsgen.cli.generate_dataset")
def test_cmd_generate_success(
    mock_generate: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test successful generate command execution."""
    # Set up test environment
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    tools_path = tmp_path / "tools.json"
    tools_data = [
        {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "Test function",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    tools_path.write_text(json.dumps(tools_data), encoding="utf-8")
    out_dir = tmp_path / "out"

    # Mock generate_dataset to return manifest
    mock_generate.return_value = {
        "num_requested": 5,
        "num_generated": 5,
        "num_failed": 0,
        "strategy": "random",
        "splits": {"train": 5},
    }

    parser = create_parser()
    args = parser.parse_args(
        [
            "generate",
            "--tools",
            str(tools_path),
            "--out",
            str(out_dir),
            "--num",
            "5",
            "--strategy",
            "random",
            "--seed",
            "42",
        ]
    )

    cmd_generate(args)

    # Verify generate_dataset was called
    assert mock_generate.called
    gen_config = mock_generate.call_args[0][1]
    assert gen_config.num_workers == 1
    assert gen_config.worker_batch_size == 1
    captured = capsys.readouterr()
    assert "Generating 5 samples" in captured.out
    assert "Generated 5 records" in captured.out


@patch("toolsgen.cli.generate_dataset")
def test_cmd_generate_with_role_models(
    mock_generate: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test generate command with role-specific models."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    tools_path = tmp_path / "tools.json"
    tools_path.write_text("[]", encoding="utf-8")

    mock_generate.return_value = {
        "num_requested": 1,
        "num_generated": 1,
        "num_failed": 0,
        "strategy": "random",
        "splits": {},
    }

    parser = create_parser()
    args = parser.parse_args(
        [
            "generate",
            "--tools",
            str(tools_path),
            "--out",
            str(tmp_path / "out"),
            "--num",
            "1",
            "--problem-model",
            "gpt-4",
            "--caller-model",
            "gpt-4o",
            "--judge-model",
            "gpt-4o-mini",
            "--problem-temp",
            "0.9",
            "--caller-temp",
            "0.1",
            "--judge-temp",
            "0.3",
            "--workers",
            "2",
            "--worker-batch-size",
            "4",
        ]
    )

    cmd_generate(args)

    # Verify the call was made with role-based config
    call_args = mock_generate.call_args
    model_config = call_args[0][2]  # 3rd positional argument
    assert hasattr(model_config, "problem_generator")
    assert model_config.problem_generator.model == "gpt-4"
    assert model_config.tool_caller.model == "gpt-4o"
    assert model_config.judge.model == "gpt-4o-mini"
    gen_config = call_args[0][1]
    assert gen_config.num_workers == 2
    assert gen_config.worker_batch_size == 4


@pytest.mark.parametrize(
    "command,expect_exit,exit_code",
    [
        ([], True, 0),  # No command - shows help
        (["version"], False, None),  # Version command succeeds
        (["invalid"], True, 2),  # Invalid command - argparse error
    ],
)
def test_main_commands(
    command: list,
    expect_exit: bool,
    exit_code: int | None,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test main function with various commands."""
    with patch.object(sys, "argv", ["toolsgen"] + command):
        if expect_exit:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == exit_code
        else:
            main()
            captured = capsys.readouterr()
            assert captured.out.strip()  # Non-empty output for version
