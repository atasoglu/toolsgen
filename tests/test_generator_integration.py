"""Integration tests for dataset generator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from toolsgen.core.config import GenerationConfig, ModelConfig, RoleBasedModelConfig
from toolsgen.core.generator import generate_dataset
from toolsgen.core.record_builder import _build_record
from toolsgen.schema import AssistantToolCall, ToolFunction, ToolSpec


@patch("toolsgen.core.record_builder.judge_tool_calls")
@patch("toolsgen.core.record_builder.generate_tool_calls")
@patch("toolsgen.core.record_builder.generate_problem")
def test_generate_sample_success(
    mock_problem: MagicMock, mock_tool_calls: MagicMock, mock_judge: MagicMock
) -> None:
    """Test successful sample generation."""
    mock_problem.return_value = "Send an email to user@example.com"

    mock_tool_call = AssistantToolCall(
        id="call_1",
        function={"name": "send_email", "arguments": '{"to": "user@example.com"}'},
    )
    mock_tool_calls.return_value = [mock_tool_call]

    mock_judge_result = MagicMock()
    mock_judge_result.to_dict.return_value = {"score": 0.9, "verdict": "accept"}
    mock_judge.return_value = mock_judge_result

    problem_client = MagicMock()
    caller_client = MagicMock()
    judge_client = MagicMock()

    tools = [
        ToolSpec(function=ToolFunction(name="send_email", description="Send email"))
    ]

    role_config = RoleBasedModelConfig.from_single_config(ModelConfig(model="gpt-4"))

    record = _build_record(
        problem_client,
        caller_client,
        judge_client,
        "rec_001",
        tools,
        role_config,
        "english",
    )

    assert record is not None
    assert record.id == "rec_001"
    assert record.language == "english"
    assert len(record.messages) == 1
    assert record.messages[0].content == "Send an email to user@example.com"
    assert len(record.assistant_calls) == 1
    assert record.judge["score"] == 0.9


@patch("toolsgen.core.record_builder.generate_problem")
def test_generate_sample_problem_fails(mock_problem: MagicMock) -> None:
    """Test sample generation when problem generation fails."""
    mock_problem.return_value = None

    problem_client = MagicMock()
    caller_client = MagicMock()
    judge_client = MagicMock()

    tools = [ToolSpec(function=ToolFunction(name="test"))]
    role_config = RoleBasedModelConfig.from_single_config(ModelConfig(model="gpt-4"))

    record = _build_record(
        problem_client,
        caller_client,
        judge_client,
        "rec_001",
        tools,
        role_config,
        "english",
    )

    assert record is None


@patch("toolsgen.core.record_builder.generate_tool_calls")
@patch("toolsgen.core.record_builder.generate_problem")
def test_generate_sample_tool_calls_fail(
    mock_problem: MagicMock, mock_tool_calls: MagicMock
) -> None:
    """Test sample generation when tool call generation fails."""
    mock_problem.return_value = "Test request"
    mock_tool_calls.return_value = []

    problem_client = MagicMock()
    caller_client = MagicMock()
    judge_client = MagicMock()

    tools = [ToolSpec(function=ToolFunction(name="test"))]
    role_config = RoleBasedModelConfig.from_single_config(ModelConfig(model="gpt-4"))

    record = _build_record(
        problem_client,
        caller_client,
        judge_client,
        "rec_001",
        tools,
        role_config,
        "english",
    )

    assert record is None


@patch("toolsgen.core.record_builder.judge_tool_calls")
@patch("toolsgen.core.record_builder.generate_tool_calls")
@patch("toolsgen.core.record_builder.generate_problem")
def test_generate_sample_judge_fails(
    mock_problem: MagicMock, mock_tool_calls: MagicMock, mock_judge: MagicMock
) -> None:
    """Test sample generation continues when judge fails."""
    mock_problem.return_value = "Test request"
    mock_tool_call = AssistantToolCall(
        id="call_1", function={"name": "test", "arguments": "{}"}
    )
    mock_tool_calls.return_value = [mock_tool_call]
    mock_judge.side_effect = Exception("Judge failed")

    problem_client = MagicMock()
    caller_client = MagicMock()
    judge_client = MagicMock()

    tools = [ToolSpec(function=ToolFunction(name="test"))]
    role_config = RoleBasedModelConfig.from_single_config(ModelConfig(model="gpt-4"))

    record = _build_record(
        problem_client,
        caller_client,
        judge_client,
        "rec_001",
        tools,
        role_config,
        "english",
    )

    assert record is not None
    assert record.id == "rec_001"


@patch("toolsgen.core.sequential.RecordBuilder.generate_record")
@patch("toolsgen.core.client.create_openai_client")
def test_generate_dataset_basic(
    mock_create_client: MagicMock,
    mock_generate_sample: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test basic dataset generation."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    tools_path = tmp_path / "tools.json"
    tools_data = [
        {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "Test",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    tools_path.write_text(json.dumps(tools_data), encoding="utf-8")

    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    mock_record = MagicMock()
    mock_record.id = "rec_000000"
    mock_record.model_dump.return_value = {"id": "rec_000000"}
    mock_generate_sample.return_value = mock_record

    gen_config = GenerationConfig(num_samples=3, strategy="random", seed=42)
    model_config = ModelConfig(model="gpt-4")
    output_dir = tmp_path / "output"

    manifest = generate_dataset(
        output_dir, gen_config, model_config, tools_path=tools_path
    )

    assert manifest["num_requested"] == 3
    assert manifest["num_generated"] == 3
    assert manifest["strategy"] == "random"
    assert manifest["seed"] == 42

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "train.jsonl").exists()


@patch("toolsgen.core.sequential.RecordBuilder.generate_record")
@patch("toolsgen.core.client.create_openai_client")
def test_generate_dataset_with_splits(
    mock_create_client: MagicMock,
    mock_generate_sample: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test dataset generation with train/val splits."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    tools_path = tmp_path / "tools.json"
    tools_path.write_text(
        '[{"type": "function", "function": {"name": "test"}}]', encoding="utf-8"
    )

    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    def create_mock_record(call_count: list[int] = [0]) -> MagicMock:
        record = MagicMock()
        record.id = f"rec_{call_count[0]:06d}"
        record.model_dump.return_value = {"id": record.id}
        call_count[0] += 1
        return record

    def side_effect(*args: object, **kwargs: object) -> MagicMock:
        return create_mock_record()

    mock_generate_sample.side_effect = side_effect

    gen_config = GenerationConfig(
        num_samples=10, strategy="random", seed=42, train_split=0.8
    )
    model_config = ModelConfig(model="gpt-4")
    output_dir = tmp_path / "output"

    manifest = generate_dataset(
        output_dir, gen_config, model_config, tools_path=tools_path
    )

    assert manifest["num_generated"] == 10
    assert manifest["train_split"] == 0.8
    assert "splits" in manifest
    assert manifest["splits"]["train"] == 8
    assert manifest["splits"]["val"] == 2

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()


@patch("toolsgen.core.sequential.RecordBuilder.generate_record")
@patch("toolsgen.core.client.create_openai_client")
def test_generate_dataset_with_failures(
    mock_create_client: MagicMock,
    mock_generate_sample: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test dataset generation with some failures."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    tools_path = tmp_path / "tools.json"
    tools_path.write_text(
        '[{"type": "function", "function": {"name": "test"}}]', encoding="utf-8"
    )

    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    call_count = [0]

    def mock_sample_gen(*args: object, **kwargs: object) -> MagicMock | None:
        call_count[0] += 1
        if call_count[0] % 2 == 1:
            return None
        else:
            record = MagicMock()
            record.id = f"rec_{(call_count[0] // 2) - 1:06d}"
            record.model_dump.return_value = {"id": record.id}
            return record

    mock_generate_sample.side_effect = mock_sample_gen

    gen_config = GenerationConfig(num_samples=3, max_attempts=2)
    model_config = ModelConfig(model="gpt-4")
    output_dir = tmp_path / "output"

    manifest = generate_dataset(
        output_dir, gen_config, model_config, tools_path=tools_path
    )

    assert manifest["num_requested"] == 3
    assert manifest["num_generated"] == 3
    assert manifest["num_failed"] >= 0


@patch("toolsgen.core.client.create_openai_client")
def test_generate_dataset_role_based_config(
    mock_create_client: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test dataset generation with role-based model config."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    tools_path = tmp_path / "tools.json"
    tools_path.write_text(
        '[{"type": "function", "function": {"name": "test"}}]', encoding="utf-8"
    )

    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    role_config = RoleBasedModelConfig(
        problem_generator=ModelConfig(model="gpt-4"),
        tool_caller=ModelConfig(model="gpt-4o"),
        judge=ModelConfig(model="gpt-4o-mini"),
    )

    gen_config = GenerationConfig(num_samples=1)
    output_dir = tmp_path / "output"

    with patch("toolsgen.core.sequential.RecordBuilder.generate_record") as mock_gen:
        mock_record = MagicMock()
        mock_record.id = "rec_000000"
        mock_record.model_dump.return_value = {"id": "rec_000000"}
        mock_gen.return_value = mock_record

        manifest = generate_dataset(
            output_dir, gen_config, role_config, tools_path=tools_path
        )

        assert manifest["models"]["problem_generator"] == "gpt-4"
        assert manifest["models"]["tool_caller"] == "gpt-4o"
        assert manifest["models"]["judge"] == "gpt-4o-mini"


@patch("toolsgen.core.sequential.RecordBuilder.generate_record")
@patch("toolsgen.core.client.create_openai_client")
def test_generate_dataset_param_aware_strategy(
    mock_create_client: MagicMock,
    mock_generate_sample: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test dataset generation with param_aware strategy."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    tools_path = tmp_path / "tools.json"
    tools_data = [
        {
            "type": "function",
            "function": {
                "name": "tool1",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool2",
                "parameters": {
                    "type": "object",
                    "properties": {"y": {"type": "string"}, "z": {"type": "string"}},
                },
            },
        },
    ]
    tools_path.write_text(json.dumps(tools_data), encoding="utf-8")

    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    mock_record = MagicMock()
    mock_record.id = "rec_000000"
    mock_record.model_dump.return_value = {"id": "rec_000000"}
    mock_generate_sample.return_value = mock_record

    gen_config = GenerationConfig(num_samples=2, strategy="param_aware", seed=42)
    model_config = ModelConfig(model="gpt-4")
    output_dir = tmp_path / "output"

    manifest = generate_dataset(
        output_dir, gen_config, model_config, tools_path=tools_path
    )

    assert manifest["strategy"] == "param_aware"
    assert manifest["num_generated"] == 2


@patch("toolsgen.core.sequential.RecordBuilder.generate_record")
@patch("toolsgen.core.client.create_openai_client")
def test_generate_dataset_with_tools_list(
    mock_create_client: MagicMock,
    mock_generate_sample: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test dataset generation with direct tools list instead of path."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    tools = [
        ToolSpec(
            function=ToolFunction(
                name="test_func",
                description="Test function",
                parameters={"type": "object", "properties": {}},
            )
        )
    ]

    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    mock_record = MagicMock()
    mock_record.id = "rec_000000"
    mock_record.model_dump.return_value = {"id": "rec_000000"}
    mock_generate_sample.return_value = mock_record

    gen_config = GenerationConfig(num_samples=2, strategy="random", seed=42)
    model_config = ModelConfig(model="gpt-4")
    output_dir = tmp_path / "output"

    manifest = generate_dataset(output_dir, gen_config, model_config, tools=tools)

    assert manifest["num_requested"] == 2
    assert manifest["num_generated"] == 2
    assert manifest["tools_count"] == 1


def test_generate_dataset_missing_tools_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that dataset generation fails when neither tools_path nor tools is provided."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    gen_config = GenerationConfig(num_samples=1)
    model_config = ModelConfig(model="gpt-4")
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="Either tools_path or tools must be provided"):
        generate_dataset(output_dir, gen_config, model_config)
