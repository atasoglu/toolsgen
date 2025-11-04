"""Tests for I/O operations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolsgen.core.io import append_record_jsonl, load_tool_specs, write_dataset_jsonl
from toolsgen.schema import Message, Record, ToolFunction, ToolSpec


def test_load_tool_specs_success(tmp_path: Path) -> None:
    """Test loading valid tool specs from JSON."""
    tools_file = tmp_path / "tools.json"
    tools_data = [
        {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "Test function",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            },
        }
    ]
    tools_file.write_text(json.dumps(tools_data), encoding="utf-8")

    specs = load_tool_specs(tools_file)

    assert len(specs) == 1
    assert specs[0].function.name == "test_func"
    assert specs[0].function.description == "Test function"
    assert "x" in specs[0].function.parameters["properties"]


def test_load_tool_specs_invalid_not_list(tmp_path: Path) -> None:
    """Test loading fails when JSON is not a list."""
    tools_file = tmp_path / "tools.json"
    tools_file.write_text('{"invalid": "data"}', encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a list"):
        load_tool_specs(tools_file)


def test_load_tool_specs_empty(tmp_path: Path) -> None:
    """Test loading empty tool list."""
    tools_file = tmp_path / "tools.json"
    tools_file.write_text("[]", encoding="utf-8")

    specs = load_tool_specs(tools_file)
    assert specs == []


def test_load_tool_specs_multiple(tmp_path: Path) -> None:
    """Test loading multiple tool specs."""
    tools_file = tmp_path / "tools.json"
    tools_data = [
        {
            "type": "function",
            "function": {
                "name": "func1",
                "description": "First function",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "func2",
                "description": "Second function",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    tools_file.write_text(json.dumps(tools_data), encoding="utf-8")

    specs = load_tool_specs(tools_file)
    assert len(specs) == 2
    assert specs[0].function.name == "func1"
    assert specs[1].function.name == "func2"


def test_load_tool_specs_with_optional_fields(tmp_path: Path) -> None:
    """Test loading tool specs with optional fields missing."""
    tools_file = tmp_path / "tools.json"
    tools_data = [
        {
            "type": "function",
            "function": {
                "name": "minimal_func",
            },
        }
    ]
    tools_file.write_text(json.dumps(tools_data), encoding="utf-8")

    specs = load_tool_specs(tools_file)
    assert len(specs) == 1
    assert specs[0].function.name == "minimal_func"
    assert specs[0].function.description is None
    assert specs[0].function.parameters == {}


def test_write_dataset_jsonl_single_record(tmp_path: Path) -> None:
    """Test writing a single record to JSONL."""
    output_path = tmp_path / "output" / "data.jsonl"

    func = ToolFunction(name="test_func", description="Test")
    spec = ToolSpec(function=func)
    msg = Message(role="user", content="Hello")
    record = Record(id="rec_001", tools=[spec], messages=[msg])

    write_dataset_jsonl([record], output_path)

    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    data = json.loads(lines[0])
    assert data["id"] == "rec_001"
    assert len(data["tools"]) == 1
    assert len(data["messages"]) == 1


def test_write_dataset_jsonl_multiple_records(tmp_path: Path) -> None:
    """Test writing multiple records to JSONL."""
    output_path = tmp_path / "data.jsonl"

    func = ToolFunction(name="test_func")
    spec = ToolSpec(function=func)
    msg = Message(role="user", content="Test")

    records = [
        Record(id=f"rec_{i:03d}", tools=[spec], messages=[msg]) for i in range(5)
    ]

    write_dataset_jsonl(records, output_path)

    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5

    for i, line in enumerate(lines):
        data = json.loads(line)
        assert data["id"] == f"rec_{i:03d}"


def test_write_dataset_jsonl_creates_directory(tmp_path: Path) -> None:
    """Test that write_dataset_jsonl creates parent directories."""
    output_path = tmp_path / "nested" / "dir" / "data.jsonl"

    func = ToolFunction(name="test")
    spec = ToolSpec(function=func)
    msg = Message(role="user", content="Test")
    record = Record(id="rec_001", tools=[spec], messages=[msg])

    write_dataset_jsonl([record], output_path)

    assert output_path.exists()
    assert output_path.parent.exists()


def test_write_dataset_jsonl_empty_list(tmp_path: Path) -> None:
    """Test writing empty list creates empty file."""
    output_path = tmp_path / "empty.jsonl"

    write_dataset_jsonl([], output_path)

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert content == ""


def test_write_dataset_jsonl_exclude_none(tmp_path: Path) -> None:
    """Test that None values are excluded from output."""
    output_path = tmp_path / "data.jsonl"

    func = ToolFunction(name="test", description=None)
    spec = ToolSpec(function=func)
    msg = Message(role="user", content="Test", tool_call_id=None)
    record = Record(id="rec_001", tools=[spec], messages=[msg])

    write_dataset_jsonl([record], output_path)

    line = output_path.read_text(encoding="utf-8").strip()
    data = json.loads(line)

    # None values should be excluded
    assert "description" not in data["tools"][0]["function"]
    assert "tool_call_id" not in data["messages"][0]


def test_append_record_jsonl_new_file(tmp_path: Path) -> None:
    """Test appending record to new file."""
    output_path = tmp_path / "append.jsonl"

    func = ToolFunction(name="test")
    spec = ToolSpec(function=func)
    msg = Message(role="user", content="Test")
    record = Record(id="rec_001", tools=[spec], messages=[msg])

    append_record_jsonl(record, output_path)

    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    data = json.loads(lines[0])
    assert data["id"] == "rec_001"


def test_append_record_jsonl_existing_file(tmp_path: Path) -> None:
    """Test appending record to existing file."""
    output_path = tmp_path / "append.jsonl"

    func = ToolFunction(name="test")
    spec = ToolSpec(function=func)
    msg = Message(role="user", content="Test")

    # Write first record
    record1 = Record(id="rec_001", tools=[spec], messages=[msg])
    append_record_jsonl(record1, output_path)

    # Append second record
    record2 = Record(id="rec_002", tools=[spec], messages=[msg])
    append_record_jsonl(record2, output_path)

    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    data1 = json.loads(lines[0])
    data2 = json.loads(lines[1])
    assert data1["id"] == "rec_001"
    assert data2["id"] == "rec_002"


def test_append_record_jsonl_creates_directory(tmp_path: Path) -> None:
    """Test that append_record_jsonl creates parent directories."""
    output_path = tmp_path / "nested" / "append.jsonl"

    func = ToolFunction(name="test")
    spec = ToolSpec(function=func)
    msg = Message(role="user", content="Test")
    record = Record(id="rec_001", tools=[spec], messages=[msg])

    append_record_jsonl(record, output_path)

    assert output_path.exists()
    assert output_path.parent.exists()


def test_append_record_jsonl_unicode(tmp_path: Path) -> None:
    """Test appending record with Unicode characters."""
    output_path = tmp_path / "unicode.jsonl"

    func = ToolFunction(name="test_func", description="Test: 你好 мир 🌍")
    spec = ToolSpec(function=func)
    msg = Message(role="user", content="Unicode: café ñ ü")
    record = Record(id="rec_001", tools=[spec], messages=[msg])

    append_record_jsonl(record, output_path)

    line = output_path.read_text(encoding="utf-8").strip()
    data = json.loads(line)

    assert "你好" in data["tools"][0]["function"]["description"]
    assert "café" in data["messages"][0]["content"]
