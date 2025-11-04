"""Tests for generator module."""

from pathlib import Path

import pytest

from toolsgen.core import load_tool_specs


def test_load_tool_specs(tmp_path: Path) -> None:
    """Test loading tool specs from JSON."""
    tools_file = tmp_path / "tools.json"
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
    import json

    tools_file.write_text(json.dumps(tools_data), encoding="utf-8")

    specs = load_tool_specs(tools_file)
    assert len(specs) == 1
    assert specs[0].function.name == "test_func"


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
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "func2",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    import json

    tools_file.write_text(json.dumps(tools_data), encoding="utf-8")

    specs = load_tool_specs(tools_file)
    assert len(specs) == 2
    assert specs[0].function.name == "func1"
    assert specs[1].function.name == "func2"
