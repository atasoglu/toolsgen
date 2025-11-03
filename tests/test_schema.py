"""Tests for schema validation."""

from toolsgen.schema import (
    AssistantToolCall,
    Message,
    Record,
    ToolFunction,
    ToolSpec,
)


def test_tool_function_creation() -> None:
    """Test creating a ToolFunction."""
    func = ToolFunction(
        name="test_func",
        description="Test function",
        parameters={"type": "object", "properties": {}},
    )
    assert func.name == "test_func"
    assert func.description == "Test function"


def test_tool_spec_creation() -> None:
    """Test creating a ToolSpec."""
    func = ToolFunction(name="test_func", description="Test")
    spec = ToolSpec(function=func)
    assert spec.type == "function"
    assert spec.function.name == "test_func"


def test_message_creation() -> None:
    """Test creating a Message."""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_assistant_tool_call_creation() -> None:
    """Test creating an AssistantToolCall."""
    call = AssistantToolCall(
        id="call_123", function={"name": "test_func", "arguments": '{"x": 1}'}
    )
    assert call.id == "call_123"
    assert call.type == "function"


def test_record_creation() -> None:
    """Test creating a Record."""
    func = ToolFunction(name="test_func")
    spec = ToolSpec(function=func)
    msg = Message(role="user", content="Test")

    record = Record(
        id="record_001",
        tools=[spec],
        messages=[msg],
    )

    assert record.id == "record_001"
    assert record.language == "en"
    assert len(record.tools) == 1
    assert len(record.messages) == 1
