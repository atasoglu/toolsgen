"""Tests for schema validation."""

from toolsgen.schema import (
    AssistantToolCall,
    Manifest,
    Message,
    Record,
    ToolFunction,
    ToolSpec,
)


def test_tool_function() -> None:
    """Test ToolFunction creation with minimal and full fields."""
    # Minimal - test defaults
    func_min = ToolFunction(name="test")
    assert func_min.name == "test"
    assert func_min.description is None
    assert func_min.parameters == {}

    # Full - test all fields
    func_full = ToolFunction(
        name="full_func",
        description="Full function",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    assert func_full.name == "full_func"
    assert func_full.description == "Full function"
    assert "x" in func_full.parameters["properties"]


def test_tool_spec() -> None:
    """Test ToolSpec creation and defaults."""
    func = ToolFunction(name="test_func", description="Test")
    spec = ToolSpec(function=func)
    assert spec.type == "function"
    assert spec.function.name == "test_func"


def test_message() -> None:
    """Test Message creation with various roles and optional fields."""
    # Test all valid roles
    from typing import Literal

    roles: list[Literal["system", "user", "assistant", "tool"]] = [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    for role in roles:
        msg = Message(role=role, content="Test")
        assert msg.role == role

    # Test optional fields
    msg_full = Message(
        role="tool", content=None, tool_call_id="call_123", name="test_tool"
    )
    assert msg_full.tool_call_id == "call_123"
    assert msg_full.name == "test_tool"
    assert msg_full.content is None

    # Test None defaults
    msg_min = Message(role="user", content="Test")
    assert msg_min.tool_call_id is None
    assert msg_min.name is None


def test_assistant_tool_call() -> None:
    """Test AssistantToolCall creation."""
    call = AssistantToolCall(
        id="call_123",
        function={
            "name": "create_user",
            "arguments": '{"name": "John", "email": "test@example.com"}',
        },
    )
    assert call.id == "call_123"
    assert call.type == "function"
    assert call.function["name"] == "create_user"
    assert "John" in call.function["arguments"]


def test_record() -> None:
    """Test Record creation with defaults and all fields."""
    func = ToolFunction(name="test_func")
    spec = ToolSpec(function=func)
    msg = Message(role="user", content="Test")

    # Test defaults
    record_min = Record(id="rec_001", tools=[spec], messages=[msg])
    assert record_min.id == "rec_001"
    assert record_min.language == "en"
    assert len(record_min.tools) == 1
    assert len(record_min.messages) == 1
    assert record_min.assistant_calls == []
    assert record_min.problem_metadata == {}
    assert record_min.judge == {}
    assert record_min.quality_tags == []

    # Test all fields
    tool_call = AssistantToolCall(
        id="call_1", function={"name": "test_func", "arguments": "{}"}
    )
    record_full = Record(
        id="rec_002",
        language="es",
        tools=[spec],
        messages=[msg],
        assistant_calls=[tool_call],
        problem_metadata={"difficulty": "easy"},
        judge={"score": 0.9},
        quality_tags=["verified"],
    )
    assert record_full.language == "es"
    assert len(record_full.assistant_calls) == 1
    assert record_full.problem_metadata["difficulty"] == "easy"


def test_manifest() -> None:
    """Test Manifest creation with defaults and all fields."""
    # Test defaults
    manifest_min = Manifest(
        num_requested=10,
        num_generated=10,
        num_failed=0,
        strategy="random",
        tools_count=5,
    )
    assert manifest_min.version == "0.1.0"
    assert manifest_min.seed is None
    assert manifest_min.train_split == 1.0
    assert manifest_min.models == {}
    assert manifest_min.splits == {}

    # Test all fields
    manifest_full = Manifest(
        num_requested=10,
        num_generated=10,
        num_failed=0,
        strategy="param_aware",
        seed=42,
        train_split=0.8,
        tools_count=20,
        models={"problem_generator": "gpt-4"},
        splits={"train": 8, "val": 2},
    )
    assert manifest_full.seed == 42
    assert manifest_full.train_split == 0.8
    assert manifest_full.models["problem_generator"] == "gpt-4"
    assert manifest_full.splits["train"] == 8


def test_model_serialization() -> None:
    """Test serialization with exclude_none option."""
    func = ToolFunction(name="test", description=None)
    spec = ToolSpec(function=func)
    msg = Message(role="user", content="Test", tool_call_id=None)
    record = Record(id="rec_001", tools=[spec], messages=[msg])

    data = record.model_dump(exclude_none=True)

    # None values should be excluded
    assert "description" not in data["tools"][0]["function"]
    assert "tool_call_id" not in data["messages"][0]
