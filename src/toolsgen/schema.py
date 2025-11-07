from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class AssistantToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Dict[str, Any]


class Record(BaseModel):
    id: str
    language: str = "en"
    tools: List[ToolSpec]
    messages: List[Message]
    assistant_calls: List[AssistantToolCall] = Field(default_factory=list)
    problem_metadata: Dict[str, Any] = Field(default_factory=dict)
    judge: Dict[str, Any] = Field(default_factory=dict)
    quality_tags: List[str] = Field(default_factory=list)
    tools_metadata: Dict[str, Any] = Field(default_factory=dict)


class Manifest(BaseModel):
    """Metadata manifest for generated datasets.

    Attributes:
        version: Dataset format version.
        num_requested: Number of samples requested.
        num_generated: Number of samples successfully generated.
        num_failed: Number of failed generation attempts.
        strategy: Sampling strategy used.
        seed: Random seed for reproducibility.
        train_split: Fraction of data in training split.
        tools_count: Total number of available tools.
        models: Model names used for each generation role.
        splits: Sample counts per split (train/val).
    """

    version: str = "0.1.0"
    num_requested: int
    num_generated: int
    num_failed: int
    strategy: str
    seed: Optional[int] = None
    train_split: float = 1.0
    tools_count: int
    models: Dict[str, str] = Field(default_factory=dict)
    splits: Dict[str, int] = Field(default_factory=dict)
