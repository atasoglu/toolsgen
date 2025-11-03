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
    language: Literal["en"] = "en"
    tools: List[ToolSpec]
    messages: List[Message]
    assistant_calls: List[AssistantToolCall] = Field(default_factory=list)
    problem_metadata: Dict[str, Any] = Field(default_factory=dict)
    judge: Dict[str, Any] = Field(default_factory=dict)
    quality_tags: List[str] = Field(default_factory=list)
