"""LLM-as-a-judge scoring for tool-calling datasets.

Implements rubric-based scoring with three dimensions:
- Tool relevance (0-0.4)
- Argument plausibility & schema adherence (0-0.4)
- Response clarity & completeness (0-0.2)
Total score: 0-1.0, verdict: accept if score >= 0.7
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from ..providers.openai_compat import ChatModelClient
from ..prompts import create_judge_prompt, create_judge_user_message
from ..schema import AssistantToolCall, ToolSpec


class JudgeResponse(BaseModel):
    """Structured response from judge LLM.

    This model is used with OpenAI's structured outputs feature
    to ensure reliable parsing.
    """

    model_config = {"extra": "forbid"}  # Required for OpenAI structured outputs

    tool_relevance: float = Field(
        ...,
        ge=0.0,
        le=0.4,
        description="Tool relevance score (0.0-0.4)",
    )
    argument_quality: float = Field(
        ...,
        ge=0.0,
        le=0.4,
        description="Argument plausibility & schema adherence score (0.0-0.4)",
    )
    clarity: float = Field(
        ...,
        ge=0.0,
        le=0.2,
        description="Response clarity & completeness score (0.0-0.2)",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Total score (sum of dimensions, 0.0-1.0)",
    )
    verdict: Literal["accept", "reject"] = Field(
        ...,
        description="Accept if score >= 0.7, otherwise reject",
    )
    rationale: str = Field(
        ...,
        description="Brief explanation of the judgment",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Record.judge field."""
        return {
            **self.model_dump(),
            "rubric_version": "0.1.0",
        }


def judge_tool_calls(
    client: ChatModelClient,
    user_request: str,
    tools: List[ToolSpec],
    tool_calls: List[AssistantToolCall],
    temperature: float = 0.3,
) -> JudgeResponse:
    """Evaluate tool calls using LLM-as-a-judge.

    Args:
        client: LLM client for judging.
        user_request: The original user request.
        tools: Available tool specifications.
        tool_calls: Generated tool calls to evaluate.
        temperature: Sampling temperature (lower for more deterministic).

    Returns:
        JudgeResponse with scores and verdict.
    """
    if not tool_calls:
        return JudgeResponse(
            tool_relevance=0.0,
            argument_quality=0.0,
            clarity=0.0,
            score=0.0,
            verdict="reject",
            rationale="No tool calls provided",
        )

    prompt = create_judge_prompt(user_request, tools, tool_calls)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": create_judge_user_message()},
    ]

    return client.create_structured(
        messages=messages,
        response_model=JudgeResponse,
        temperature=temperature,
        max_tokens=500,
    )
