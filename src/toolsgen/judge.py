"""LLM-as-a-judge scoring for tool-calling datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from .prompts import create_judge_system_prompt, create_judge_user_prompt
from .schema import AssistantToolCall, ToolSpec


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
    client: OpenAI,
    model: str,
    user_request: str,
    tools: List[ToolSpec],
    tool_calls: List[AssistantToolCall],
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> JudgeResponse:
    """Evaluate tool calls using LLM-as-a-judge.

    Args:
        client: OpenAI client for judging.
        model: Model name to use.
        user_request: The original user request.
        tools: Available tool specifications.
        tool_calls: Generated tool calls to evaluate.
        temperature: Sampling temperature (lower for more deterministic).
        max_tokens: Optional maximum tokens to generate.

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

    system_prompt = create_judge_system_prompt(user_request, tools, tool_calls)
    user_prompt = create_judge_user_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": JudgeResponse.__name__,
                "schema": JudgeResponse.model_json_schema(),
                "strict": True,
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Judge LLM returned empty content")

    return JudgeResponse.model_validate_json(content)
