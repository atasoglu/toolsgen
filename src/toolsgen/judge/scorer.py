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


class JudgeResult:
    """Result from LLM-as-a-judge scoring.

    Attributes:
        score: Overall score (0.0-1.0)
        verdict: "accept" or "reject"
        rationale: Brief explanation of the judgment
        rubric_version: Version of the rubric used
        tool_relevance: Score for tool relevance (0-0.4)
        argument_quality: Score for argument plausibility (0-0.4)
        clarity: Score for clarity and completeness (0-0.2)
    """

    def __init__(
        self,
        score: float,
        verdict: str,
        rationale: str,
        tool_relevance: float = 0.0,
        argument_quality: float = 0.0,
        clarity: float = 0.0,
        rubric_version: str = "0.1.0",
    ) -> None:
        self.score = score
        self.verdict = verdict
        self.rationale = rationale
        self.tool_relevance = tool_relevance
        self.argument_quality = argument_quality
        self.clarity = clarity
        self.rubric_version = rubric_version

    @classmethod
    def from_response(cls, response: JudgeResponse) -> "JudgeResult":
        """Create JudgeResult from structured JudgeResponse.

        Args:
            response: Structured response from judge LLM.

        Returns:
            JudgeResult instance.
        """
        return cls(
            score=response.score,
            verdict=response.verdict,
            rationale=response.rationale,
            tool_relevance=response.tool_relevance,
            argument_quality=response.argument_quality,
            clarity=response.clarity,
            rubric_version="0.1.0",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Record.judge field."""
        return {
            "score": self.score,
            "verdict": self.verdict,
            "rationale": self.rationale,
            "rubric_version": self.rubric_version,
            "tool_relevance": self.tool_relevance,
            "argument_quality": self.argument_quality,
            "clarity": self.clarity,
        }


def judge_tool_calls(
    client: ChatModelClient,
    user_request: str,
    tools: List[ToolSpec],
    tool_calls: List[AssistantToolCall],
    temperature: float = 0.3,
) -> JudgeResult:
    """Evaluate tool calls using LLM-as-a-judge.

    Args:
        client: LLM client for judging.
        user_request: The original user request.
        tools: Available tool specifications.
        tool_calls: Generated tool calls to evaluate.
        temperature: Sampling temperature (lower for more deterministic).

    Returns:
        JudgeResult with scores and verdict.

    Raises:
        ValueError: If judge response cannot be parsed or is invalid.
    """
    if not tool_calls:
        return JudgeResult(
            score=0.0,
            verdict="reject",
            rationale="No tool calls provided",
            rubric_version="0.1.0",
        )

    prompt = create_judge_prompt(user_request, tools, tool_calls)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": create_judge_user_message()},
    ]

    # Use structured outputs with pydantic model
    response = client.create_structured(
        messages=messages,
        response_model=JudgeResponse,
        temperature=temperature,
        max_tokens=500,
    )

    # response is already a JudgeResponse pydantic model
    return JudgeResult.from_response(response)
