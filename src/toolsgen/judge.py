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

    def generate_quality_tags(
        self,
        high_quality_threshold: float = 0.9,
        medium_quality_threshold: float = 0.7,
        excellent_dimension_pct: float = 0.875,
        poor_dimension_pct: float = 0.5,
    ) -> List[str]:
        """Generate quality tags based on scores.

        Args:
            high_quality_threshold: Overall score threshold for high_quality tag (default: 0.9).
            medium_quality_threshold: Overall score threshold for medium_quality tag (default: 0.7).
            excellent_dimension_pct: Percentage of max score for excellent tags (default: 0.875 = 87.5%).
            poor_dimension_pct: Percentage of max score for poor tags (default: 0.5 = 50%).

        Returns:
            List of quality tags describing the sample.
        """
        tags = []

        # Overall quality
        if self.score >= high_quality_threshold:
            tags.append("high_quality")
        elif self.score >= medium_quality_threshold:
            tags.append("medium_quality")
        else:
            tags.append("low_quality")

        # Dimension-specific tags (based on percentage of max possible score)
        tool_rel_excellent = 0.4 * excellent_dimension_pct
        tool_rel_poor = 0.4 * poor_dimension_pct
        if self.tool_relevance >= tool_rel_excellent:
            tags.append("excellent_tool_selection")
        elif self.tool_relevance < tool_rel_poor:
            tags.append("poor_tool_selection")

        arg_qual_excellent = 0.4 * excellent_dimension_pct
        arg_qual_poor = 0.4 * poor_dimension_pct
        if self.argument_quality >= arg_qual_excellent:
            tags.append("excellent_arguments")
        elif self.argument_quality < arg_qual_poor:
            tags.append("poor_arguments")

        clarity_excellent = 0.2 * excellent_dimension_pct
        clarity_poor = 0.2 * poor_dimension_pct
        if self.clarity >= clarity_excellent:
            tags.append("high_clarity")
        elif self.clarity < clarity_poor:
            tags.append("low_clarity")

        return tags


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
