"""Build complete dataset records through multi-stage LLM pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI

from ..judge import judge_tool_calls
from ..problem_generator import generate_problem
from ..schema import Message, Record, ToolSpec
from ..tool_caller import generate_tool_calls
from .client import create_openai_client
from .config import RoleBasedModelConfig


def _build_record(
    problem_client: OpenAI,
    caller_client: OpenAI,
    judge_client: OpenAI,
    record_id: str,
    tools: List[ToolSpec],
    role_config: RoleBasedModelConfig,
    language: str,
) -> Optional[Record]:
    """Execute three-stage pipeline: problem generation → tool calling → judging."""

    user_request = generate_problem(
        client=problem_client,
        model=role_config.problem_generator.model,
        tools=tools,
        language=language,
        temperature=role_config.problem_generator.temperature,
        max_tokens=role_config.problem_generator.max_tokens,
    )

    if not user_request:
        return None

    tool_calls = generate_tool_calls(
        client=caller_client,
        model=role_config.tool_caller.model,
        user_request=user_request,
        tools=tools,
        temperature=role_config.tool_caller.temperature,
        max_tokens=role_config.tool_caller.max_tokens,
    )

    if not tool_calls:
        return None

    judge_dict = {
        "model": role_config.judge.model,
        "temperature": role_config.judge.temperature,
    }
    quality_tags = []
    try:
        judge_result = judge_tool_calls(
            client=judge_client,
            model=role_config.judge.model,
            user_request=user_request,
            tools=tools,
            tool_calls=tool_calls,
            temperature=role_config.judge.temperature,
            max_tokens=role_config.judge.max_tokens,
        )
        judge_dict.update(judge_result.to_dict())
        quality_tags = judge_result.generate_quality_tags()
    except Exception:
        pass  # Continue without judge data

    return Record(
        id=record_id,
        language=language,
        tools=tools,
        messages=[Message(role="user", content=user_request)],
        assistant_calls=tool_calls,
        problem_metadata={"generated": True, "user_request": user_request},
        judge=judge_dict,
        quality_tags=quality_tags,
        tools_metadata={"num_tools": len(tools)},
    )


@dataclass
class RecordBuilder:
    """Build dataset records through multi-role LLM pipeline."""

    role_config: RoleBasedModelConfig
    language: str = "english"
    problem_client: Optional[OpenAI] = field(default=None, init=False)
    caller_client: Optional[OpenAI] = field(default=None, init=False)
    judge_client: Optional[OpenAI] = field(default=None, init=False)

    def ensure_clients(self) -> None:
        """Create OpenAI clients on-demand."""

        if self.problem_client is None:
            self.problem_client = create_openai_client(
                self.role_config.problem_generator
            )
        if self.caller_client is None:
            self.caller_client = create_openai_client(self.role_config.tool_caller)
        if self.judge_client is None:
            self.judge_client = create_openai_client(self.role_config.judge)

    def generate_record(
        self, record_id: str, tools: List[ToolSpec]
    ) -> Optional[Record]:
        """Generate a single record; returns None if any stage fails."""

        self.ensure_clients()
        if (
            self.problem_client is None
            or self.caller_client is None
            or self.judge_client is None
        ):
            return None

        return _build_record(
            self.problem_client,
            self.caller_client,
            self.judge_client,
            record_id,
            tools,
            self.role_config,
            self.language,
        )
