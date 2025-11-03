"""Prompt templates for problem generation, tool calling, and judging."""

from __future__ import annotations

import json
from typing import List

from .schema import AssistantToolCall, ToolSpec


def create_problem_generation_prompt(tools: List[ToolSpec]) -> str:
    """Create a prompt for generating natural language user requests.

    Args:
        tools: List of available tools.

    Returns:
        System prompt for generating user requests.
    """
    tools_desc = []
    for tool in tools:
        name = tool.function.name
        desc = tool.function.description or "No description"
        tools_desc.append(f"- {name}: {desc}")

    tools_list = "\n".join(tools_desc)

    return f"""You are generating realistic user requests that would naturally lead to using these tools:

{tools_list}

Generate a natural, conversational English request that a user might make that would require using one or more of these tools. The request should be:
- Natural and conversational
- Specific enough to require tool calls
- Realistic for a real-world scenario
- Not explicitly mentioning tool names

Return only the user's request, nothing else."""


def create_problem_generation_user_message() -> str:
    """User message for problem generation prompt.

    Returns:
        User message content.
    """
    return "Generate a realistic user request that would require using these tools."


def create_caller_system_prompt() -> str:
    """System prompt for tool-calling assistant generation.

    Returns:
        System prompt for generating tool calls.
    """
    return "You are a helpful assistant that uses tools to answer user requests. Generate appropriate tool calls based on the user's request."


def create_judge_prompt(
    user_request: str,
    tools: List[ToolSpec],
    tool_calls: List[AssistantToolCall],
) -> str:
    """Create the judge prompt for evaluating tool calls.

    Args:
        user_request: The original user request.
        tools: Available tool specifications.
        tool_calls: Generated tool calls to evaluate.

    Returns:
        System prompt for the judge LLM.
    """
    tools_desc = []
    for tool in tools:
        name = tool.function.name
        desc = tool.function.description or "No description"
        params = tool.function.parameters
        tools_desc.append(f"- {name}: {desc} (params: {json.dumps(params, indent=2)})")

    tools_list = "\n".join(tools_desc)

    tool_calls_str = []
    for tc in tool_calls:
        func_info = tc.function
        name = func_info.get("name", "unknown")
        args = func_info.get("arguments", "{}")
        tool_calls_str.append(f"- {name}({args})")

    calls_list = "\n".join(tool_calls_str) if tool_calls_str else "None"

    return f"""You are an expert evaluator judging the quality of tool-calling responses.

**User Request:**
{user_request}

**Available Tools:**
{tools_list}

**Generated Tool Calls:**
{calls_list}

**Evaluation Rubric (v0.1.0):**

1. **Tool Relevance (0-0.4 points)**
   - Are the selected tools appropriate for the user's request?
   - Do the tools match the intent and context?
   - Score: 0.0 (irrelevant) to 0.4 (highly relevant)

2. **Argument Plausibility & Schema Adherence (0-0.4 points)**
   - Are the arguments provided to each tool call plausible and realistic?
   - Do the arguments conform to the tool's parameter schema?
   - Are required parameters present and correctly formatted?
   - Score: 0.0 (invalid/implausible) to 0.4 (excellent adherence)

3. **Response Clarity & Completeness (0-0.2 points)**
   - Does the response adequately address the user's request?
   - Are all necessary tool calls present?
   - Score: 0.0 (incomplete/unclear) to 0.2 (clear and complete)

**Output Format (JSON only):**
{{
  "tool_relevance": <float 0.0-0.4>,
  "argument_quality": <float 0.0-0.4>,
  "clarity": <float 0.0-0.2>,
  "score": <float 0.0-1.0>,
  "verdict": "<accept|reject>",
  "rationale": "<brief explanation>"
}}

Calculate the total score as the sum of the three dimensions.
Verdict should be "accept" if score >= 0.7, otherwise "reject".

Return ONLY valid JSON, no other text."""


def create_judge_user_message() -> str:
    """User message for judge prompt.

    Returns:
        User message content.
    """
    return "Evaluate the tool calls according to the rubric and return the JSON score."
