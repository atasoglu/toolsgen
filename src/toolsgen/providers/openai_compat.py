from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ChatModelClient:
    """Wrapper around the official OpenAI Python SDK (Chat Completions).

    Supports custom `base_url` via `OPENAI_BASE_URL` for OpenAI-compatible
    providers (Azure, OpenRouter, vLLM). Reads API key from `OPENAI_API_KEY`
    unless explicitly provided.

    Features:
    - Automatic retries with exponential backoff
    - Simple rate limiting (token bucket)
    - Structured error handling
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the chat model client.

        Args:
            model: Model name to use.
            base_url: Optional base URL for API (overrides OPENAI_BASE_URL).
            api_key: Optional API key (overrides OPENAI_API_KEY).
        """
        self.model = model
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL")
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key not provided. Set OPENAI_API_KEY or pass api_key."
            )
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base,
            max_retries=3,
            timeout=60.0,
        )

    def _dump(self, obj: Any) -> Dict[str, Any]:
        # SDK objects provide model_dump (pydantic v2) or to_dict depending on version
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        try:
            import json

            return json.loads(obj.json())
        except Exception:  # pragma: no cover
            return dict(obj)

    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a chat completion with retries and rate limiting.

        Args:
            messages: List of message dictionaries.
            tools: Optional list of tool definitions.
            tool_choice: Optional tool choice strategy.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            seed: Optional random seed.
            extra: Optional additional parameters.

        Returns:
            API response dictionary.
        """
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if seed is not None:
            params["seed"] = seed
        if extra:
            params.update(extra)

        resp = self._client.chat.completions.create(**params)
        return self._dump(resp)

    def create_structured(
        self,
        *,
        messages: List[Dict[str, Any]],
        response_model: Type[T],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> T:
        """Create a chat completion with structured output.

        Uses OpenAI's structured outputs feature (response_format with JSON schema)
        to ensure the response conforms to a pydantic model.

        Args:
            messages: List of message dictionaries.
            response_model: Pydantic model class for response structure.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            seed: Optional random seed.

        Returns:
            Instance of response_model populated with API response.
        """
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                },
            },
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if seed is not None:
            params["seed"] = seed

        resp = self._client.chat.completions.create(**params)

        # Extract content and parse into pydantic model
        choices = self._dump(resp).get("choices", [])
        if not choices:
            raise ValueError("LLM returned no choices")

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise ValueError("LLM returned empty content")

        return response_model.model_validate_json(content)
