from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


class ChatModelClient:
    """Wrapper around the official OpenAI Python SDK (Chat Completions).

    Supports custom `base_url` via `OPENAI_BASE_URL` for OpenAI-compatible
    providers (Azure, OpenRouter, vLLM). Reads API key from `OPENAI_API_KEY`
    unless explicitly provided.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL")
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key not provided. Set OPENAI_API_KEY or pass api_key."
            )
        if resolved_base:
            self._client = OpenAI(api_key=resolved_key, base_url=resolved_base)
        else:
            self._client = OpenAI(api_key=resolved_key)

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
