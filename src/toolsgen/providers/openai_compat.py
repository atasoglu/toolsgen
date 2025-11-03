from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import OpenAI
from openai._exceptions import APIError, RateLimitError
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class RetryableError(Exception):
    """Base exception for retryable errors."""

    pass


class RateLimitExceeded(RetryableError):
    """Rate limit exceeded error."""

    pass


class APIRequestError(RetryableError):
    """API request error."""

    pass


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
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        rate_limit_tokens_per_minute: Optional[int] = None,
    ) -> None:
        """Initialize the chat model client.

        Args:
            model: Model name to use.
            base_url: Optional base URL for API (overrides OPENAI_BASE_URL).
            api_key: Optional API key (overrides OPENAI_API_KEY).
            max_retries: Maximum number of retry attempts.
            initial_retry_delay: Initial delay between retries in seconds.
            max_retry_delay: Maximum delay between retries in seconds.
            rate_limit_tokens_per_minute: Optional rate limit in tokens/minute.
        """
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

        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay

        # Simple rate limiting: token bucket
        self.rate_limit_tokens_per_minute = rate_limit_tokens_per_minute
        self._tokens_used = 0
        self._rate_limit_window_start = time.time()

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should be retried.

        Args:
            error: The exception that occurred.

        Returns:
            True if the error should be retried.
        """
        if isinstance(error, RateLimitError):
            return True
        if isinstance(error, APIError):
            # Retry on 5xx server errors
            status_code = getattr(error, "status_code", None)
            if status_code and 500 <= status_code < 600:
                return True
        return False

    def _wait_for_rate_limit(self, estimated_tokens: int) -> None:
        """Wait if rate limit would be exceeded.

        Args:
            estimated_tokens: Estimated number of tokens for the request.
        """
        if not self.rate_limit_tokens_per_minute:
            return

        current_time = time.time()
        window_duration = 60.0  # 1 minute window

        # Reset window if expired
        if current_time - self._rate_limit_window_start >= window_duration:
            self._tokens_used = 0
            self._rate_limit_window_start = current_time

        # Check if we would exceed rate limit
        if self._tokens_used + estimated_tokens > self.rate_limit_tokens_per_minute:
            # Wait until window resets
            wait_time = window_duration - (current_time - self._rate_limit_window_start)
            if wait_time > 0:
                time.sleep(wait_time)
                self._tokens_used = 0
                self._rate_limit_window_start = time.time()

        self._tokens_used += estimated_tokens

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

        Raises:
            RateLimitExceeded: If rate limit is exceeded after retries.
            APIRequestError: If API request fails after retries.
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

        # Estimate tokens for rate limiting (rough heuristic: ~4 chars per token)
        estimated_tokens = (
            sum(len(str(msg.get("content", ""))) for msg in messages) // 4
        )
        if max_tokens:
            estimated_tokens += max_tokens

        # Rate limiting
        self._wait_for_rate_limit(estimated_tokens)

        # Retry logic with exponential backoff
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**params)
                return self._dump(resp)
            except (RateLimitError, APIError) as e:
                last_error = e
                if not self._should_retry(e) or attempt >= self.max_retries:
                    break

                # Exponential backoff
                delay = min(
                    self.initial_retry_delay * (2**attempt),
                    self.max_retry_delay,
                )

                # For rate limit errors, use retry-after header if available
                if isinstance(e, RateLimitError):
                    response = getattr(e, "response", None)
                    if response and hasattr(response, "headers"):
                        retry_after = response.headers.get("retry-after")
                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except (ValueError, TypeError):
                                pass

                time.sleep(delay)
            except Exception as e:
                # Non-retryable error
                raise APIRequestError(f"API request failed: {e}") from e

        # All retries exhausted
        if isinstance(last_error, RateLimitError):
            raise RateLimitExceeded(
                f"Rate limit exceeded after {self.max_retries} retries"
            ) from last_error
        if isinstance(last_error, APIError):
            raise APIRequestError(
                f"API request failed after {self.max_retries} retries: {last_error}"
            ) from last_error

        raise APIRequestError("Unknown error occurred") from last_error

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

        Raises:
            RateLimitExceeded: If rate limit is exceeded after retries.
            APIRequestError: If API request fails after retries.
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

        # Estimate tokens for rate limiting
        estimated_tokens = (
            sum(len(str(msg.get("content", ""))) for msg in messages) // 4
        )
        if max_tokens:
            estimated_tokens += max_tokens

        # Rate limiting
        self._wait_for_rate_limit(estimated_tokens)

        # Retry logic with exponential backoff
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**params)

                # Extract content and parse into pydantic model
                choices = self._dump(resp).get("choices", [])
                if not choices:
                    raise ValueError("LLM returned no choices")

                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError("LLM returned empty content")

                # Parse JSON into pydantic model
                return response_model.model_validate_json(content)

            except (RateLimitError, APIError) as e:
                last_error = e
                if not self._should_retry(e) or attempt >= self.max_retries:
                    break

                # Exponential backoff
                delay = min(
                    self.initial_retry_delay * (2**attempt),
                    self.max_retry_delay,
                )

                # For rate limit errors, use retry-after header if available
                if isinstance(e, RateLimitError):
                    response = getattr(e, "response", None)
                    if response and hasattr(response, "headers"):
                        retry_after = response.headers.get("retry-after")
                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except (ValueError, TypeError):
                                pass

                time.sleep(delay)
            except Exception as e:
                # Non-retryable error
                raise APIRequestError(f"API request failed: {e}") from e

        # All retries exhausted
        if isinstance(last_error, RateLimitError):
            raise RateLimitExceeded(
                f"Rate limit exceeded after {self.max_retries} retries"
            ) from last_error
        if isinstance(last_error, APIError):
            raise APIRequestError(
                f"API request failed after {self.max_retries} retries: {last_error}"
            ) from last_error

        raise APIRequestError("Unknown error occurred") from last_error
