"""
app/proxy.py
────────────
LiteLLM async wrapper.

Responsibilities:
  - Translate ChatRequest → litellm.acompletion args
  - Return a normalised dict (OpenAI response shape)
  - Handle provider errors with clean exception types
"""

from __future__ import annotations

import time
from typing import Any

import litellm
from litellm import ModelResponse

from app.config import get_settings
from app.logger import get_logger
from app.models import ChatRequest

settings = get_settings()
logger = get_logger("router.proxy")

litellm.drop_params = True          # Ignore unsupported params per provider
litellm.set_verbose = False         # Silence litellm's own debug noise
litellm.num_retries = 2             # Automatic retry on transient failures
litellm.request_timeout = 60        # Seconds

# Propagate API keys from our settings so litellm picks them up
if settings.openai_api_key:
    litellm.openai_key = settings.openai_api_key
if settings.anthropic_api_key:
    litellm.anthropic_key = settings.anthropic_api_key
if settings.groq_api_key:
    litellm.groq_key = settings.groq_api_key


class ProxyError(Exception):
    """Raised when the upstream LLM call fails after retries."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


async def call_llm(request: ChatRequest, model: str) -> dict[str, Any]:
    """
    Call the specified model via LiteLLM and return the raw response dict.

    Args:
        request: The validated ChatRequest from the API layer.
        model:   The model string decided by the routing pipeline
                 (e.g. "gpt-4o-mini", "anthropic/claude-3-haiku-20240307").

    Returns:
        A dict matching the OpenAI chat completion response shape.

    Raises:
        ProxyError: On any upstream failure.
    """
    messages = [m.model_dump(exclude_none=True) for m in request.messages]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens

    logger.debug("Calling model=%s messages=%d", model, len(messages))
    t0 = time.perf_counter()

    try:
        response: ModelResponse = await litellm.acompletion(**kwargs)
    except litellm.exceptions.AuthenticationError as exc:
        raise ProxyError(f"Authentication failed for model '{model}': {exc}", 401) from exc
    except litellm.exceptions.RateLimitError as exc:
        raise ProxyError(f"Rate limited by '{model}': {exc}", 429) from exc
    except litellm.exceptions.NotFoundError as exc:
        raise ProxyError(f"Model '{model}' not found: {exc}", 404) from exc
    except litellm.exceptions.Timeout as exc:
        raise ProxyError(f"Timeout calling '{model}': {exc}", 504) from exc
    except Exception as exc:
        raise ProxyError(f"Unexpected error from '{model}': {exc}", 502) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("model=%s latency=%.1fms", model, elapsed_ms)

    result = response.model_dump()
    result["_latency_ms"] = elapsed_ms  # Internal field, stripped at response layer
    return result
