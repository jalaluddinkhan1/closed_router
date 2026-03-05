"""
tests/test_foundation.py
────────────────────────
Integration tests for the system foundation and LiteLLM proxy.

Uses httpx AsyncClient with the FastAPI test app.
LiteLLM is patched so NO real API keys are needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# Fake LiteLLM response matching the ModelResponse structure
FAKE_LLM_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18,
    },
}


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """GET /health should return 200 with status=ok."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "components" in data


@pytest.mark.asyncio
async def test_chat_completions_passthrough(client: AsyncClient) -> None:
    """POST /v1/chat/completions should return a valid ChatResponse."""

    mock_response = AsyncMock()
    mock_response.model_dump.return_value = dict(FAKE_LLM_RESPONSE)

    with patch("app.proxy.litellm.acompletion", return_value=mock_response):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello."}],
                "model": "gpt-4o-mini",
                "user_tier": "free",
            },
        )

    assert response.status_code == 200
    data = response.json()

    # Validate OpenAI-compatible fields
    assert "id" in data
    assert data["model"] == "gpt-4o-mini"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] == "Hello! How can I help?"

    # Validate routing metadata
    meta = data["routing_metadata"]
    assert meta["decision_layer"] == "passthrough"
    assert meta["decision_reason"] == "default_passthrough_active"
    assert meta["model_selected"] == "gpt-4o-mini"
    assert meta["pii_detected"] is False


@pytest.mark.asyncio
async def test_chat_completions_uses_default_model(client: AsyncClient) -> None:
    """When no model is provided, the router should use DEFAULT_MODEL."""

    mock_response = AsyncMock()
    payload = dict(FAKE_LLM_RESPONSE)
    mock_response.model_dump.return_value = payload

    with patch("app.proxy.litellm.acompletion", return_value=mock_response) as mock_call:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 200
    # The model forwarded to litellm should be the default
    call_kwargs = mock_call.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_chat_completions_upstream_error(client: AsyncClient) -> None:
    """A ProxyError from litellm should surface as the correct HTTP status."""
    import litellm as _litellm

    with patch(
        "app.proxy.litellm.acompletion",
        side_effect=_litellm.exceptions.AuthenticationError(
            "bad key", llm_provider="openai", model="gpt-4o-mini"
        ),
    ):
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Test"}], "model": "gpt-4o-mini"},
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_invalid_request_missing_messages(client: AsyncClient) -> None:
    """Request with no messages field should return 422 Unprocessable Entity."""
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_routing_metadata_always_present(client: AsyncClient) -> None:
    """routing_metadata must always be present regardless of routing path."""

    mock_response = AsyncMock()
    mock_response.model_dump.return_value = dict(FAKE_LLM_RESPONSE)

    with patch("app.proxy.litellm.acompletion", return_value=mock_response):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test routing metadata"}],
                "model": "gpt-4o-mini",
                "user_tier": "enterprise",
                "tags": {"env": "test"},
            },
        )

    assert response.status_code == 200
    meta = response.json()["routing_metadata"]
    required_fields = {
        "decision_layer", "decision_reason", "model_selected",
        "confidence", "latency_ms", "pii_detected", "pii_entities",
    }
    assert required_fields.issubset(meta.keys())
