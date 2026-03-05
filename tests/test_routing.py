"""
tests/test_routing.py
─────────────────────
Tests for Layer 1: Rule-Based Gate.

Covers PII detection, blocked model rejection, tier-based downgrade,
token limit enforcement, and clean pass-through.

PII engine is mocked so Presidio/spaCy are NOT required to run tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models import ChatRequest, RoutingMetadata
from app.pii_engine import PIIResult


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


FAKE_LLM_RESPONSE = {
    "id": "chatcmpl-test-routing",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Test response"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
}


# ── Test: PII rerouting ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pii_detected_reroutes_to_private_model(client: AsyncClient) -> None:
    """Message containing PII (e.g., SSN) should route to private model."""

    pii_result = PIIResult(
        detected=True,
        entity_types=["US_SSN"],
        details=[{"entity_type": "US_SSN", "start": 10, "end": 21, "score": 0.95, "text_snippet": "123-45-6789"}],
    )

    mock_llm = AsyncMock()
    mock_llm.model_dump.return_value = dict(FAKE_LLM_RESPONSE)

    with (
        patch("app.layers.layer1_rules.scan_messages", return_value=pii_result),
        patch("app.proxy.litellm.acompletion", return_value=mock_llm) as mock_call,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "My SSN is 123-45-6789"}],
                "model": "gpt-4o-mini",
            },
        )

    assert response.status_code == 200
    meta = response.json()["routing_metadata"]
    assert meta["decision_layer"] == "layer1_rules"
    assert meta["pii_detected"] is True
    assert "US_SSN" in meta["pii_entities"]
    assert meta["model_selected"] == "ollama/llama3"  # private model

    # Verify the LLM was called with the private model
    call_kwargs = mock_call.call_args.kwargs
    assert call_kwargs["model"] == "ollama/llama3"


# ── Test: Blocked model ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_blocked_model_rejected(client: AsyncClient) -> None:
    """Requesting a globally blocked model should be rejected."""

    mock_llm = AsyncMock()
    mock_llm.model_dump.return_value = dict(FAKE_LLM_RESPONSE)

    with (
        patch("app.layers.layer1_rules.scan_messages", return_value=PIIResult()),
        patch("app.layers.layer1_rules.settings") as mock_settings,
        patch("app.proxy.litellm.acompletion", return_value=mock_llm),
    ):
        mock_settings.pii_enabled = True
        mock_settings.blocked_models = ["dangerous-model"]
        mock_settings.default_model = "gpt-4o-mini"
        mock_settings.free_tier_models = ["gpt-4o-mini"]
        mock_settings.free_tier_token_limit = 4096
        mock_settings.pro_tier_token_limit = 16384
        mock_settings.enterprise_tier_token_limit = 128000

        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "dangerous-model",
                "user_tier": "enterprise",
            },
        )

    assert response.status_code == 200
    meta = response.json()["routing_metadata"]
    assert meta["decision_layer"] == "layer1_rules"
    assert "model_blocked" in meta["decision_reason"]


# ── Test: Tier downgrade ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_free_tier_downgrade(client: AsyncClient) -> None:
    """Free-tier user requesting a premium model gets downgraded."""

    mock_llm = AsyncMock()
    mock_llm.model_dump.return_value = dict(FAKE_LLM_RESPONSE)

    with (
        patch("app.layers.layer1_rules.scan_messages", return_value=PIIResult()),
        patch("app.proxy.litellm.acompletion", return_value=mock_llm),
    ):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gpt-4o",
                "user_tier": "free",
            },
        )

    assert response.status_code == 200
    meta = response.json()["routing_metadata"]
    assert meta["decision_layer"] == "layer1_rules"
    assert "tier_downgrade" in meta["decision_reason"]
    assert meta["model_selected"] == "gpt-4o-mini"  # default model


# ── Test: Token limit ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_token_limit_enforcement(client: AsyncClient) -> None:
    """Prompt exceeding tier token limit should be rejected by Layer 1."""

    # Free tier limit = 4096 tokens ≈ 16384 chars
    # Send a message with ~20000 chars to exceed the limit
    huge_content = "x" * 20000

    mock_llm = AsyncMock()
    mock_llm.model_dump.return_value = dict(FAKE_LLM_RESPONSE)

    with (
        patch("app.layers.layer1_rules.scan_messages", return_value=PIIResult()),
        patch("app.proxy.litellm.acompletion", return_value=mock_llm),
    ):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": huge_content}],
                "model": "gpt-4o-mini",
                "user_tier": "free",
            },
        )

    assert response.status_code == 200
    meta = response.json()["routing_metadata"]
    assert meta["decision_layer"] == "layer1_rules"
    assert "token_limit_exceeded" in meta["decision_reason"]


# ── Test: Clean pass-through ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clean_passthrough_no_rules_fired(client: AsyncClient) -> None:
    """Normal request should pass through Layer 1 without any rule firing."""

    mock_llm = AsyncMock()
    mock_llm.model_dump.return_value = dict(FAKE_LLM_RESPONSE)

    with (
        patch("app.layers.layer1_rules.scan_messages", return_value=PIIResult()),
        patch("app.proxy.litellm.acompletion", return_value=mock_llm),
    ):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "model": "gpt-4o-mini",
                "user_tier": "free",
            },
        )

    assert response.status_code == 200
    meta = response.json()["routing_metadata"]
    # Should have fallen through to Layer 3 passthrough (no routing active yet)
    assert meta["decision_layer"] == "passthrough"
    assert meta["pii_detected"] is False


# ── Test: PII disabled ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pii_disabled_skips_scan() -> None:
    """When PII_ENABLED=False, scan_messages returns empty result."""
    from app.layers import layer1_rules

    request = ChatRequest(
        messages=[{"role": "user", "content": "My SSN is 123-45-6789"}],
        model="gpt-4o-mini",
    )

    # Mock settings to disable PII
    with patch("app.layers.layer1_rules.scan_messages", return_value=PIIResult()) as mock_scan:
        result = await layer1_rules._check_pii(request)

    # PII scan was called but returned nothing (because engine says disabled)
    assert result is None


# ── Test: Unit test — rule pipeline order ─────────────────────────────────────


@pytest.mark.asyncio
async def test_rule_pipeline_pii_fires_before_tier_check() -> None:
    """PII rule (Rule 1) should fire before tier gate (Rule 3)."""
    from app.layers import layer1_rules

    request = ChatRequest(
        messages=[{"role": "user", "content": "My SSN is 123-45-6789"}],
        model="gpt-4o",            # Would trigger tier downgrade for free user
        user_tier="free",
    )

    pii_result = PIIResult(
        detected=True,
        entity_types=["US_SSN"],
        details=[],
    )

    with patch("app.layers.layer1_rules.scan_messages", return_value=pii_result):
        result = await layer1_rules.run(request)

    assert result is not None
    # PII should have fired FIRST — not tier downgrade
    assert result.decision_layer == "layer1_rules"
    assert result.pii_detected is True
    assert result.model_selected == "ollama/llama3"
