"""
app/routers/chat.py
───────────────────
POST /v1/chat/completions — OpenAI-compatible chat endpoint.

Tri-Modal dispatch:
  Mode 1 (Deterministic) → direct_result set → skip LiteLLM, return instantly
  Mode 2 (Probabilistic) → direct_result None → call LiteLLM as before
  Mode 3 (Agentic)       → direct_result set → skip LiteLLM, return agent output
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.logger import get_logger, log_request_event
from app.models import (
    ChatRequest,
    ChatResponse,
    Choice,
    ChoiceMessage,
    ExecutionMode,
    RoutingMetadata,
    UsageInfo,
)
from app.proxy import ProxyError, call_llm
from app.embedding_engine import embed_text
from app.vector_store import store_routing_decision
from app.routing_graph import route_request

router = APIRouter()
logger = get_logger("router.chat")


@router.post(
    "/chat/completions",
    response_model=ChatResponse,
    summary="Chat Completions (Tri-Modal Orchestrated)",
    description=(
        "OpenAI-compatible chat completions endpoint. "
        "Requests are routed through a Tri-Modal Orchestrator: "
        "Deterministic (Mode 1), Probabilistic LLM (Mode 2), or "
        "Agentic multi-step (Mode 3)."
    ),
    tags=["Chat"],
)
async def chat_completions(
    body: ChatRequest,
    http_request: Request,
) -> ChatResponse:
    t_start    = time.perf_counter()
    request_id = http_request.headers.get(
        "X-Request-ID", f"req-{uuid.uuid4().hex[:8]}"
    )

    logger.info(
        "INCOMING | id=%s user=%s tier=%s messages=%d",
        request_id, body.user_id, body.user_tier, len(body.messages),
    )

    decision = await route_request(body)
    routing  = decision.routing
    routing.latency_ms = (time.perf_counter() - t_start) * 1000

    if decision.direct_result is not None:
        latency_ms = (time.perf_counter() - t_start) * 1000
        routing.latency_ms = latency_ms

        response = ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            model=routing.model_selected,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=decision.direct_result,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            routing_metadata=routing,
        )

        mode_label = (
            "MODE1_DETERMINISTIC"
            if routing.execution_mode == ExecutionMode.DETERMINISTIC
            else "MODE3_AGENTIC"
        )
        logger.info(
            "%s | id=%s model=%s saved=$%.6f latency=%.1fms",
            mode_label,
            request_id,
            routing.model_selected,
            routing.cost_saved_usd,
            latency_ms,
        )

        await log_request_event(
            request_id=request_id,
            user_id=body.user_id,
            model_used=routing.model_selected,
            decision_layer=routing.decision_layer,
            decision_reason=routing.decision_reason,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=len(decision.direct_result.split()),
            pii_detected=routing.pii_detected,
            user_tier=body.user_tier,
            pii_entities=routing.pii_entities,
            execution_mode=routing.execution_mode.value,
            cost_saved_usd=routing.cost_saved_usd,
        )

        return response

    # ── 2b. Mode 2 — call LLM via LiteLLM ────────────────────────────────────
    try:
        raw = await call_llm(body, routing.model_selected)
    except ProxyError as exc:
        logger.error(
            "ProxyError | id=%s status=%d msg=%s", request_id, exc.status_code, exc
        )
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    # ── 3. Build structured response ──────────────────────────────────────────
    latency_ms: float = raw.pop("_latency_ms", routing.latency_ms)
    routing.latency_ms = latency_ms

    raw_choices = raw.get("choices", [{}])
    choices = [
        Choice(
            index=c.get("index", 0),
            message=ChoiceMessage(
                role=c.get("message", {}).get("role", "assistant"),
                content=c.get("message", {}).get("content"),
            ),
            finish_reason=c.get("finish_reason"),
        )
        for c in raw_choices
    ]

    raw_usage = raw.get("usage") or {}
    usage = UsageInfo(
        prompt_tokens=raw_usage.get("prompt_tokens", 0),
        completion_tokens=raw_usage.get("completion_tokens", 0),
        total_tokens=raw_usage.get("total_tokens", 0),
    )

    response = ChatResponse(
        id=raw.get("id", request_id),
        model=routing.model_selected,
        choices=choices,
        usage=usage,
        routing_metadata=routing,
    )

    # ── 4. Async logging (console + SQLite) ───────────────────────────────────
    await log_request_event(
        request_id=request_id,
        user_id=body.user_id,
        model_used=routing.model_selected,
        decision_layer=routing.decision_layer,
        decision_reason=routing.decision_reason,
        latency_ms=latency_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        pii_detected=routing.pii_detected,
        user_tier=body.user_tier,
        pii_entities=routing.pii_entities,
        execution_mode=routing.execution_mode.value,
        cost_saved_usd=0.0,
    )

    # ── 5. Semantic feedback loop — store successful Mode 2 routes in Qdrant ──
    if (
        not routing.pii_detected
        and routing.decision_layer != "layer2_semantic"
    ):
        query_text = " ".join(
            msg.content for msg in body.messages
            if msg.role == "user" and msg.content
        )
        if query_text:
            try:
                embedding = await embed_text(query_text)
                if embedding is not None:
                    await store_routing_decision(
                        embedding=embedding,
                        query_text=query_text,
                        model_used=routing.model_selected,
                        decision_reason=routing.decision_reason,
                        user_tier=body.user_tier,
                    )
            except Exception as exc:
                logger.warning("Semantic feedback store failed: %s", exc)

    return response
