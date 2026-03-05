"""
app/models.py
─────────────
Pydantic V2 request / response schemas for the LLM Router API.
All models are OpenAI-compatible to allow drop-in replacement.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    """
    The three compute paradigms of the Tri-Modal Orchestrator.
      Mode 1 — Deterministic: Code/math/regex. $0 cost, <50ms, 100% accuracy.
      Mode 2 — Probabilistic: Single LLM call. Low cost, 1-3s latency.
      Mode 3 — Agentic:       Multi-step agent with tools. High quality, 10-30s.
    """
    DETERMINISTIC = "mode1_deterministic"
    PROBABILISTIC = "mode2_probabilistic"
    AGENTIC       = "mode3_agentic"


class ChatMessage(BaseModel):
    """A single turn in a conversation."""

    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str
    name: str | None = None  # Optional display name (OpenAI spec)


class ChatRequest(BaseModel):
    """
    OpenAI-compatible chat completions request, extended with router fields.
    """

    messages: list[ChatMessage]
    model: str | None = None  # If None, router will decide
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)

    # ── Router extensions ────────────────────────────────────────────────────
    user_tier: Literal["free", "pro", "enterprise"] = "free"
    user_id: str = Field(default_factory=lambda: f"anon-{uuid.uuid4().hex[:8]}")
    tags: dict[str, str] = Field(default_factory=dict)


class RoutingMetadata(BaseModel):
    """Explains how the orchestrator made its compute-paradigm + model decision."""

    decision_layer: Literal[
        # Safety gate
        "layer1_rules",
        # Model-selector (legacy tri-modal, still used for Mode 2 selection)
        "layer2_semantic",
        "layer3_agent",
        "passthrough",
        # Tri-modal cascade classifier layers
        "heuristic_layer1",
        "semantic_layer2",
        "llm_classifier_layer3",
        "default_no_content",
        # MoE Router
        "moe_router",
    ]
    decision_reason: str
    model_selected: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    latency_ms: float = 0.0
    pii_detected: bool = False
    pii_entities: list[str] = Field(default_factory=list)

    execution_mode: ExecutionMode = ExecutionMode.PROBABILISTIC
    cost_saved_usd: float = 0.0
    verifier_passed: bool | None = None

    experts_selected: list[str] = Field(default_factory=list)
    experts_scores: dict[str, float] = Field(default_factory=dict)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    """OpenAI-compatible response, enriched with routing_metadata."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    routing_metadata: RoutingMetadata


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str = "1.0.0"
    components: dict[str, Any] = Field(default_factory=dict)

