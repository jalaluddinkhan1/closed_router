"""
app/layers/layer3_agent.py
──────────────────────────
Layer 3: Agent Router powered by LangGraph.

This is the final layer in the Hybrid Funnel. It uses a small, cheap LLM
(e.g. gpt-4o-mini) to analyse the user's query and select the optimal
model from a registry of available models.

The LLM is prompted with:
  - The user query (truncated for cost)
  - The user's tier
  - A model catalogue with capabilities and cost tiers
  - Explicit instructions to return a structured JSON decision

If the LLM call or JSON parsing fails, the layer falls back to the
user-supplied model or DEFAULT_MODEL — it never raises an error.
"""

from __future__ import annotations

import json
from typing import Any

import litellm

from app.config import get_settings
from app.logger import get_logger
from app.models import ChatRequest, RoutingMetadata

settings = get_settings()
logger = get_logger("router.layer3")


MODEL_CATALOGUE: list[dict[str, Any]] = [
    {
        "id": "gpt-4o-mini",
        "provider": "openai",
        "strengths": "Fast, cheap, great for simple Q&A, summarisation, classification",
        "cost_tier": "low",
        "context_window": 128000,
    },
    {
        "id": "gpt-4o",
        "provider": "openai",
        "strengths": "Strong reasoning, coding, complex analysis, multi-step tasks",
        "cost_tier": "high",
        "context_window": 128000,
    },
    {
        "id": "anthropic/claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "strengths": "Excellent writing, nuanced reasoning, long-form content, safety",
        "cost_tier": "high",
        "context_window": 200000,
    },
    {
        "id": "anthropic/claude-3-haiku-20240307",
        "provider": "anthropic",
        "strengths": "Fast, cheap, good for simple tasks and classification",
        "cost_tier": "low",
        "context_window": 200000,
    },
    {
        "id": "groq/llama-3.1-8b-instant",
        "provider": "groq",
        "strengths": "Ultra-fast inference, good for simple tasks, open-source",
        "cost_tier": "very_low",
        "context_window": 131072,
    },
    {
        "id": "groq/llama-3.1-70b-versatile",
        "provider": "groq",
        "strengths": "Strong open-source model, good reasoning, fast via Groq",
        "cost_tier": "medium",
        "context_window": 131072,
    },
]


def _build_catalogue_text() -> str:
    """Render the model catalogue as a compact string for the prompt."""
    lines = []
    for m in MODEL_CATALOGUE:
        lines.append(
            f"- {m['id']} | cost={m['cost_tier']} | {m['strengths']}"
        )
    return "\n".join(lines)


ROUTER_SYSTEM_PROMPT = f"""\
You are a MODEL ROUTER. Your job is to select the best LLM from the catalogue
below to handle the user's query. You must consider:

1. **Query complexity**: Simple factual Q&A → cheap model. Complex reasoning,
   coding, multi-step analysis → powerful model.
2. **User tier**: "free" users should get cost-effective models. "pro" and
   "enterprise" users can use premium models.
3. **Task type**: Classify the query into one of: simple_qa, coding,
   creative_writing, analysis, summarisation, translation, math, other.

## Model Catalogue
{_build_catalogue_text()}

## Response Format
Respond with ONLY a JSON object (no markdown fences, no explanation):
{{
  "selected_model": "<model_id from catalogue>",
  "task_type": "<one of: simple_qa, coding, creative_writing, analysis, summarisation, translation, math, other>",
  "reasoning": "<one-line explanation>",
  "confidence": <float 0.0-1.0>
}}
"""


async def _call_router_llm(
    query_text: str, user_tier: str
) -> dict[str, Any] | None:
    """
    Call the small router LLM and parse its JSON response.
    Returns the parsed dict or None on any failure.
    """
    user_prompt = (
        f"User tier: {user_tier}\n"
        f"Query: {query_text[:2000]}"  # Truncate to save tokens
    )

    try:
        response = await litellm.acompletion(
            model=settings.agent_router_model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},  # Force JSON (OpenAI-compatible)
        )

        raw_content = response.choices[0].message.content
        if not raw_content:
            logger.warning("Layer3: router LLM returned empty content")
            return None

        # Parse JSON — strip markdown fences if the model wraps them
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        decision = json.loads(cleaned)
        logger.debug("Layer3 agent decision: %s", decision)
        return decision

    except json.JSONDecodeError as exc:
        logger.warning("Layer3: failed to parse router LLM response as JSON: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Layer3: router LLM call failed: %s", exc)
        return None


def _validate_model_selection(decision: dict[str, Any]) -> str | None:
    """
    Validate that the selected model exists in our catalogue.
    Returns the model ID or None if invalid.
    """
    selected = decision.get("selected_model", "")
    valid_ids = {m["id"] for m in MODEL_CATALOGUE}
    if selected in valid_ids:
        return selected
    # Try case-insensitive match
    for mid in valid_ids:
        if mid.lower() == selected.lower():
            return mid
    return None


async def run(request: ChatRequest) -> RoutingMetadata:
    """
    Use a small LLM agent to analyse query intent and select a model.

    Always returns a RoutingMetadata (never None) — this is the final layer.
    Falls back to the user-supplied model or DEFAULT_MODEL on any failure.
    """
    fallback_model = request.model or settings.default_model

    # Extract the user query
    user_texts = [
        msg.content for msg in request.messages
        if msg.role == "user" and msg.content
    ]
    if not user_texts:
        logger.debug("Layer3: no user messages — using fallback model=%s", fallback_model)
        return RoutingMetadata(
            decision_layer="layer3_agent",
            decision_reason="no_user_messages_fallback",
            model_selected=fallback_model,
            confidence=0.5,
        )

    query_text = user_texts[-1]

    # Call the router LLM
    decision = await _call_router_llm(query_text, request.user_tier)

    if decision is None:
        logger.info("Layer3: agent failed — falling back to model=%s", fallback_model)
        return RoutingMetadata(
            decision_layer="layer3_agent",
            decision_reason="agent_call_failed_fallback",
            model_selected=fallback_model,
            confidence=0.5,
        )

    # Validate model exists in catalogue
    selected_model = _validate_model_selection(decision)
    if selected_model is None:
        logger.warning(
            "Layer3: agent selected unknown model '%s' — falling back",
            decision.get("selected_model"),
        )
        return RoutingMetadata(
            decision_layer="layer3_agent",
            decision_reason=f"agent_selected_unknown_model::{decision.get('selected_model', '?')}_fallback",
            model_selected=fallback_model,
            confidence=0.5,
        )

    confidence = float(decision.get("confidence", 0.8))
    task_type = decision.get("task_type", "other")
    reasoning = decision.get("reasoning", "")

    logger.info(
        "Layer3 DECISION | model=%s task=%s confidence=%.2f reason=%s",
        selected_model, task_type, confidence, reasoning,
    )

    return RoutingMetadata(
        decision_layer="layer3_agent",
        decision_reason=f"agent_routed::task={task_type}::{reasoning[:100]}",
        model_selected=selected_model,
        confidence=min(confidence, 1.0),
    )
