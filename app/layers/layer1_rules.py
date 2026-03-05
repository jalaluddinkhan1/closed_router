"""
app/layers/layer1_rules.py
──────────────────────────
Layer 1: Deterministic Rule-Based Gate.

Evaluates an ordered pipeline of rules. The first rule to fire
short-circuits the funnel and returns a RoutingMetadata. If no rule
fires, returns None to defer to Layer 2 (Semantic).

Rules (in evaluation order):
  1. PII Detection       → reroute to private_model
  2. Blocked Model Check → reject the request
  3. User Tier Gate      → downgrade model for free-tier users
  4. Token Limit Check   → reject if prompt exceeds tier limit
"""

from __future__ import annotations

from app.config import get_settings
from app.logger import get_logger
from app.models import ChatRequest, RoutingMetadata
from app.pii_engine import scan_messages

logger = get_logger("router.layer1")
settings = get_settings()


async def _check_pii(request: ChatRequest) -> RoutingMetadata | None:
    """
    Rule 1: Scan user messages for PII via Presidio.
    If PII is detected, reroute to the private (on-prem) model.
    """
    user_texts = [
        msg.content for msg in request.messages
        if msg.role == "user" and msg.content
    ]
    if not user_texts:
        return None

    result = await scan_messages(user_texts)

    if result.detected:
        logger.warning(
            "PII detected: entities=%s — rerouting to private model '%s'",
            result.entity_types,
            settings.private_model,
        )
        return RoutingMetadata(
            decision_layer="layer1_rules",
            decision_reason=f"pii_detected::{','.join(result.entity_types)}",
            model_selected=settings.private_model,
            confidence=1.0,
            pii_detected=True,
            pii_entities=result.entity_types,
        )

    return None


async def _check_blocked_model(request: ChatRequest) -> RoutingMetadata | None:
    """
    Rule 2: Reject requests that explicitly target a blocked model.
    """
    if not request.model or not settings.blocked_models:
        return None

    if request.model.lower() in [m.lower() for m in settings.blocked_models]:
        logger.warning(
            "Blocked model requested: '%s' — rejecting",
            request.model,
        )
        return RoutingMetadata(
            decision_layer="layer1_rules",
            decision_reason=f"model_blocked::{request.model}",
            model_selected=settings.default_model,
            confidence=1.0,
        )

    return None


async def _check_tier_gate(request: ChatRequest) -> RoutingMetadata | None:
    """
    Rule 3: Free-tier users can only use models in the allowlist.
    If they request a premium model, silently downgrade to the default.
    """
    if request.user_tier != "free":
        return None

    if not request.model:
        return None  # No explicit model — Layer 3 will decide

    allowed = [m.lower() for m in settings.free_tier_models]
    if request.model.lower() not in allowed:
        logger.info(
            "Free-tier user '%s' requested premium model '%s' — "
            "downgrading to '%s'",
            request.user_id,
            request.model,
            settings.default_model,
        )
        return RoutingMetadata(
            decision_layer="layer1_rules",
            decision_reason=f"tier_downgrade::free_tier_cannot_use_{request.model}",
            model_selected=settings.default_model,
            confidence=1.0,
        )

    return None


def _estimate_tokens(text: str) -> int:
    """
    Quick token estimation: ~4 characters per token (English).
    This avoids pulling in tiktoken as a dependency just for a gate check.
    """
    return max(1, len(text) // 4)


async def _check_token_limit(request: ChatRequest) -> RoutingMetadata | None:
    """
    Rule 4: Reject or warn if the prompt exceeds the per-tier token limit.
    """
    tier_limits = {
        "free": settings.free_tier_token_limit,
        "pro": settings.pro_tier_token_limit,
        "enterprise": settings.enterprise_tier_token_limit,
    }
    limit = tier_limits.get(request.user_tier, settings.free_tier_token_limit)

    total_chars = sum(len(msg.content) for msg in request.messages if msg.content)
    estimated_tokens = max(1, total_chars // 4) if total_chars > 0 else 0

    if estimated_tokens > limit:
        logger.warning(
            "Token limit exceeded: user=%s tier=%s estimated=%d limit=%d",
            request.user_id,
            request.user_tier,
            estimated_tokens,
            limit,
        )
        return RoutingMetadata(
            decision_layer="layer1_rules",
            decision_reason=(
                f"token_limit_exceeded::"
                f"estimated_{estimated_tokens}_tokens_exceeds_{request.user_tier}_limit_{limit}"
            ),
            model_selected=settings.default_model,
            confidence=1.0,
        )

    return None


_RULES = [
    _check_pii,
    _check_blocked_model,
    _check_tier_gate,
    _check_token_limit,
]


async def run(request: ChatRequest) -> RoutingMetadata | None:
    """
    Execute the Layer 1 rule pipeline.

    Returns:
        RoutingMetadata if a rule fires.
        None to pass control to Layer 2.
    """
    for rule_fn in _RULES:
        result = await rule_fn(request)
        if result is not None:
            logger.info(
                "Layer1 FIRED | rule=%s reason=%s model=%s",
                rule_fn.__name__,
                result.decision_reason,
                result.model_selected,
            )
            return result

    logger.debug("Layer1: all rules passed — deferring to Layer2")
    return None
