"""
app/logger.py
─────────────
Structured logging for the LLM Router.

Dual output:
  1. Structured console logs (always active)
  2. SQLite persistence via app.database (when initialised)
"""

import logging
import sys
from typing import Any

from app.config import get_settings

settings = get_settings()


def _build_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(settings.log_level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(fmt)
    return handler


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured for the router."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_build_handler())
    logger.setLevel(settings.log_level)
    logger.propagate = False
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Request event logger — writes to console AND SQLite
# ─────────────────────────────────────────────────────────────────────────────

_req_logger = get_logger("router.request")


async def log_request_event(
    request_id: str,
    user_id: str,
    model_used: str,
    decision_layer: str,
    decision_reason: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    pii_detected: bool,
    user_tier: str = "free",
    pii_entities: list[str] | None = None,
    execution_mode: str = "mode2_probabilistic",
    cost_saved_usd: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Log a completed routing + inference event to:
      1. Console (structured text)
      2. SQLite (persistent, queryable)
    """
    # ── Console log ───────────────────────────────────────────────────────────
    _req_logger.info(
        "REQUEST_COMPLETE | id=%s user=%s model=%s mode=%s layer=%s "
        "latency=%.1fms tokens=%d/%d saved=$%.6f pii=%s",
        request_id,
        user_id,
        model_used,
        execution_mode,
        decision_layer,
        latency_ms,
        prompt_tokens,
        completion_tokens,
        cost_saved_usd,
        pii_detected,
    )
    if extra:
        _req_logger.debug("REQUEST_EXTRA | id=%s extra=%s", request_id, extra)

    # ── SQLite persistence ────────────────────────────────────────────────────
    try:
        from app.database import log_request

        await log_request(
            request_id=request_id,
            user_id=user_id,
            user_tier=user_tier,
            model_used=model_used,
            decision_layer=decision_layer,
            decision_reason=decision_reason,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            pii_detected=pii_detected,
            pii_entities=pii_entities,
            execution_mode=execution_mode,
            cost_saved_usd=cost_saved_usd,
        )
    except Exception as exc:
        _req_logger.warning("SQLite log failed (non-fatal): %s", exc)
