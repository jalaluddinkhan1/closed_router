"""
app/core/classifier.py
──────────────────────
Tri-Modal Cascade Classifier (The "Brain")

  *** DEPRECATED — Superseded by the MoE Expert Router ***

  This module implemented a 3-layer cascade (Heuristic → Semantic → LLM) to
  select an ExecutionMode before the Tri-Modal Orchestrator was replaced by the
  System-Level Mixture of Experts (MoE) architecture.

  The MoE Router (app/experts/router.py) now handles ALL routing decisions:
    - Expert capability scoring replaces the heuristic/semantic/LLM cascade.
    - Parallel expert execution (app/experts/executor.py) replaces mode dispatch.
    - The cascade's Mode 1 patterns are embedded in MathExpert, DatetimeExpert,
      JsonFormatterExpert, and PIIRedactorExpert capability tags.

  DO NOT add new callers. This file is retained only for reference.

References:
  FrugalGPT — Cascade classification strategy.
  ReAct     — Reasoning + Acting (applied to Mode 3 routing decision).
"""

from __future__ import annotations

import json
import re

import litellm

from app.config import get_settings
from app.embedding_engine import embed_text
from app.logger import get_logger
from app.models import ChatRequest, ExecutionMode
from app.vector_store import search

logger = get_logger("router.classifier")
settings = get_settings()


_MATH_RE = re.compile(
    r"(?:what\s+is\s+|calculate\s+|compute\s+|solve\s+)"
    r"[\d\s\+\-\*\/\(\)\.%\^]+|"
    r"^\s*[\d\s\+\-\*\/\(\)\.%\^]+\s*[=?]?\s*$|"
    r"\b(multiply|divide|subtract|add|modulo|remainder|percentage of)\b",
    re.IGNORECASE | re.MULTILINE,
)
_PII_REDACT_RE = re.compile(
    r"\b(redact|mask|anonymize|anonymise|scrub|remove pii|hide ssn|censor)\b",
    re.IGNORECASE,
)
_FORMAT_RE = re.compile(
    r"\b(format|prettify|pretty.?print|indent|beautify)\b.{0,30}\b(json|csv|xml|yaml)\b|"
    r"\b(json|csv|xml|yaml)\b.{0,30}\b(format|prettify|indent)\b",
    re.IGNORECASE | re.DOTALL,
)
_DATETIME_RE = re.compile(
    r"\b(what.*(time|date|day)|current (time|date|datetime)|"
    r"today['s]?\s*(date|day)|what\s+day\s+is\s+(it|today))\b",
    re.IGNORECASE,
)

_AGENTIC_RE = re.compile(
    r"\b(search\s+for|look\s+up|browse|find\s+out|research|"
    r"latest|current\s+news|who\s+won|what\s+happened|"
    r"compare\s+\w+\s+and|first\s+.*\s+then|"
    r"plan\s+(a|the|my)|schedule|"
    r"write\s+.*\s+and\s+(save|send|email)|"
    r"create\s+.*\s+and\s+(publish|post|upload)|"
    r"analyze\s+multiple|multi.?step|step.by.step\s+plan|"
    r"real.?time|live\s+data)\b",
    re.IGNORECASE | re.DOTALL,
)


def _heuristic_classify(query: str) -> ExecutionMode | None:
    """
    Layer 1: Pure regex classification.
    Returns a mode or None to defer to the next layer.
    Mode 1 signals take priority over Mode 3 signals.
    """
    if _MATH_RE.search(query) or _PII_REDACT_RE.search(query) \
            or _FORMAT_RE.search(query) or _DATETIME_RE.search(query):
        logger.debug("Heuristic L1 → Mode 1 (Deterministic)")
        return ExecutionMode.DETERMINISTIC

    if _AGENTIC_RE.search(query):
        logger.debug("Heuristic L1 → Mode 3 (Agentic)")
        return ExecutionMode.AGENTIC

    return None  # defer to Layer 2


async def _semantic_classify(query: str) -> ExecutionMode | None:
    """
    Layer 2: Embed the query and compare against the Qdrant history of
    past successful routing decisions.

    If a very similar query was previously handled well by Mode 2, assume
    this query is also Mode 2 and skip the LLM classifier.
    """
    embedding = await embed_text(query)
    if embedding is None:
        return None  # embedding engine unavailable → defer

    result = await search(embedding, top_k=1)
    if result.found and result.best is not None:
        score = result.best.score
        if score >= settings.semantic_similarity_threshold:
            logger.debug(
                "Semantic L2 → Mode 2 (Probabilistic): score=%.4f matched='%s'",
                score,
                result.best.query_text[:60],
            )
            return ExecutionMode.PROBABILISTIC

    return None  # defer to Layer 3


_CLASSIFIER_SYSTEM = """\
You are a TASK CLASSIFIER for an intelligent compute orchestrator.
Analyze the user's query and classify it into exactly one execution mode.

MODES:
- mode1_deterministic : Math calculations, unit conversions, PII redaction,
  JSON/CSV formatting, current date/time. These can be handled by Python code
  alone — no language model needed.
- mode2_probabilistic : Summarization, creative writing, translation, simple
  factual Q&A, sentiment analysis, tone adjustment. One LLM call is sufficient.
- mode3_agentic       : Web research requiring live data, multi-step tasks
  ("do X then Y"), complex analysis across multiple sources, planning.
  Requires an agent with tools (search, code execution).

RULES:
- Free-tier users CANNOT use mode3_agentic (too expensive). Downgrade to
  mode2_probabilistic for free-tier if you would otherwise choose mode3.
- When in doubt between mode1 and mode2, choose mode2.
- When in doubt between mode2 and mode3, choose mode2.

Respond with ONLY valid JSON — no markdown, no explanation:
{"mode": "<mode1_deterministic|mode2_probabilistic|mode3_agentic>",
 "reasoning": "<one concise sentence>",
 "confidence": <float 0.0-1.0>}"""


async def _llm_classify(query: str, user_tier: str) -> ExecutionMode:
    """
    Layer 3: Use a small LLM to classify query intent.
    Always returns a mode (final fallback = Mode 2).
    """
    user_prompt = f"User tier: {user_tier}\nQuery: {query[:1500]}"

    try:
        response = await litellm.acompletion(
            model=settings.agent_router_model,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        data = json.loads(raw)
        mode_str   = data.get("mode", "mode2_probabilistic")
        reasoning  = data.get("reasoning", "")
        confidence = float(data.get("confidence", 0.8))

        # Enforce tier restrictions: free users cannot use agentic mode
        if mode_str == ExecutionMode.AGENTIC and user_tier == "free":
            logger.info("LLM Classifier: Mode 3 blocked for free-tier → Mode 2")
            mode_str = ExecutionMode.PROBABILISTIC

        try:
            mode = ExecutionMode(mode_str)
        except ValueError:
            mode = ExecutionMode.PROBABILISTIC

        logger.info(
            "LLM Classifier L3 → %s | confidence=%.2f | %s",
            mode.value, confidence, reasoning,
        )
        return mode

    except Exception as exc:
        logger.warning("LLM classifier failed (%s) — defaulting to Mode 2", exc)
        return ExecutionMode.PROBABILISTIC


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


async def classify(request: ChatRequest) -> tuple[ExecutionMode, str]:
    """
    Run the cascade classifier and return (ExecutionMode, classifier_layer_used).

    Cascade order (FrugalGPT pattern):
      Heuristic (free, <1ms) → Semantic (free, ~20ms) → LLM (~$0.00001, ~500ms)
    """
    user_msgs = [
        m.content for m in request.messages
        if m.role == "user" and m.content
    ]
    if not user_msgs:
        return ExecutionMode.PROBABILISTIC, "default_no_content"

    query = user_msgs[-1].strip()

    # Layer 1: Heuristic
    mode = _heuristic_classify(query)
    if mode is not None:
        return mode, "heuristic_layer1"

    # Layer 2: Semantic
    mode = await _semantic_classify(query)
    if mode is not None:
        return mode, "semantic_layer2"

    # Layer 3: LLM classifier
    mode = await _llm_classify(query, request.user_tier)
    return mode, "llm_classifier_layer3"
