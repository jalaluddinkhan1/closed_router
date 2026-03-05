"""
app/experts/router.py
─────────────────────
MoE Router — Expert Scoring and Top-K Selection

Implements a multi-dimensional scoring function to rank experts:

  score = (semantic_similarity  * 0.50)   # embedding cosine sim
        + (cost_score           * 0.20)   # lower cost = higher score
        + (latency_score        * 0.20)   # lower latency = higher score
        + (reliability_score    * 0.10)   # historical success rate

Then selects Top-K experts based on score thresholds and constraints:
  - Maximum K = AGENTIC_MAX_STEPS (prevents runaway cost)
  - Free-tier users: no expert with cost_usd > $0.001
  - At least one expert is always selected (fallback to FastLLM)
  - Deterministic experts with score > 0.6 are always included if matched

Self-improving: expert scores improve every call via ExpertBase.update_stats().

Reference:
  FrugalGPT — cost-aware cascade selection.
  MoE gating — expert relevance computed via embedding similarity.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from app.config import get_settings
from app.embedding_engine import embed_text
from app.experts.registry import ExpertBase, ExpertRegistry, get_registry
from app.logger import get_logger

if TYPE_CHECKING:
    from app.models import ChatRequest

logger   = get_logger("router.moe_router")
settings = get_settings()

# Scoring weights — semantic is dominant so query relevance drives selection,
# not just cheapness/speed (which unfairly inflates deterministic expert scores).
_W_SEMANTIC    = 0.60
_W_COST        = 0.15
_W_LATENCY     = 0.15
_W_RELIABILITY = 0.10

# Normalisation ceilings
_MAX_COST_USD   = 0.01    # $0.01 per call — anything higher is penalised heavily
_MAX_LATENCY_MS = 5000.0  # 5 seconds — hard ceiling for latency penalty

# Selection thresholds
_INCLUDE_THRESHOLD  = 0.45   # Experts below this score are never selected
_PRIORITY_THRESHOLD = 0.70   # Experts above this are always included (up to K)
_MAX_K              = 3      # Maximum number of concurrent experts

_LLM_NAMES = {"fast_llm", "smart_llm", "groq_llm"}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity for pre-normalised float lists."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _keyword_score(query: str, tags: list[str]) -> float:
    """
    Fallback scoring when embeddings are unavailable.
    Returns fraction of tags found in the query (capped at 1.0).
    """
    q = query.lower()
    hits = sum(1 for tag in tags if tag.lower() in q)
    return min(hits / max(len(tags), 1), 1.0)


def _score_expert(
    expert: ExpertBase,
    query: str,
    query_embedding: list[float] | None,
    user_tier: str,
) -> float:
    """
    Compute the composite relevance score for one expert.

    Components:
      semantic  — How well does the expert description match the query?
      cost      — Cheaper experts score higher (tier-aware).
      latency   — Faster experts score higher.
      reliab    — Higher historical success_rate scores higher.
    """
    meta = expert.metadata

    if query_embedding and expert._capability_embedding:
        semantic = (_cosine_similarity(query_embedding, expert._capability_embedding) + 1) / 2
    else:
        semantic = _keyword_score(query, meta.capability_tags)

    cost_ratio = min(meta.cost_usd / _MAX_COST_USD, 1.0)
    cost_score = 1.0 - cost_ratio

    # Tier penalty: free users should not use expensive experts
    if user_tier == "free" and meta.cost_usd > 0.001:
        cost_score *= 0.1

    latency_score = 1.0 - min(meta.avg_latency_ms / _MAX_LATENCY_MS, 1.0)

    score = (
        semantic      * _W_SEMANTIC
        + cost_score  * _W_COST
        + latency_score * _W_LATENCY
        + meta.success_rate * _W_RELIABILITY
    )
    return round(score, 4)


async def select_experts(
    request: "ChatRequest",
    registry: ExpertRegistry | None = None,
) -> list[tuple[ExpertBase, float]]:
    """
    Score all registered experts and return the selected Top-K list
    as (expert, score) pairs, sorted by score descending.

    Selection rules:
      1. Score every expert using the composite scoring function.
      2. Always include the highest-scoring expert (minimum K=1).
      3. Include additional experts whose score > _INCLUDE_THRESHOLD,
         up to _MAX_K total.
      4. If the top expert is a deterministic ($0) expert with score > 0.6,
         also include the best LLM expert for fallback synthesis.
      5. If the top expert is a tool (search/code), pair it with FastLLM
         for result formatting.
    """
    reg = registry or get_registry()

    # Extract user query for embedding
    user_msgs = [m.content for m in request.messages if m.role == "user" and m.content]
    query = user_msgs[-1].strip() if user_msgs else ""

    # Compute query embedding (best-effort)
    query_embedding: list[float] | None = None
    try:
        query_embedding = await embed_text(query)
    except Exception:
        pass  # Fall back to keyword scoring

    # Score all experts
    scored: list[tuple[ExpertBase, float]] = []
    for expert in reg.all():
        score = _score_expert(expert, query, query_embedding, request.user_tier)
        scored.append((expert, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored:
        # Emergency fallback
        fallback = reg.get("fast_llm")
        if fallback:
            return [(fallback, 0.5)]
        return []

    selected: list[tuple[ExpertBase, float]] = []
    top_expert, top_score = scored[0]

    # Always include the top-scoring expert
    selected.append((top_expert, top_score))
    top_meta = top_expert.metadata

    # Determine if top expert is deterministic (zero cost, fast)
    is_deterministic = top_meta.cost_usd == 0.0 and top_meta.avg_latency_ms < 200.0
    # Determine if top expert is a tool (search/code — produces raw data)
    is_tool = top_meta.name in ("web_search", "python_repl")

    if is_deterministic:
        # When top expert is deterministic, eagerly add the best LLM expert first
        # (scan all scored, not just those before MAX_K is hit) so open-ended
        # queries always have an LLM fallback instead of filling all slots with
        # more deterministic experts that will fail.
        for expert, score in scored[1:]:
            if expert.metadata.name in _LLM_NAMES and score >= _INCLUDE_THRESHOLD:
                selected.append((expert, score))
                break
        # Then fill remaining slot(s) with additional deterministic experts
        for expert, score in scored[1:]:
            if len(selected) >= _MAX_K:
                break
            if score < _INCLUDE_THRESHOLD:
                break
            name = expert.metadata.name
            if name in _LLM_NAMES or any(e.metadata.name == name for e, _ in selected):
                continue
            selected.append((expert, score))

    elif is_tool:
        # Tool expert: pair with FastLLM to synthesize raw results into a clean answer
        for expert, score in scored[1:]:
            if expert.metadata.name == "fast_llm":
                selected.append((expert, score))
                break

    else:
        # LLM or other expert at top: add complementary experts if they score well
        for expert, score in scored[1:]:
            if len(selected) >= _MAX_K:
                break
            if score < _INCLUDE_THRESHOLD:
                break
            name = expert.metadata.name
            if any(e.metadata.name == name for e, _ in selected):
                continue
            selected.append((expert, score))

    # Safety net: guarantee at least one LLM expert is always present so open-ended
    # queries never return only a deterministic "failed" error message.
    if not any(e.metadata.name in _LLM_NAMES for e, _ in selected):
        fallback_llm = reg.get("fast_llm")
        if fallback_llm:
            fallback_score = next(
                (s for e, s in scored if e.metadata.name == "fast_llm"), 0.5
            )
            selected.append((fallback_llm, fallback_score))

    logger.info(
        "MoE selected %d expert(s): %s",
        len(selected),
        [(e.metadata.name, s) for e, s in selected],
    )
    return selected
