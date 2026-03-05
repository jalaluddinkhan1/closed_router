"""
app/layers/layer2_semantic.py
─────────────────────────────
Layer 2: Semantic Router.

Embeds the incoming query with sentence-transformers, searches Qdrant
for similar past successful routing decisions, and returns the model
used in the best match if similarity exceeds the configured threshold.

If the embedding engine or Qdrant are unavailable, this layer silently
defers to Layer 3 (Agent).
"""

from __future__ import annotations

from app.config import get_settings
from app.embedding_engine import embed_text
from app.logger import get_logger
from app.models import ChatRequest, RoutingMetadata
from app.vector_store import search

logger = get_logger("router.layer2")
settings = get_settings()


def _extract_query(request: ChatRequest) -> str:
    """
    Build a single search string from the user messages.
    Uses the last user message for maximum semantic relevance.
    """
    user_msgs = [
        msg.content for msg in request.messages
        if msg.role == "user" and msg.content
    ]
    if not user_msgs:
        return ""
    # Use the last user message as the primary query
    return user_msgs[-1].strip()


async def run(request: ChatRequest) -> RoutingMetadata | None:
    """
    Look up semantically similar past requests in Qdrant.

    Flow:
      1. Extract the user query text.
      2. Embed it with sentence-transformers.
      3. Search Qdrant for neighbours above the similarity threshold.
      4. If a high-confidence match is found, return its model.

    Returns:
        RoutingMetadata if a confident match is found.
        None to pass control to Layer 3.
    """
    query = _extract_query(request)
    if not query:
        logger.debug("Layer2: no user messages to embed, deferring to Layer3")
        return None

    embedding = await embed_text(query)
    if embedding is None:
        logger.debug("Layer2: embedding engine unavailable, deferring to Layer3")
        return None

    result = await search(embedding, top_k=3)

    if not result.found or result.best is None:
        logger.debug("Layer2: no semantic match found, deferring to Layer3")
        return None

    best = result.best
    logger.info(
        "Layer2 HIT | score=%.4f model=%s reason=%s query_snippet='%s'",
        best.score,
        best.model_used,
        best.decision_reason,
        best.query_text[:60],
    )

    return RoutingMetadata(
        decision_layer="layer2_semantic",
        decision_reason=(
            f"semantic_cache_hit::score={best.score:.4f}::"
            f"matched_query={best.query_text[:80]}"
        ),
        model_selected=best.model_used,
        confidence=best.score,
    )
