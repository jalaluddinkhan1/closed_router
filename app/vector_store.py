"""
app/vector_store.py
───────────────────
Qdrant vector database wrapper.

Manages the Qdrant collection used by Layer 2 (Semantic Router) to
store and retrieve past successful routing decisions.

Supports two modes:
  - In-memory (default, no QDRANT_URL set) — great for dev/testing
  - Remote     (QDRANT_URL points to a Qdrant server)

Each stored point contains:
  - vector : the query embedding
  - payload: { query_text, model_used, decision_reason, timestamp, user_tier }
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.embedding_engine import get_embedding_dimension
from app.logger import get_logger

logger = get_logger("router.vectorstore")
settings = get_settings()

_client: Any | None = None
_client_available: bool | None = None


@dataclass
class SemanticMatch:
    """A single search result from Qdrant."""

    query_text: str
    model_used: str
    decision_reason: str
    score: float
    user_tier: str = "free"
    timestamp: float = 0.0


@dataclass
class SemanticSearchResult:
    """Wraps zero or more matches."""

    found: bool = False
    matches: list[SemanticMatch] = field(default_factory=list)
    best: SemanticMatch | None = None


def _get_client() -> Any | None:
    """Return the Qdrant client, creating it on first call."""
    global _client, _client_available

    if _client_available is False:
        return None
    if _client is not None:
        return _client

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if settings.qdrant_url:
            logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
        else:
            logger.info("Using in-memory Qdrant (no QDRANT_URL set)")
            _client = QdrantClient(location=":memory:")

        # Ensure collection exists
        collections = [c.name for c in _client.get_collections().collections]
        if settings.qdrant_collection not in collections:
            dim = get_embedding_dimension()
            _client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d, cosine)",
                settings.qdrant_collection,
                dim,
            )
        else:
            logger.info("Qdrant collection '%s' already exists", settings.qdrant_collection)

        _client_available = True
        return _client

    except Exception as exc:
        _client_available = False
        logger.warning(
            "Qdrant unavailable — Layer 2 disabled. Error: %s", exc,
        )
        return None


async def initialise() -> bool:
    """
    Eagerly initialise the Qdrant client at startup.
    Returns True if ready, False otherwise.
    """
    import asyncio
    client = await asyncio.to_thread(_get_client)
    return client is not None


async def shutdown() -> None:
    """Close the Qdrant client connection on app shutdown."""
    global _client, _client_available
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None
    _client_available = None
    logger.info("Qdrant client closed")


async def search(
    embedding: list[float],
    *,
    top_k: int = 3,
    score_threshold: float | None = None,
) -> SemanticSearchResult:
    """
    Search Qdrant for the nearest neighbours of the given embedding.

    Args:
        embedding:       The query vector.
        top_k:           Number of results to return.
        score_threshold: Override the default SEMANTIC_THRESHOLD.

    Returns:
        SemanticSearchResult with matches sorted by descending score.
    """
    import asyncio

    client = _get_client()
    if client is None:
        return SemanticSearchResult()

    threshold = score_threshold or settings.semantic_threshold

    def _search() -> SemanticSearchResult:
        try:
            hits = client.search(
                collection_name=settings.qdrant_collection,
                query_vector=embedding,
                limit=top_k,
                score_threshold=threshold,
            )
        except Exception as exc:
            logger.error("Qdrant search failed: %s", exc)
            return SemanticSearchResult()

        if not hits:
            return SemanticSearchResult()

        matches = []
        for hit in hits:
            payload = hit.payload or {}
            matches.append(
                SemanticMatch(
                    query_text=payload.get("query_text", ""),
                    model_used=payload.get("model_used", ""),
                    decision_reason=payload.get("decision_reason", ""),
                    score=hit.score,
                    user_tier=payload.get("user_tier", "free"),
                    timestamp=payload.get("timestamp", 0.0),
                )
            )

        return SemanticSearchResult(
            found=True,
            matches=matches,
            best=matches[0] if matches else None,
        )

    return await asyncio.to_thread(_search)


async def store_routing_decision(
    embedding: list[float],
    query_text: str,
    model_used: str,
    decision_reason: str,
    user_tier: str = "free",
) -> bool:
    """
    Store a successful routing decision in Qdrant so Layer 2 can
    recommend the same model for semantically similar future queries.

    Returns True on success, False on failure.
    """
    import asyncio
    from qdrant_client.models import PointStruct

    client = _get_client()
    if client is None:
        return False

    point_id = str(uuid.uuid4())

    point = PointStruct(
        id=point_id,
        vector=embedding,
        payload={
            "query_text": query_text[:500],  # Truncate for storage efficiency
            "model_used": model_used,
            "decision_reason": decision_reason,
            "user_tier": user_tier,
            "timestamp": time.time(),
        },
    )

    def _upsert() -> bool:
        try:
            client.upsert(
                collection_name=settings.qdrant_collection,
                points=[point],
            )
            logger.debug(
                "Stored routing decision: model=%s reason=%s",
                model_used,
                decision_reason,
            )
            return True
        except Exception as exc:
            logger.error("Qdrant upsert failed: %s", exc)
            return False

    return await asyncio.to_thread(_upsert)


def reset_store() -> None:
    """Reset the singleton — useful for testing."""
    global _client, _client_available
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None
    _client_available = None
