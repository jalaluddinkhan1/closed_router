"""
app/embedding_engine.py
───────────────────────
Local embedding model via sentence-transformers.

Provides a lazy-initialised SentenceTransformer singleton and an async
wrapper that offloads encoding to a thread so the event loop stays free.

Graceful degradation: if sentence-transformers is not installed, all
encode calls return None and Layer 2 is effectively skipped.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from app.config import get_settings
from app.logger import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger("router.embedding")
settings = get_settings()

_model: SentenceTransformer | None = None
_model_available: bool | None = None  # tri-state


def _get_model() -> SentenceTransformer | None:
    """Return the SentenceTransformer, creating it on first call."""
    global _model, _model_available

    if _model_available is False:
        return None
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer as _ST

        logger.info("Loading embedding model '%s' …", settings.embedding_model)
        _model = _ST(settings.embedding_model)
        _model_available = True
        logger.info(
            "Embedding model loaded — dimension=%d",
            _model.get_sentence_embedding_dimension(),
        )
        return _model

    except Exception as exc:
        _model_available = False
        logger.warning(
            "sentence-transformers unavailable — Layer 2 disabled. Error: %s",
            exc,
        )
        return None


def get_embedding_dimension() -> int:
    """Return the dimension of the active model, or 384 as a safe default."""
    model = _get_model()
    if model is not None:
        return model.get_sentence_embedding_dimension()
    return 384  # default for all-MiniLM-L6-v2


async def embed_text(text: str) -> list[float] | None:
    """
    Encode a single string into a dense vector.

    Returns:
        A list of floats (the embedding), or None if the model is unavailable.
    """
    model = _get_model()
    if model is None:
        return None

    def _encode() -> list[float]:
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    return await asyncio.to_thread(_encode)


async def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """
    Batch-encode multiple strings.

    Returns:
        A list of embedding vectors, or None if the model is unavailable.
    """
    model = _get_model()
    if model is None:
        return None

    def _encode_batch() -> list[list[float]]:
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]

    return await asyncio.to_thread(_encode_batch)


def reset_engine() -> None:
    """Reset the singleton — useful for testing."""
    global _model, _model_available
    _model = None
    _model_available = None
