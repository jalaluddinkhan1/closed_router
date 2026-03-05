"""
app/pii_engine.py
─────────────────
PII detection via Microsoft Presidio.

Provides a lazy-initialised AnalyzerEngine singleton and a convenience
function to scan chat messages for personally identifiable information.

Graceful degradation: if Presidio or spaCy are not installed / the NLP
model is missing, PII scanning is silently disabled and all requests
pass through without blocking.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import get_settings
from app.logger import get_logger

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine

logger = get_logger("router.pii")
settings = get_settings()

DEFAULT_PII_ENTITIES: list[str] = [
    "CREDIT_CARD",
    "CRYPTO",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "MEDICAL_LICENSE",
    "NRP",
    "PERSON",
    "PHONE_NUMBER",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_PASSPORT",
    "US_SSN",
]


@dataclass
class PIIResult:
    """Holds the outcome of a PII scan."""

    detected: bool = False
    entity_types: list[str] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)


_engine: AnalyzerEngine | None = None
_engine_available: bool | None = None


def _get_analyzer() -> AnalyzerEngine | None:
    """
    Return the Presidio AnalyzerEngine, creating it on first call.
    Returns None if Presidio/spaCy are not available.
    """
    global _engine, _engine_available

    if _engine_available is False:
        return None
    if _engine is not None:
        return _engine

    try:
        from presidio_analyzer import AnalyzerEngine as _AE
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        try:
            nlp_engine = provider.create_engine()
        except Exception:
            logger.warning("en_core_web_lg not found, trying en_core_web_sm")
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            })
            nlp_engine = provider.create_engine()

        _engine = _AE(nlp_engine=nlp_engine, supported_languages=["en"])
        _engine_available = True
        logger.info("Presidio AnalyzerEngine initialised successfully")
        return _engine

    except Exception as exc:
        _engine_available = False
        logger.warning(
            "Presidio PII engine unavailable — PII scanning disabled. "
            "Install presidio-analyzer + a spaCy model to enable. Error: %s",
            exc,
        )
        return None


async def scan_messages(
    texts: list[str],
    *,
    entities: list[str] | None = None,
    score_threshold: float | None = None,
) -> PIIResult:
    """
    Scan a list of texts for PII.

    Args:
        texts:           The raw message strings (typically user messages).
        entities:        Override the default entity list.
        score_threshold: Minimum confidence to consider a match (0–1).

    Returns:
        PIIResult with detected=True if anything was found.
    """
    if not settings.pii_enabled:
        return PIIResult()

    analyzer = _get_analyzer()
    if analyzer is None:
        return PIIResult()

    threshold = score_threshold or settings.pii_score_threshold
    target_entities = entities or DEFAULT_PII_ENTITIES

    # Run blocking Presidio call in a thread so we don't stall the event loop
    def _scan() -> PIIResult:
        all_entities: list[str] = []
        all_details: list[dict] = []

        for text in texts:
            if not text or not text.strip():
                continue

            results = analyzer.analyze(
                text=text,
                entities=target_entities,
                language="en",
                score_threshold=threshold,
            )

            for r in results:
                all_entities.append(r.entity_type)
                all_details.append({
                    "entity_type": r.entity_type,
                    "start": r.start,
                    "end": r.end,
                    "score": round(r.score, 3),
                    "text_snippet": text[max(0, r.start - 5): r.end + 5],
                })

        # Deduplicate entity types
        unique_types = sorted(set(all_entities))
        return PIIResult(
            detected=len(unique_types) > 0,
            entity_types=unique_types,
            details=all_details,
        )

    return await asyncio.to_thread(_scan)


def reset_engine() -> None:
    """Reset the singleton — useful for testing."""
    global _engine, _engine_available
    _engine = None
    _engine_available = None
