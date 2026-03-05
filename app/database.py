"""
app/database.py
───────────────
Async SQLite persistence for request logging and observability.

Uses aiosqlite directly (no ORM overhead) for maximum simplicity
and performance. Creates the database and tables on first init.

Schema:
  - request_logs: every routed request with model, layer, latency, cost, PII
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import aiosqlite

from app.config import get_settings
from app.logger import get_logger

logger = get_logger("router.database")
settings = get_settings()

_db: aiosqlite.Connection | None = None

MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o-mini":                       {"input": 0.00015, "output": 0.0006},
    "gpt-4o":                            {"input": 0.0025,  "output": 0.01},
    "anthropic/claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "anthropic/claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "groq/llama-3.1-8b-instant":         {"input": 0.00005, "output": 0.00008},
    "groq/llama-3.1-70b-versatile":      {"input": 0.00059, "output": 0.00079},
    "ollama/llama3":                     {"input": 0.0,     "output": 0.0},
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the cost of a request in USD."""
    costs = MODEL_COSTS.get(model, {"input": 0.001, "output": 0.002})
    return (
        (prompt_tokens / 1000) * costs["input"]
        + (completion_tokens / 1000) * costs["output"]
    )


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS request_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    user_id         TEXT    NOT NULL,
    user_tier       TEXT    NOT NULL DEFAULT 'free',
    model_used      TEXT    NOT NULL,
    decision_layer  TEXT    NOT NULL,
    decision_reason TEXT    NOT NULL,
    latency_ms      REAL    NOT NULL DEFAULT 0.0,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    estimated_cost  REAL    NOT NULL DEFAULT 0.0,
    pii_detected    INTEGER NOT NULL DEFAULT 0,
    pii_entities    TEXT    DEFAULT '',
    execution_mode  TEXT    NOT NULL DEFAULT 'mode2_probabilistic',
    cost_saved_usd  REAL    NOT NULL DEFAULT 0.0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON request_logs(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_logs_model ON request_logs(model_used);",
    "CREATE INDEX IF NOT EXISTS idx_logs_layer ON request_logs(decision_layer);",
    "CREATE INDEX IF NOT EXISTS idx_logs_user ON request_logs(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_logs_mode ON request_logs(execution_mode);",
]

# Migration: add new columns to existing databases that pre-date the upgrade
_MIGRATION_SQL = [
    "ALTER TABLE request_logs ADD COLUMN execution_mode TEXT NOT NULL DEFAULT 'mode2_probabilistic';",
    "ALTER TABLE request_logs ADD COLUMN cost_saved_usd REAL NOT NULL DEFAULT 0.0;",
]


async def initialise() -> bool:
    """Create the database file, tables, and indexes."""
    global _db

    try:
        # Extract the SQLite file path from the DATABASE_URL
        db_path = settings.database_url
        if ":///" in db_path:
            db_path = db_path.split(":///")[-1]

        # Ensure the parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        _db = await aiosqlite.connect(db_path)
        _db.row_factory = aiosqlite.Row

        await _db.execute(CREATE_TABLE_SQL)
        for idx_sql in CREATE_INDEX_SQL:
            await _db.execute(idx_sql)

        # Run migrations for pre-existing databases (ignore errors if column exists)
        for migration in _MIGRATION_SQL:
            try:
                await _db.execute(migration)
            except Exception:
                pass  # Column already exists — safe to ignore

        await _db.commit()

        logger.info("SQLite database initialised at '%s'", db_path)
        return True

    except Exception as exc:
        logger.error("Failed to initialise SQLite: %s", exc)
        _db = None
        return False


async def shutdown() -> None:
    """Close the database connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
        logger.info("SQLite connection closed")


INSERT_SQL = """
INSERT INTO request_logs (
    request_id, timestamp, user_id, user_tier, model_used,
    decision_layer, decision_reason, latency_ms,
    prompt_tokens, completion_tokens, total_tokens,
    estimated_cost, pii_detected, pii_entities,
    execution_mode, cost_saved_usd
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


async def log_request(
    request_id: str,
    user_id: str,
    user_tier: str,
    model_used: str,
    decision_layer: str,
    decision_reason: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    pii_detected: bool,
    pii_entities: list[str] | None = None,
    execution_mode: str = "mode2_probabilistic",
    cost_saved_usd: float = 0.0,
) -> bool:
    """
    Persist a request log row to SQLite.
    Returns True on success, False on failure (never raises).
    """
    if _db is None:
        return False

    total_tokens = prompt_tokens + completion_tokens
    cost = _estimate_cost(model_used, prompt_tokens, completion_tokens)
    entities_str = ",".join(pii_entities) if pii_entities else ""

    try:
        await _db.execute(
            INSERT_SQL,
            (
                request_id,
                time.time(),
                user_id,
                user_tier,
                model_used,
                decision_layer,
                decision_reason,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost,
                int(pii_detected),
                entities_str,
                execution_mode,
                cost_saved_usd,
            ),
        )
        await _db.commit()
        return True
    except Exception as exc:
        logger.error("Failed to log request: %s", exc)
        return False


async def get_total_requests() -> int:
    """Return total number of logged requests."""
    if _db is None:
        return 0
    cursor = await _db.execute("SELECT COUNT(*) FROM request_logs")
    row = await cursor.fetchone()
    return row[0] if row else 0


async def get_total_cost() -> float:
    """Return total estimated cost in USD."""
    if _db is None:
        return 0.0
    cursor = await _db.execute("SELECT COALESCE(SUM(estimated_cost), 0) FROM request_logs")
    row = await cursor.fetchone()
    return row[0] if row else 0.0


async def get_requests_per_model() -> list[dict[str, Any]]:
    """Return request count grouped by model."""
    if _db is None:
        return []
    cursor = await _db.execute(
        "SELECT model_used, COUNT(*) as count, "
        "COALESCE(SUM(estimated_cost), 0) as total_cost, "
        "COALESCE(AVG(latency_ms), 0) as avg_latency "
        "FROM request_logs GROUP BY model_used ORDER BY count DESC"
    )
    rows = await cursor.fetchall()
    return [
        {
            "model": r[0],
            "count": r[1],
            "total_cost": round(r[2], 6),
            "avg_latency_ms": round(r[3], 1),
        }
        for r in rows
    ]


async def get_routing_distribution() -> list[dict[str, Any]]:
    """Return request count grouped by decision layer."""
    if _db is None:
        return []
    cursor = await _db.execute(
        "SELECT decision_layer, COUNT(*) as count "
        "FROM request_logs GROUP BY decision_layer ORDER BY count DESC"
    )
    rows = await cursor.fetchall()
    return [{"layer": r[0], "count": r[1]} for r in rows]


async def get_pii_stats() -> dict[str, Any]:
    """Return PII detection statistics."""
    if _db is None:
        return {"total_blocks": 0, "entity_breakdown": []}
    cursor = await _db.execute(
        "SELECT COUNT(*) FROM request_logs WHERE pii_detected = 1"
    )
    row = await cursor.fetchone()
    total_blocks = row[0] if row else 0

    cursor = await _db.execute(
        "SELECT pii_entities, COUNT(*) as count "
        "FROM request_logs WHERE pii_detected = 1 AND pii_entities != '' "
        "GROUP BY pii_entities ORDER BY count DESC LIMIT 20"
    )
    rows = await cursor.fetchall()
    entity_breakdown = [{"entities": r[0], "count": r[1]} for r in rows]

    return {"total_blocks": total_blocks, "entity_breakdown": entity_breakdown}


async def get_recent_requests(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent request logs."""
    if _db is None:
        return []
    cursor = await _db.execute(
        "SELECT request_id, timestamp, user_id, user_tier, model_used, "
        "decision_layer, decision_reason, latency_ms, prompt_tokens, "
        "completion_tokens, estimated_cost, pii_detected "
        "FROM request_logs ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    )
    rows = await cursor.fetchall()
    return [
        {
            "request_id": r[0],
            "timestamp": r[1],
            "user_id": r[2],
            "user_tier": r[3],
            "model_used": r[4],
            "decision_layer": r[5],
            "decision_reason": r[6],
            "latency_ms": round(r[7], 1),
            "prompt_tokens": r[8],
            "completion_tokens": r[9],
            "estimated_cost": round(r[10], 6),
            "pii_detected": bool(r[11]),
        }
        for r in rows
    ]


async def get_total_cost_saved() -> float:
    """Return total estimated cost saved by Mode 1 (Deterministic) in USD."""
    if _db is None:
        return 0.0
    cursor = await _db.execute(
        "SELECT COALESCE(SUM(cost_saved_usd), 0) FROM request_logs"
    )
    row = await cursor.fetchone()
    return row[0] if row else 0.0


async def get_mode_distribution() -> list[dict[str, Any]]:
    """Return request count grouped by execution mode."""
    if _db is None:
        return []
    cursor = await _db.execute(
        "SELECT execution_mode, COUNT(*) as count, "
        "COALESCE(SUM(cost_saved_usd), 0) as total_saved, "
        "COALESCE(AVG(latency_ms), 0) as avg_latency "
        "FROM request_logs GROUP BY execution_mode ORDER BY count DESC"
    )
    rows = await cursor.fetchall()
    return [
        {
            "mode": r[0],
            "count": r[1],
            "total_saved_usd": round(r[2], 6),
            "avg_latency_ms": round(r[3], 1),
        }
        for r in rows
    ]


async def get_hourly_stats(hours: int = 24) -> list[dict[str, Any]]:
    """Return request count and cost per hour for the last N hours."""
    if _db is None:
        return []
    since = time.time() - (hours * 3600)
    cursor = await _db.execute(
        "SELECT CAST(timestamp / 3600 AS INTEGER) * 3600 as hour_bucket, "
        "COUNT(*) as count, "
        "COALESCE(SUM(estimated_cost), 0) as cost, "
        "COALESCE(AVG(latency_ms), 0) as avg_latency "
        "FROM request_logs WHERE timestamp > ? "
        "GROUP BY hour_bucket ORDER BY hour_bucket",
        (since,),
    )
    rows = await cursor.fetchall()
    return [
        {
            "hour": r[0],
            "count": r[1],
            "cost": round(r[2], 6),
            "avg_latency_ms": round(r[3], 1),
        }
        for r in rows
    ]
