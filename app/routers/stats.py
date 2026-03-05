"""
app/routers/stats.py
────────────────────
GET /v1/stats — Observability endpoint for dashboards and monitoring.

Returns aggregated statistics from the SQLite request log database.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app import database
from app.logger import get_logger

router = APIRouter()
logger = get_logger("router.stats")


@router.get(
    "/stats",
    summary="Router Statistics",
    description="Returns aggregated routing statistics from the request log database.",
    tags=["Observability"],
)
async def get_stats() -> dict[str, Any]:
    """Return a comprehensive stats snapshot including tri-modal metrics."""
    total_requests  = await database.get_total_requests()
    total_cost      = await database.get_total_cost()
    total_saved     = await database.get_total_cost_saved()
    per_model       = await database.get_requests_per_model()
    routing_dist    = await database.get_routing_distribution()
    mode_dist       = await database.get_mode_distribution()
    pii_stats       = await database.get_pii_stats()

    return {
        "total_requests":    total_requests,
        "total_cost_usd":    round(total_cost, 6),
        "total_saved_usd":   round(total_saved, 6),
        "requests_per_model":  per_model,
        "routing_distribution": routing_dist,
        "mode_distribution":   mode_dist,
        "pii_stats":           pii_stats,
    }


@router.get(
    "/stats/recent",
    summary="Recent Requests",
    description="Returns the most recent request logs.",
    tags=["Observability"],
)
async def get_recent(
    limit: int = Query(default=50, ge=1, le=500, description="Number of records"),
) -> list[dict[str, Any]]:
    """Return recent request logs."""
    return await database.get_recent_requests(limit=limit)


@router.get(
    "/stats/hourly",
    summary="Hourly Stats",
    description="Returns per-hour request counts and costs.",
    tags=["Observability"],
)
async def get_hourly(
    hours: int = Query(default=24, ge=1, le=168, description="Hours to look back"),
) -> list[dict[str, Any]]:
    """Return hourly aggregated stats."""
    return await database.get_hourly_stats(hours=hours)


@router.get(
    "/stats/modes",
    summary="Execution Mode Distribution",
    description="Returns per-mode request counts, savings, and latency.",
    tags=["Observability"],
)
async def get_mode_distribution() -> list[dict[str, Any]]:
    """Return stats grouped by execution mode (Mode 1/2/3)."""
    return await database.get_mode_distribution()
