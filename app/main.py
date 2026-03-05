"""
app/main.py
───────────
FastAPI application entry point — Tri-Modal Adaptive Orchestrator (MoE Edition)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logger import get_logger
from app.models import HealthResponse
from app.routers import chat, stats
from app import vector_store, embedding_engine, database
from app.experts.registry import get_registry

settings = get_settings()
logger   = get_logger("router.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Startup:  embedding engine → Qdrant → SQLite → Expert Registry
    Shutdown: SQLite → Qdrant
    """
    logger.info("═══ Tri-Modal MoE Orchestrator starting ═══")
    logger.info("Default model : %s", settings.default_model)
    logger.info("Agentic model : %s", settings.agentic_model)
    logger.info("Log level     : %s", settings.log_level)

    # Embedding engine + Qdrant (needed by MoE router for semantic scoring)
    qdrant_ok = await vector_store.initialise()
    logger.info("Qdrant ready  : %s", qdrant_ok)

    # SQLite request log
    db_ok = await database.initialise()
    logger.info("SQLite ready  : %s", db_ok)

    # Expert Registry — pre-compute capability embeddings for all 9 experts
    registry = get_registry()
    await registry.initialize()
    logger.info("Expert registry: %d experts ready", len(registry.all()))

    yield

    # Graceful shutdown
    await database.shutdown()
    await vector_store.shutdown()
    logger.info("═══ Tri-Modal MoE Orchestrator shutting down ═══")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tri-Modal Adaptive Orchestrator",
        description=(
            "System-Level Mixture of Experts (MoE) orchestrator. "
            "Every request is scored against 9 specialised experts "
            "(Deterministic, Tool, LLM) and routed to the optimal Top-K "
            "combination for parallel execution."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router, prefix="/v1")
    app.include_router(stats.router, prefix="/v1")

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Meta"],
    )
    async def health() -> HealthResponse:
        from app.vector_store import _client_available as qdrant_ok
        from app.embedding_engine import _model_available as embed_ok

        registry = get_registry()
        expert_stats: list[dict[str, Any]] = registry.stats()

        components: dict[str, Any] = {
            "safety_gate":      "active",
            "embedding_engine": "active" if embed_ok else "unavailable",
            "qdrant":           "active" if qdrant_ok else "unavailable",
            "moe_router":       "active",
            "moe_executor":     "active",
            "expert_registry":  f"{len(expert_stats)} experts registered",
            "experts":          {e["name"]: f"sr={e['success_rate']:.2f} lat={e['avg_latency_ms']:.0f}ms" for e in expert_stats},
        }
        status = "ok" if embed_ok else "degraded"
        return HealthResponse(status=status, components=components)

    @app.get(
        "/v1/experts",
        summary="Expert Registry Stats",
        description="Live performance stats for all registered experts (self-improving metrics).",
        tags=["Observability"],
    )
    async def expert_stats() -> list[dict[str, Any]]:
        return get_registry().stats()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=True,
    )
