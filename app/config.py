"""
app/config.py
─────────────
Central configuration loaded from environment variables / .env file.
Uses pydantic-settings for type-safe, validated settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider Keys ────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")

    # ── Router Defaults ──────────────────────────────────────────────────────
    default_model: str = Field(default="gpt-4o-mini", alias="DEFAULT_MODEL")
    private_model: str = Field(default="ollama/llama3", alias="PRIVATE_MODEL")
    agent_router_model: str = Field(default="gpt-4o-mini", alias="AGENT_ROUTER_MODEL")

    # ── Layer 1: Rule-Based Gate ─────────────────────────────────────────────
    free_tier_models: list[str] = Field(
        default=["gpt-4o-mini"], alias="FREE_TIER_MODELS"
    )
    blocked_models: list[str] = Field(default=[], alias="BLOCKED_MODELS")
    free_tier_token_limit: int = Field(default=4096, alias="FREE_TIER_TOKEN_LIMIT")
    pro_tier_token_limit: int = Field(default=16384, alias="PRO_TIER_TOKEN_LIMIT")
    enterprise_tier_token_limit: int = Field(
        default=128000, alias="ENTERPRISE_TIER_TOKEN_LIMIT"
    )

    # ── PII Detection ────────────────────────────────────────────────────────
    pii_enabled: bool = Field(default=True, alias="PII_ENABLED")
    pii_score_threshold: float = Field(default=0.7, alias="PII_SCORE_THRESHOLD")

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="router_history", alias="QDRANT_COLLECTION")
    semantic_threshold: float = Field(default=0.85, alias="SEMANTIC_THRESHOLD")

    # ── Embedding Model ──────────────────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/router.db", alias="DATABASE_URL"
    )

    # ── Tri-Modal Orchestrator ────────────────────────────────────────────────
    # Semantic similarity threshold used by the tri-modal classifier Layer 2.
    # Queries with a score above this are routed to Mode 2 without calling the LLM.
    semantic_similarity_threshold: float = Field(
        default=0.82, alias="SEMANTIC_SIMILARITY_THRESHOLD"
    )

    # Mode 3 — Agentic Engine (ReAct agent with tools)
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    agentic_max_steps: int = Field(default=5, alias="AGENTIC_MAX_STEPS")
    agentic_timeout_seconds: float = Field(default=30.0, alias="AGENTIC_TIMEOUT_SECONDS")
    agentic_model: str = Field(default="gpt-4o-mini", alias="AGENTIC_MODEL")

    # Verifier — validates Mode 1 output with a small LLM (Speculative Decoding)
    verifier_enabled: bool = Field(default=True, alias="VERIFIER_ENABLED")
    verifier_model: str = Field(default="gpt-4o-mini", alias="VERIFIER_MODEL")

    # Estimated Mode 2 cost used to calculate Mode 1 savings (USD/1K tokens)
    mode2_reference_cost_per_1k: float = Field(
        default=0.00015, alias="MODE2_REFERENCE_COST_PER_1K"
    )

    @field_validator("semantic_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("semantic_threshold must be between 0 (exclusive) and 1 (inclusive)")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
