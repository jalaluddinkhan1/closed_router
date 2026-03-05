"""
app/experts/registry.py
───────────────────────
System-Level Mixture of Experts (MoE) — Expert Registry

Every capability is a named Expert with:
  - Semantic description    (embedded at startup for similarity scoring)
  - Capability tags         (fallback keyword matching when embedding unavailable)
  - Cost / latency profile  (for cost-aware routing)
  - Success rate            (self-improving via exponential moving average)

Expert implementations wrap existing mode engines:
  Deterministic: math, datetime, JSON, PII redaction
  Tool:          web_search, python_repl
  LLM:           fast (gpt-4o-mini), smart (gpt-4o), groq (llama-3.1-70b)

Reference: "Mixture of Agents (MoA)" — multiple specialised agents collaborating.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import litellm

from app.config import get_settings
from app.logger import get_logger

if TYPE_CHECKING:
    from app.models import ChatRequest

logger   = get_logger("router.experts")
settings = get_settings()

_EMA_ALPHA = 0.1


@dataclass
class ExpertMetadata:
    name: str
    description: str           # Used to compute capability embedding
    capability_tags: list[str]
    cost_usd: float            # Estimated per-call cost
    avg_latency_ms: float      # Running average latency (EMA)
    success_rate: float        # Historical success rate [0,1] (EMA)
    parallelizable: bool = True  # Can this expert run alongside others?


@dataclass
class ExpertResult:
    expert_name: str
    output: str | None
    success: bool
    latency_ms: float
    cost_usd: float = 0.0
    error: str | None = None


class ExpertBase(ABC):
    """Base class for every expert in the registry."""

    metadata: ExpertMetadata
    _capability_embedding: list[float] | None = None

    async def initialize(self) -> None:
        """Pre-compute the capability embedding (called once at startup)."""
        try:
            from app.embedding_engine import embed_text
            emb = await embed_text(self.metadata.description)
            self._capability_embedding = emb
            logger.debug("Expert '%s' embedding ready", self.metadata.name)
        except Exception as exc:
            logger.warning("Expert '%s' embedding failed: %s", self.metadata.name, exc)

    def update_stats(self, success: bool, latency_ms: float) -> None:
        """
        Self-improvement: update success_rate and avg_latency via EMA.
        This is the 'Reinforcement Optimization' component of the MoE system.
        """
        self.metadata.success_rate = (
            _EMA_ALPHA * int(success)
            + (1 - _EMA_ALPHA) * self.metadata.success_rate
        )
        self.metadata.avg_latency_ms = (
            _EMA_ALPHA * latency_ms
            + (1 - _EMA_ALPHA) * self.metadata.avg_latency_ms
        )

    @abstractmethod
    async def execute(self, query: str, request: "ChatRequest") -> ExpertResult:
        """Run this expert and return a result (never raises)."""


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Experts  (Mode 1 — $0.00, <50ms, 100% accurate)
# ─────────────────────────────────────────────────────────────────────────────


class MathExpert(ExpertBase):
    metadata = ExpertMetadata(
        name="math_evaluator",
        description=(
            "Evaluates mathematical expressions safely using Python AST — "
            "no LLM needed. Handles arithmetic, percentages, CAGR calculations, "
            "compound interest, and any numeric expression."
        ),
        capability_tags=["math", "calculate", "arithmetic", "compute", "percentage",
                         "cagr", "multiply", "divide", "add", "subtract", "result", "="],
        cost_usd=0.0,
        avg_latency_ms=5.0,
        success_rate=0.95,
    )

    async def execute(self, query: str, request: "ChatRequest") -> ExpertResult:
        from app.modes.deterministic import handle_math
        t = time.perf_counter()
        result = handle_math(query)
        latency = (time.perf_counter() - t) * 1000
        success = bool(result and result.success)
        self.update_stats(success, latency)
        return ExpertResult(
            expert_name=self.metadata.name,
            output=result.output if success else None,
            success=success,
            latency_ms=latency,
            error=result.error if result else "No math expression found",
        )


class DatetimeExpert(ExpertBase):
    metadata = ExpertMetadata(
        name="datetime_handler",
        description=(
            "Returns the current date, time, or datetime instantly. "
            "Handles queries about 'what time is it', 'today's date', "
            "'what day is it', 'current datetime'."
        ),
        capability_tags=["date", "time", "today", "current", "datetime", "day", "now", "when"],
        cost_usd=0.0,
        avg_latency_ms=1.0,
        success_rate=0.99,
    )

    async def execute(self, query: str, request: "ChatRequest") -> ExpertResult:
        from app.modes.deterministic import handle_datetime
        t = time.perf_counter()
        result = handle_datetime(query)
        latency = (time.perf_counter() - t) * 1000
        success = bool(result and result.success)
        self.update_stats(success, latency)
        return ExpertResult(
            expert_name=self.metadata.name,
            output=result.output if success else None,
            success=success,
            latency_ms=latency,
        )


class JsonFormatterExpert(ExpertBase):
    metadata = ExpertMetadata(
        name="json_formatter",
        description=(
            "Formats, prettifies, and indents JSON data. "
            "Use when the user wants to format JSON, prettify JSON, "
            "indent JSON, or parse and display JSON structure."
        ),
        capability_tags=["json", "format", "prettify", "indent", "beautify",
                         "parse", "structure", "yaml", "xml", "data"],
        cost_usd=0.0,
        avg_latency_ms=5.0,
        success_rate=0.98,
    )

    async def execute(self, query: str, request: "ChatRequest") -> ExpertResult:
        from app.modes.deterministic import handle_json_format
        t = time.perf_counter()
        result = handle_json_format(query)
        latency = (time.perf_counter() - t) * 1000
        success = bool(result and result.success)
        self.update_stats(success, latency)
        return ExpertResult(
            expert_name=self.metadata.name,
            output=result.output if success else None,
            success=success,
            latency_ms=latency,
        )


class PIIRedactorExpert(ExpertBase):
    metadata = ExpertMetadata(
        name="pii_redactor",
        description=(
            "Detects and redacts Personally Identifiable Information (PII) "
            "from text using Microsoft Presidio. Handles SSN, email addresses, "
            "phone numbers, credit card numbers, and other sensitive data."
        ),
        capability_tags=["pii", "redact", "anonymize", "privacy", "ssn",
                         "email", "phone", "credit card", "sensitive", "mask", "scrub"],
        cost_usd=0.0,
        avg_latency_ms=60.0,
        success_rate=0.90,
    )

    async def execute(self, query: str, request: "ChatRequest") -> ExpertResult:
        from app.modes.deterministic import handle_pii_redaction
        t = time.perf_counter()
        result = await handle_pii_redaction(query)
        latency = (time.perf_counter() - t) * 1000
        success = bool(result and result.success)
        self.update_stats(success, latency)
        return ExpertResult(
            expert_name=self.metadata.name,
            output=result.output if success else None,
            success=success,
            latency_ms=latency,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool Experts  (Mode 3 sub-components)
# ─────────────────────────────────────────────────────────────────────────────


class WebSearchExpert(ExpertBase):
    metadata = ExpertMetadata(
        name="web_search",
        description=(
            "Searches the web for real-time information, current events, news, "
            "stock prices, sports results, recent research, and any factual "
            "information that may have changed after a training data cutoff."
        ),
        capability_tags=["search", "web", "internet", "news", "current", "latest",
                         "real-time", "today", "recent", "live", "stock", "price",
                         "who won", "what happened", "find", "look up"],
        cost_usd=0.001,
        avg_latency_ms=2500.0,
        success_rate=0.85,
    )

    async def execute(self, query: str, request: "ChatRequest") -> ExpertResult:
        from app.modes.agentic import _web_search
        t = time.perf_counter()
        try:
            output = await _web_search(query)
            latency = (time.perf_counter() - t) * 1000
            success = not output.startswith("[Search unavailable") and not output.startswith("[Search error")
            self.update_stats(success, latency)
            return ExpertResult(
                expert_name=self.metadata.name,
                output=output,
                success=success,
                latency_ms=latency,
                cost_usd=self.metadata.cost_usd,
            )
        except Exception as exc:
            latency = (time.perf_counter() - t) * 1000
            self.update_stats(False, latency)
            return ExpertResult(
                expert_name=self.metadata.name,
                output=None,
                success=False,
                latency_ms=latency,
                error=str(exc),
            )


class PythonReplExpert(ExpertBase):
    metadata = ExpertMetadata(
        name="python_repl",
        description=(
            "Executes Python code in a sandboxed environment for complex "
            "calculations, data analysis, statistical computations, CAGR, "
            "compound interest, data transformations, and custom algorithms."
        ),
        capability_tags=["python", "code", "execute", "compute", "calculate",
                         "data", "algorithm", "script", "programming", "cagr",
                         "compound", "statistical", "analysis"],
        cost_usd=0.0,
        avg_latency_ms=1200.0,
        success_rate=0.82,
        parallelizable=True,
    )

    async def execute(self, query: str, request: "ChatRequest") -> ExpertResult:
        from app.modes.agentic import _python_repl
        # Ask a small LLM to write the code, then execute it
        t = time.perf_counter()
        try:
            code_prompt = (
                f"Write a short Python script (using only stdlib) that answers:\n{query}\n"
                "Print the final answer. No imports except math, datetime, json."
            )
            code_resp = await litellm.acompletion(
                model=settings.agent_router_model,
                messages=[
                    {"role": "system", "content": "Write clean, executable Python. Print the answer."},
                    {"role": "user", "content": code_prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            raw_code = (code_resp.choices[0].message.content or "").strip()
            # Strip markdown fences
            if "```python" in raw_code:
                raw_code = raw_code.split("```python")[-1].split("```")[0].strip()
            elif "```" in raw_code:
                raw_code = raw_code.split("```")[1].split("```")[0].strip()

            output = await _python_repl(raw_code)
            latency = (time.perf_counter() - t) * 1000
            success = not output.startswith("[") and bool(output.strip())
            self.update_stats(success, latency)
            return ExpertResult(
                expert_name=self.metadata.name,
                output=output.strip() if success else None,
                success=success,
                latency_ms=latency,
            )
        except Exception as exc:
            latency = (time.perf_counter() - t) * 1000
            self.update_stats(False, latency)
            return ExpertResult(
                expert_name=self.metadata.name,
                output=None,
                success=False,
                latency_ms=latency,
                error=str(exc),
            )


# ─────────────────────────────────────────────────────────────────────────────
# LLM Experts  (Mode 2 — probabilistic, single-call)
# ─────────────────────────────────────────────────────────────────────────────


class _LLMExpertBase(ExpertBase):
    """Shared logic for all LLM-backed experts."""
    MODEL: str = "gpt-4o-mini"

    async def execute(self, query: str, request: "ChatRequest") -> ExpertResult:
        from app.proxy import call_llm, ProxyError
        t = time.perf_counter()
        try:
            raw = await call_llm(request, self.MODEL)
            latency = raw.pop("_latency_ms", (time.perf_counter() - t) * 1000)
            content = (
                raw.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ) or ""
            usage = raw.get("usage") or {}
            cost = self._estimate_cost(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
            )
            self.update_stats(True, latency)
            return ExpertResult(
                expert_name=self.metadata.name,
                output=content,
                success=True,
                latency_ms=latency,
                cost_usd=cost,
            )
        except ProxyError as exc:
            latency = (time.perf_counter() - t) * 1000
            self.update_stats(False, latency)
            return ExpertResult(
                expert_name=self.metadata.name,
                output=None,
                success=False,
                latency_ms=latency,
                error=str(exc),
            )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # These are approximate — litellm has exact cost maps in production
        costs = {
            "gpt-4o-mini":                       (0.00015, 0.0006),
            "gpt-4o":                            (0.0025,  0.01),
            "groq/llama-3.1-70b-versatile":      (0.00059, 0.00079),
            "anthropic/claude-3-5-sonnet-20241022": (0.003, 0.015),
        }
        inp, out = costs.get(self.MODEL, (0.001, 0.002))
        return (prompt_tokens / 1000) * inp + (completion_tokens / 1000) * out


class FastLLMExpert(_LLMExpertBase):
    MODEL = "gpt-4o-mini"
    metadata = ExpertMetadata(
        name="fast_llm",
        description=(
            "Fast and cost-effective LLM for simple question answering, "
            "summarization, translation, classification, sentiment analysis, "
            "tone adjustment, and short creative writing."
        ),
        capability_tags=["summarize", "explain", "translate", "classify", "qa",
                         "answer", "simple", "what is", "how to", "describe",
                         "sentiment", "tone", "short", "quick", "cheap"],
        cost_usd=0.00015,
        avg_latency_ms=1500.0,
        success_rate=0.93,
    )


class SmartLLMExpert(_LLMExpertBase):
    MODEL = "gpt-4o"
    metadata = ExpertMetadata(
        name="smart_llm",
        description=(
            "Powerful LLM for complex multi-step reasoning, advanced coding, "
            "architecture design, debugging difficult problems, mathematical "
            "proofs, and tasks requiring deep analytical thinking."
        ),
        capability_tags=["complex", "reasoning", "coding", "debug", "architecture",
                         "design", "proof", "difficult", "hard", "advanced",
                         "multi-step", "analysis", "deep", "solve"],
        cost_usd=0.005,
        avg_latency_ms=3000.0,
        success_rate=0.95,
    )


class GroqExpert(_LLMExpertBase):
    MODEL = "groq/llama-3.1-70b-versatile"
    metadata = ExpertMetadata(
        name="groq_llm",
        description=(
            "Ultra-fast open-source LLM via Groq for reasoning, coding, "
            "and general tasks where speed matters and cost should be low. "
            "Excellent for structured output and straightforward reasoning."
        ),
        capability_tags=["fast", "open-source", "reasoning", "code", "cheap",
                         "structured", "format", "list", "outline", "draft"],
        cost_usd=0.0006,
        avg_latency_ms=800.0,
        success_rate=0.88,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Expert Registry — singleton that manages all experts
# ─────────────────────────────────────────────────────────────────────────────


class ExpertRegistry:
    """
    Singleton registry of all available experts.

    Responsibilities:
      - Hold all expert instances
      - Initialize capability embeddings at startup
      - Expose experts for scoring and selection
      - Persist self-improvement stats across requests
    """

    def __init__(self) -> None:
        self._experts: list[ExpertBase] = [
            # Deterministic (cheapest, fastest — always try first)
            MathExpert(),
            DatetimeExpert(),
            JsonFormatterExpert(),
            PIIRedactorExpert(),
            # Tool (real-world access)
            WebSearchExpert(),
            PythonReplExpert(),
            # LLM (probabilistic — last resort for open-ended tasks)
            FastLLMExpert(),
            SmartLLMExpert(),
            GroqExpert(),
        ]
        self._initialized = False

    async def initialize(self) -> None:
        """Pre-compute capability embeddings for all experts (called once at startup)."""
        logger.info("Initializing %d experts...", len(self._experts))
        await asyncio.gather(
            *[e.initialize() for e in self._experts],
            return_exceptions=True,
        )
        self._initialized = True
        logger.info("Expert registry ready")

    def all(self) -> list[ExpertBase]:
        return self._experts

    def get(self, name: str) -> ExpertBase | None:
        return next((e for e in self._experts if e.metadata.name == name), None)

    def stats(self) -> list[dict]:
        """Return current expert stats for observability."""
        return [
            {
                "name": e.metadata.name,
                "cost_usd": e.metadata.cost_usd,
                "avg_latency_ms": round(e.metadata.avg_latency_ms, 1),
                "success_rate": round(e.metadata.success_rate, 3),
            }
            for e in self._experts
        ]


# Module-level singleton
_registry: ExpertRegistry | None = None


def get_registry() -> ExpertRegistry:
    """Return the global ExpertRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = ExpertRegistry()
    return _registry
