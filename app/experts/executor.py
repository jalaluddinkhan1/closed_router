"""
app/experts/executor.py
───────────────────────
MoE Parallel Executor and Result Aggregator

Execution:
  Run all selected experts simultaneously using asyncio.gather().
  This is the core performance advantage over sequential agent loops:
  a search + Python calculation that would take 4s sequentially
  takes ~2.5s in parallel.

Aggregation Strategy (priority order):
  1. Deterministic expert succeeded  → use it (100% accurate, $0)
  2. Tool experts ran (search/code)  → synthesize with FastLLM
  3. Multiple LLM experts ran        → use highest-success-rate result
  4. Single LLM expert               → return directly
  5. All failed                      → return descriptive error

Reference:
  "Mixture of Agents (MoA)" — parallel expert collaboration.
  Aggregator acts as the "proposer" in the MoA framework.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import litellm

from app.config import get_settings
from app.experts.registry import ExpertBase, ExpertResult
from app.logger import get_logger

if TYPE_CHECKING:
    from app.models import ChatRequest

logger   = get_logger("router.moe_executor")
settings = get_settings()

# Expert name sets by paradigm
_DETERMINISTIC_EXPERTS = {"math_evaluator", "datetime_handler", "json_formatter", "pii_redactor"}
_TOOL_EXPERTS          = {"web_search", "python_repl"}
_LLM_EXPERTS           = {"fast_llm", "smart_llm", "groq_llm"}


async def run_parallel(
    experts: list[tuple[ExpertBase, float]],
    query: str,
    request: "ChatRequest",
) -> list[ExpertResult]:
    """
    Execute all selected experts simultaneously.
    Returns all results (including failures) for the aggregator.
    """
    async def _safe_execute(expert: ExpertBase, score: float) -> ExpertResult:
        try:
            return await expert.execute(query, request)
        except Exception as exc:
            logger.error("Expert '%s' raised unexpectedly: %s", expert.metadata.name, exc)
            return ExpertResult(
                expert_name=expert.metadata.name,
                output=None,
                success=False,
                latency_ms=0.0,
                error=str(exc),
            )

    tasks = [_safe_execute(e, s) for e, s in experts]
    results: list[ExpertResult] = await asyncio.gather(*tasks)  # type: ignore[assignment]

    for r in results:
        status = "OK" if r.success else "FAIL"
        logger.info(
            "Expert '%s' → %s | latency=%.1fms cost=$%.6f",
            r.expert_name, status, r.latency_ms, r.cost_usd,
        )
    return results


async def _synthesize_with_llm(
    query: str,
    tool_results: list[ExpertResult],
    request: "ChatRequest",
) -> str:
    """
    Use FastLLM to synthesize / format raw tool outputs into a clean answer.
    This is the 'aggregator' in the Mixture of Agents pattern.
    """
    context_parts = []
    for r in tool_results:
        if r.output:
            context_parts.append(
                f"[{r.expert_name}]\n{r.output[:1500]}"
            )

    synthesis_prompt = (
        "You are an expert at synthesizing information from multiple sources.\n\n"
        "Given the following data collected by specialized tools:\n\n"
        + "\n\n".join(context_parts)
        + f"\n\nAnswer this question concisely and accurately:\n{query}"
    )

    try:
        response = await litellm.acompletion(
            model=settings.agent_router_model,
            messages=[
                {"role": "system",  "content": synthesis_prompt},
                {"role": "user",    "content": query},
            ],
            temperature=0.1,
            max_tokens=600,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Synthesis LLM call failed: %s", exc)
        # Fall back to concatenating tool outputs
        return "\n\n".join(r.output for r in tool_results if r.output)


async def aggregate(
    results: list[ExpertResult],
    query: str,
    request: "ChatRequest",
) -> str:
    """
    Aggregate expert results into a single final answer.

    Priority:
      1. Any deterministic expert that succeeded → return immediately
      2. Tool results (search/code) → synthesize with LLM
      3. LLM results → return the best one
      4. All failed → informative error
    """
    successes = [r for r in results if r.success and r.output]

    if not successes:
        failed_names = [r.expert_name for r in results]
        logger.warning("All experts failed: %s", failed_names)
        return (
            "I was unable to answer this question. "
            f"The following experts were tried but failed: {', '.join(failed_names)}. "
            "Please try rephrasing your question."
        )

    # ── Priority 1: Deterministic experts ────────────────────────────────────
    deterministic_results = [
        r for r in successes if r.expert_name in _DETERMINISTIC_EXPERTS
    ]
    if deterministic_results:
        # Prefer the lowest-latency deterministic result
        best = min(deterministic_results, key=lambda r: r.latency_ms)
        logger.info(
            "Aggregator: deterministic result from '%s' chosen", best.expert_name
        )
        return best.output  # type: ignore[return-value]

    # ── Priority 2: Tool results → synthesize ────────────────────────────────
    tool_results = [r for r in successes if r.expert_name in _TOOL_EXPERTS]
    if tool_results:
        logger.info(
            "Aggregator: synthesizing %d tool result(s) with LLM", len(tool_results)
        )
        return await _synthesize_with_llm(query, tool_results, request)

    # ── Priority 3: LLM results ───────────────────────────────────────────────
    llm_results = [r for r in successes if r.expert_name in _LLM_EXPERTS]
    if llm_results:
        # If multiple LLM experts ran, prefer the one with best historical success rate
        # (stored as cost_usd proxy — not ideal, but simple)
        # In practice, SmartLLM result wins over FastLLM if both ran
        priority_order = ["smart_llm", "groq_llm", "fast_llm"]
        for name in priority_order:
            for r in llm_results:
                if r.expert_name == name:
                    logger.info("Aggregator: LLM result from '%s' chosen", name)
                    return r.output  # type: ignore[return-value]
        # Fallback: first success
        return llm_results[0].output  # type: ignore[return-value]

    # ── Fallback: return first success regardless of type ────────────────────
    return successes[0].output  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# High-level API
# ─────────────────────────────────────────────────────────────────────────────


async def execute_and_aggregate(
    experts: list[tuple[ExpertBase, float]],
    query: str,
    request: "ChatRequest",
) -> tuple[str, list[ExpertResult]]:
    """
    Run experts in parallel and aggregate.
    Returns (final_answer, all_results).
    """
    results = await run_parallel(experts, query, request)
    answer  = await aggregate(results, query, request)
    return answer, results
