"""
app/routing_graph.py
────────────────────
Tri-Modal Adaptive Orchestrator — LangGraph State Machine (MoE Edition)

Replaces the old 6-node tri-modal graph with a clean 3-node MoE graph:

  START
    │
    ▼
  safety_gate    ← Layer 1 Rules (PII, tier, blocked models)
    │ if rule fired ─────────────────────────────────→ END
    │
    ▼
  moe_route      ← MoE Expert Router
    │               Score all experts (embedding sim + cost + latency + reliability)
    │               Select Top-K experts
    │
    ▼
  moe_execute    ← MoE Executor
    │               Run selected experts in parallel (asyncio.gather)
    │               Aggregate results (deterministic > tool > LLM priority)
    │               Optional: verifier pass on deterministic outputs
    │
    ▼
  END

The RoutingDecision return type from route_request():
  - routing.experts_selected  : list of expert names chosen
  - routing.execution_mode    : derived from which experts succeeded
  - direct_result             : if set, chat.py skips LiteLLM call
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.experts.executor import execute_and_aggregate
from app.experts.registry import ExpertBase, ExpertResult, get_registry
from app.experts.router import select_experts
from app.layers import layer1_rules
from app.logger import get_logger
from app.models import ChatRequest, ExecutionMode, RoutingMetadata

logger   = get_logger("router.graph")
settings = get_settings()

_DETERMINISTIC_NAMES = {"math_evaluator", "datetime_handler", "json_formatter", "pii_redactor"}
_TOOL_NAMES          = {"web_search", "python_repl"}


@dataclass
class RoutingDecision:
    """
    Everything the chat endpoint needs to dispatch the request.

    routing       : metadata attached to every API response.
    direct_result : if set, skip LiteLLM and return this string as the
                    assistant content (set by all non-Mode-2 paths).
    """
    routing: RoutingMetadata
    direct_result: str | None = None


class RouterState(TypedDict):
    request:          ChatRequest
    routing:          RoutingMetadata | None
    # MoE intermediate state
    selected_experts: list[tuple]      # list of (ExpertBase, score) — not serializable
    expert_results:   list[ExpertResult]
    direct_result:    str | None       # aggregated output; None → use LiteLLM


async def node_safety_gate(state: RouterState) -> RouterState:
    """
    Run hard safety rules (PII, tier gate, blocked models, token limits).
    If a rule fires we short-circuit: set routing and go directly to END.
    The safety-selected model is used as a Mode 2 (single LLM) call.
    """
    logger.debug("Graph → safety_gate")
    result = await layer1_rules.run(state["request"])
    if result is not None:
        result.execution_mode = ExecutionMode.PROBABILISTIC
        logger.info(
            "Safety gate fired | reason=%s model=%s",
            result.decision_reason, result.model_selected,
        )
    return {**state, "routing": result}


async def node_moe_route(state: RouterState) -> RouterState:
    """
    Score all registered experts and select the Top-K to run in parallel.
    Stores (expert, score) pairs in state for the executor node.
    """
    logger.debug("Graph → moe_route")
    selected = await select_experts(state["request"], get_registry())
    logger.info(
        "MoE route: selected=[%s]",
        ", ".join(f"{e.metadata.name}({s:.3f})" for e, s in selected),
    )
    return {**state, "selected_experts": selected}


def _derive_execution_mode(results: list[ExpertResult]) -> ExecutionMode:
    """Derive the execution mode from which experts actually succeeded."""
    successful_names = {r.expert_name for r in results if r.success}
    if successful_names & _DETERMINISTIC_NAMES:
        return ExecutionMode.DETERMINISTIC
    if successful_names & _TOOL_NAMES:
        return ExecutionMode.AGENTIC
    return ExecutionMode.PROBABILISTIC


async def node_moe_execute(state: RouterState) -> RouterState:
    """
    Run selected experts in parallel, aggregate results, and build
    the RoutingMetadata that will be attached to the response.
    """
    logger.debug("Graph → moe_execute")
    request  = state["request"]
    selected: list[tuple[ExpertBase, float]] = state["selected_experts"]

    if not selected:
        # No experts selected — shouldn't happen, but handle defensively
        logger.error("No experts selected — falling back to fast_llm")
        fallback = get_registry().get("fast_llm")
        if fallback:
            selected = [(fallback, 0.5)]
        else:
            routing = RoutingMetadata(
                decision_layer="moe_router",
                decision_reason="no_experts_available",
                model_selected=settings.default_model,
                confidence=0.3,
                execution_mode=ExecutionMode.PROBABILISTIC,
            )
            return {**state, "routing": routing, "direct_result": None}

    # Extract user query
    user_msgs = [m.content for m in request.messages if m.role == "user" and m.content]
    query = user_msgs[-1].strip() if user_msgs else ""

    # Run in parallel + aggregate
    final_answer, expert_results = await execute_and_aggregate(selected, query, request)

    # Derive metadata
    exec_mode = _derive_execution_mode(expert_results)
    expert_names  = [e.metadata.name for e, _ in selected]
    expert_scores = {e.metadata.name: round(s, 3) for e, s in selected}

    # Best confidence = score of highest-scored expert
    top_score = selected[0][1] if selected else 0.5

    # Cost saved = sum of savings from deterministic experts
    cost_saved = sum(
        r.latency_ms * 0.0 +  # no cost for deterministic
        settings.mode2_reference_cost_per_1k * (max(50, len(query) // 4) / 1000)
        for r in expert_results
        if r.success and r.expert_name in _DETERMINISTIC_NAMES
    )

    # Total actual cost of LLM experts used
    # (deterministic + tool experts have cost_usd=0 in their results)
    total_cost = sum(r.cost_usd for r in expert_results if r.success)

    # Determine primary model name for the response
    successful_llm = next(
        (r for r in expert_results if r.success and r.expert_name not in _DETERMINISTIC_NAMES and r.expert_name not in _TOOL_NAMES),
        None,
    )
    if exec_mode == ExecutionMode.DETERMINISTIC:
        model_name = "deterministic_engine"
    elif successful_llm:
        # Map expert name to model string
        _expert_models = {
            "fast_llm":  settings.agent_router_model,
            "smart_llm": "gpt-4o",
            "groq_llm":  "groq/llama-3.1-70b-versatile",
        }
        model_name = _expert_models.get(successful_llm.expert_name, settings.default_model)
    else:
        model_name = settings.default_model

    # Build reason string
    success_names  = [r.expert_name for r in expert_results if r.success]
    fail_names     = [r.expert_name for r in expert_results if not r.success]
    reason_parts   = [f"moe_experts_used={','.join(success_names)}"]
    if fail_names:
        reason_parts.append(f"failed={','.join(fail_names)}")

    routing = RoutingMetadata(
        decision_layer="moe_router",
        decision_reason="::".join(reason_parts),
        model_selected=model_name,
        confidence=top_score,
        execution_mode=exec_mode,
        cost_saved_usd=cost_saved,
        experts_selected=expert_names,
        experts_scores=expert_scores,
    )

    # For Mode 2 (LLM-only), we return direct_result=None so chat.py calls LiteLLM.
    # But since we already ran the LLM expert here, we have the answer — return it directly.
    # This avoids a second LLM call in the chat endpoint.
    return {
        **state,
        "routing":        routing,
        "expert_results": expert_results,
        "direct_result":  final_answer,  # always set — MoE always produces output
    }


def after_safety_gate(
    state: RouterState,
) -> Literal["moe_route", "__end__"]:
    """If safety gate fired, go to END (Mode 2 call done by chat.py)."""
    if state["routing"] is not None:
        return "__end__"
    return "moe_route"


def build_routing_graph() -> Any:
    """
    Compile the MoE LangGraph state machine.

    Topology: safety_gate → moe_route → moe_execute → END
    """
    graph = StateGraph(RouterState)

    graph.add_node("safety_gate", node_safety_gate)
    graph.add_node("moe_route",   node_moe_route)
    graph.add_node("moe_execute", node_moe_execute)

    graph.set_entry_point("safety_gate")

    graph.add_conditional_edges(
        "safety_gate",
        after_safety_gate,
        {"moe_route": "moe_route", "__end__": END},
    )
    graph.add_edge("moe_route",   "moe_execute")
    graph.add_edge("moe_execute", END)

    compiled = graph.compile()
    logger.info("MoE routing graph compiled: safety_gate → moe_route → moe_execute")
    return compiled


_compiled_graph: Any | None = None


def get_routing_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_routing_graph()
    return _compiled_graph


async def route_request(request: ChatRequest) -> RoutingDecision:
    """
    Run a ChatRequest through the MoE orchestrator.

    Returns RoutingDecision:
      - direct_result is always set by MoE (experts already produced output)
      - When the safety gate fires, direct_result=None → chat.py calls LiteLLM
    """
    graph = get_routing_graph()

    initial_state: RouterState = {
        "request":          request,
        "routing":          None,
        "selected_experts": [],
        "expert_results":   [],
        "direct_result":    None,
    }

    final_state   = await graph.ainvoke(initial_state)
    routing       = final_state.get("routing")
    direct_result = final_state.get("direct_result")

    if routing is None:
        logger.error("Graph returned no routing decision — using default model")
        routing = RoutingMetadata(
            decision_layer="moe_router",
            decision_reason="graph_fallback_no_decision",
            model_selected=request.model or settings.default_model,
            confidence=0.3,
            execution_mode=ExecutionMode.PROBABILISTIC,
        )

    return RoutingDecision(routing=routing, direct_result=direct_result)
