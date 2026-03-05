"""
app/modes/agentic.py
────────────────────
Mode 3: Agentic Engine (The "Researcher")

Implements a ReAct (Reasoning + Acting) agent using LiteLLM's tool-calling
API. The agent has access to:
  1. web_search   — Tavily-powered web search for live/current information.
  2. python_repl  — Sandboxed Python execution via subprocess with timeout.

The agent runs in a loop (max AGENTIC_MAX_STEPS iterations):
  1. LLM receives the user query + tool schemas
  2. If LLM calls a tool → run the tool → append result → repeat
  3. If LLM returns a final answer → return it
  4. If timeout or max steps reached → return best partial answer

Security:
  python_repl runs user-agent-generated code in a subprocess with:
    - Hard timeout (10s per execution)
    - No network access to the host process
  For production, replace subprocess with E2B or a Docker container.

References:
  ReAct: "Synergizing Reasoning and Acting in Language Models" (Yao et al.)
  Chain of Thought: Encouraged via the system prompt (step-by-step reasoning).
  Mixture of Agents: Agent can call sub-tools (search + code) in sequence.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

import litellm

from app.config import get_settings
from app.logger import get_logger
from app.models import ChatRequest

logger   = get_logger("router.mode3")
settings = get_settings()


@dataclass
class AgenticResult:
    success: bool
    output: str
    steps_taken: int
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None


_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current, real-time information. "
                "Use when the query requires live data, news, recent events, "
                "or facts that may have changed after your training cutoff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_repl",
            "description": (
                "Execute Python code in a sandboxed environment. "
                "Use for calculations, data manipulation, or generating structured output. "
                "Output is captured from stdout. Keep code concise and self-contained."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Must print() results.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────


async def _web_search(query: str) -> str:
    """
    Run a Tavily web search.
    Falls back to a descriptive error if Tavily is unavailable.
    """
    if not settings.tavily_api_key:
        return (
            "[Search unavailable: TAVILY_API_KEY not configured. "
            "Answer based on your training knowledge.]"
        )
    try:
        from tavily import TavilyClient  # type: ignore[import-not-found]

        def _run_search() -> str:
            client = TavilyClient(api_key=settings.tavily_api_key)
            resp   = client.search(
                query=query,
                max_results=4,
                search_depth="basic",
            )
            results = resp.get("results", [])
            if not results:
                return "No results found."
            parts = []
            for r in results:
                parts.append(
                    f"Source: {r.get('url', 'unknown')}\n"
                    f"{r.get('content', '')[:600]}"
                )
            return "\n\n---\n\n".join(parts)

        return await asyncio.to_thread(_run_search)

    except ImportError:
        return (
            "[Search unavailable: tavily-python not installed. "
            "Run: pip install tavily-python]"
        )
    except Exception as exc:
        logger.warning("Tavily search error: %s", exc)
        return f"[Search error: {exc}]"


async def _python_repl(code: str) -> str:
    """
    Execute Python code in a subprocess with a hard timeout.

    Security notes:
      - The subprocess inherits the current environment but cannot communicate
        back except via stdout/stderr.
      - Timeout is enforced at OS level (subprocess.run timeout).
      - For production, replace with E2B cloud sandbox or Docker container.
    """
    # Basic safety: reject obvious injection patterns
    forbidden = ["import os", "import sys", "subprocess", "__import__", "open(", "exec(", "eval("]
    code_lower = code.lower()
    for pattern in forbidden:
        if pattern in code_lower:
            return f"[Blocked: '{pattern}' is not allowed in the sandbox]"

    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = proc.stdout or proc.stderr or "(no output)"
        return output[:2000]  # cap response size

    except subprocess.TimeoutExpired:
        return "[Error: Code execution timed out after 10 seconds]"
    except Exception as exc:
        return f"[Error: {exc}]"


async def _dispatch_tool(name: str, args: dict[str, Any]) -> str:
    """Route a tool call to the correct implementation."""
    if name == "web_search":
        return await _web_search(args.get("query", ""))
    if name == "python_repl":
        return await _python_repl(args.get("code", ""))
    return f"[Unknown tool: {name}]"


# ─────────────────────────────────────────────────────────────────────────────
# ReAct Agent loop
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_SYSTEM = """\
You are an expert AI assistant with access to web search and Python execution.

Approach every task using Chain-of-Thought reasoning (ReAct pattern):
  1. THINK: Reason about what information or computation you need.
  2. ACT:   Call the appropriate tool (web_search or python_repl).
  3. OBSERVE: Read the tool result.
  4. Repeat until you have a complete answer.
  5. ANSWER: Provide a clear, concise final response.

Rules:
  - Always verify facts with web_search when the query involves current events.
  - Use python_repl for precise calculations rather than estimating.
  - Do not call the same tool with the exact same arguments twice.
  - When you have enough information, give the final answer directly."""


async def _run_react_loop(
    messages: list[dict[str, Any]],
    max_steps: int,
) -> tuple[str, list[str]]:
    """
    Core ReAct loop. Returns (final_answer, tools_used_list).
    """
    tools_used: list[str] = []

    for step in range(max_steps):
        logger.debug("Agent step %d/%d", step + 1, max_steps)

        response = await litellm.acompletion(
            model=settings.agentic_model,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=1000,
        )

        msg = response.choices[0].message

        # Append assistant message (may contain tool_calls or content)
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # No tool calls → final answer
        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
            final = (msg.content or "").strip()
            if final:
                return final, tools_used
            # Empty response — shouldn't happen but handle gracefully
            return "I was unable to generate a complete answer.", tools_used

        # Execute tool calls
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            logger.info("Agent → tool='%s' args=%s", tool_name, str(tool_args)[:120])
            tools_used.append(tool_name)

            tool_result = await _dispatch_tool(tool_name, tool_args)
            logger.debug("Tool '%s' result: %s", tool_name, tool_result[:200])

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

    # Max steps reached — ask for a final summary
    logger.warning("Agent reached max steps (%d) — requesting summary", max_steps)
    messages.append({
        "role": "user",
        "content": "You've reached the step limit. Summarize what you found so far.",
    })
    try:
        final_resp = await litellm.acompletion(
            model=settings.agentic_model,
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )
        final = (final_resp.choices[0].message.content or "").strip()
        return final or "Agent reached step limit without a complete answer.", tools_used
    except Exception:
        return "Agent reached step limit. Please try a simpler query.", tools_used


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


async def run(request: ChatRequest) -> AgenticResult:
    """
    Execute a ReAct agent loop to handle a complex, multi-step query.

    Always returns an AgenticResult (never raises).
    Falls back gracefully on timeout or LLM failure.
    """
    # Build initial message list (system + conversation history)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _AGENT_SYSTEM},
    ]
    for m in request.messages:
        messages.append({"role": m.role, "content": m.content})

    user_query = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        "",
    )

    try:
        output, tools_used = await asyncio.wait_for(
            _run_react_loop(messages, settings.agentic_max_steps),
            timeout=settings.agentic_timeout_seconds,
        )
        steps = len(tools_used) + 1
        logger.info(
            "Mode 3 complete | steps=%d tools=%s",
            steps, tools_used,
        )
        return AgenticResult(
            success=True,
            output=output,
            steps_taken=steps,
            tools_used=tools_used,
        )

    except asyncio.TimeoutError:
        logger.warning(
            "Mode 3 timed out after %.0fs — falling back",
            settings.agentic_timeout_seconds,
        )
        return AgenticResult(
            success=False,
            output=(
                f"The request timed out after {settings.agentic_timeout_seconds:.0f}s. "
                "Please try a simpler question or break it into smaller parts."
            ),
            steps_taken=settings.agentic_max_steps,
            error="timeout",
        )
    except Exception as exc:
        logger.error("Mode 3 agent error: %s", exc)
        return AgenticResult(
            success=False,
            output=f"The agentic engine encountered an error: {exc}",
            steps_taken=0,
            error=str(exc),
        )
