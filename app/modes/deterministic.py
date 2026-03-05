"""
app/modes/deterministic.py
──────────────────────────
Mode 1: Deterministic Engine (The "Calculator")

Handles tasks that require zero LLM involvement:
  - Math expressions  (AST-based safe eval — NO Python eval())
  - Date / Time queries
  - JSON prettification
  - PII Redaction (Presidio)

Cost:    $0.00
Latency: <50ms
Accuracy: 100% (deterministic)

Security: math is evaluated via the AST module, never via eval().
          Code execution is intentionally NOT in this module; it lives
          exclusively in the Agentic engine (Mode 3) with subprocess sandboxing.

References:
  FrugalGPT: "Use deterministic engines for deterministic tasks."
"""

from __future__ import annotations

import ast
import asyncio
import json
import operator
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from app.logger import get_logger

logger = get_logger("router.mode1")


@dataclass
class DeterministicResult:
    success: bool
    output: str
    handler: str        # which sub-handler fired
    error: str | None = None


_ALLOWED_OPS: dict = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.Pow:      operator.pow,
    ast.USub:     operator.neg,
    ast.UAdd:     operator.pos,
    ast.Mod:      operator.mod,
    ast.FloorDiv: operator.floordiv,
}

# "what is 54 * 12?" or "54 * 12" or "calculate 3 + 4"
_MATH_NATURAL = re.compile(
    r"(?:what\s+is\s+|calculate\s+|compute\s+|solve\s+)"
    r"([\d\s\+\-\*\/\(\)\.%\^]+)\??$",
    re.IGNORECASE,
)
_MATH_DIRECT = re.compile(
    r"^\s*([\d\s\+\-\*\/\(\)\.%\^]+)\s*[=?]?\s*$"
)
_MATH_WORD = re.compile(
    r"\b(square root of|factorial of|log of|sin of|cos of)\b",
    re.IGNORECASE,
)


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate a pure-arithmetic AST node. No side effects."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Non-numeric constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        left  = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if op_type is ast.Div and right == 0:
            raise ZeroDivisionError("Division by zero")
        return _ALLOWED_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
        return _ALLOWED_OPS[op_type](_safe_eval_node(node.operand))
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def handle_math(query: str) -> DeterministicResult | None:
    """
    Try to extract and evaluate a math expression from the query.
    Returns None if the query is not a math expression.
    """
    if _MATH_WORD.search(query):
        return None  # defer advanced math to Mode 2/3

    # Extract expression string
    m = _MATH_NATURAL.search(query)
    expr_str = m.group(1).strip() if m else None

    if expr_str is None:
        m2 = _MATH_DIRECT.match(query)
        expr_str = m2.group(1).strip() if m2 else None

    if expr_str is None:
        return None

    try:
        expr_str = expr_str.replace("^", "**")
        tree = ast.parse(expr_str, mode="eval")
        result = _safe_eval_node(tree.body)

        # Clean output: strip trailing .0 for integer results
        if result == int(result) and abs(result) < 1e15:
            output = str(int(result))
        else:
            output = f"{result:.10g}"

        logger.info("Math handler: '%s' = %s", expr_str, output)
        return DeterministicResult(success=True, output=output, handler="math_evaluator")

    except (ValueError, ZeroDivisionError, SyntaxError) as exc:
        return DeterministicResult(
            success=False, output="", handler="math_evaluator", error=str(exc)
        )


_DATETIME_PATTERN = re.compile(
    r"\b(what.*(time|date|day)|current (time|date|datetime)|today['s]?\s*(date|day)|"
    r"what\s+day\s+is\s+(it|today))\b",
    re.IGNORECASE,
)


def handle_datetime(query: str) -> DeterministicResult | None:
    if not _DATETIME_PATTERN.search(query):
        return None

    now = datetime.now(timezone.utc)
    q_lower = query.lower()

    if "time" in q_lower and "date" not in q_lower:
        output = now.strftime("The current UTC time is %H:%M:%S.")
    elif "date" in q_lower and "time" not in q_lower:
        output = now.strftime("Today's date is %A, %B %d, %Y (UTC).")
    else:
        output = now.strftime("Current UTC datetime: %A, %B %d, %Y at %H:%M:%S.")

    logger.info("Datetime handler triggered")
    return DeterministicResult(success=True, output=output, handler="datetime_handler")


_JSON_FORMAT_TRIGGER = re.compile(
    r"\b(format|prettify|pretty.?print|indent|beautify|parse)\b.{0,30}\bjson\b|"
    r"\bjson\b.{0,30}\b(format|prettify|pretty.?print|indent|beautify)\b",
    re.IGNORECASE | re.DOTALL,
)
_JSON_BLOB = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def handle_json_format(query: str) -> DeterministicResult | None:
    if not _JSON_FORMAT_TRIGGER.search(query):
        return None

    m = _JSON_BLOB.search(query)
    if not m:
        return None

    try:
        data   = json.loads(m.group())
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        logger.info("JSON formatter: %d chars formatted", len(pretty))
        return DeterministicResult(
            success=True,
            output=f"```json\n{pretty}\n```",
            handler="json_formatter",
        )
    except json.JSONDecodeError:
        return None


_PII_REDACT_TRIGGER = re.compile(
    r"\b(redact|mask|remove|hide|anonymize|anonymise|censor|scrub)\b.{0,40}"
    r"\b(pii|ssn|email|phone|credit.?card|password|personal|private|sensitive)\b",
    re.IGNORECASE | re.DOTALL,
)


async def handle_pii_redaction(query: str) -> DeterministicResult | None:
    """
    When a user explicitly asks to redact PII from text, this handler runs
    Presidio directly and returns the redacted text — no LLM call needed.
    """
    if not _PII_REDACT_TRIGGER.search(query):
        return None

    try:
        from app.pii_engine import _get_analyzer  # type: ignore[attr-defined]

        analyzer = await asyncio.to_thread(_get_analyzer)
        if analyzer is None:
            return None

        def _run_redact() -> str:
            results = analyzer.analyze(text=query, language="en")
            if not results:
                return query
            results.sort(key=lambda r: r.start, reverse=True)
            redacted = query
            for r in results:
                redacted = redacted[: r.start] + f"[{r.entity_type}]" + redacted[r.end :]
            return redacted

        redacted_text = await asyncio.to_thread(_run_redact)
        logger.info("PII redaction handler: query redacted")
        return DeterministicResult(
            success=True,
            output=redacted_text,
            handler="pii_redactor",
        )

    except Exception as exc:
        logger.warning("PII redaction handler failed: %s", exc)
        return None


async def run(query: str) -> DeterministicResult | None:
    """
    Try all deterministic handlers in priority order.
    Returns the first successful result, or None if no handler applies.

    Priority:
      1. Math evaluator   — fastest, zero external deps
      2. Date/time        — zero external deps
      3. JSON formatter   — zero external deps
      4. PII redactor     — requires Presidio (async)
    """
    for sync_handler in (handle_math, handle_datetime, handle_json_format):
        result = sync_handler(query)
        if result is not None:
            return result

    # Async handler
    return await handle_pii_redaction(query)
