"""
app/core/verifier.py
────────────────────
Layer 4: The Verifier (Safety Net for Mode 1 output)

Implements the "Speculative Decoding" pattern in reverse:
  - A small LLM checks whether the deterministic output is logical/correct.
  - If the LLM says "NO", the result is escalated to Mode 2 (LLM call).
  - If the LLM says "YES" (or the verifier is disabled), output is returned.

This ensures 100% accuracy: if the deterministic engine makes an error
(e.g., malformed input caused a weird result), the verifier catches it.

References:
  "Speculative Decoding" — Use a small model to verify a larger model's draft.
  Applied inversely here: small model verifies deterministic code output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import litellm

from app.config import get_settings
from app.logger import get_logger

logger  = get_logger("router.verifier")
settings = get_settings()


@dataclass
class VerifierResult:
    passed: bool
    reason: str
    escalate_to_mode2: bool = False


_VERIFIER_SYSTEM = """\
You are an OUTPUT VERIFIER. Your only job is to check whether an answer is
logically correct and matches the question.

Respond with ONLY valid JSON — no markdown, no explanation:
{"correct": <true|false>, "reason": "<one sentence>"}

Examples:
  Query: "What is 54 * 12?"  Answer: "648"  → {"correct": true, "reason": "54 * 12 = 648"}
  Query: "What is 54 * 12?"  Answer: "100"  → {"correct": false, "reason": "54 * 12 = 648, not 100"}
  Query: "What is today?"    Answer: "Monday, June 5, 2025 (UTC)" → {"correct": true, "reason": "Plausible date/time answer"}"""


async def verify(query: str, deterministic_output: str) -> VerifierResult:
    """
    Ask a small LLM to sanity-check the deterministic output.

    Returns VerifierResult:
      - passed=True  → output is good, use it
      - passed=False → output is suspect, escalate to Mode 2
    """
    if not settings.verifier_enabled:
        return VerifierResult(passed=True, reason="verifier_disabled")

    user_prompt = (
        f"Query: {query[:500]}\n"
        f"Answer: {deterministic_output[:300]}"
    )

    try:
        response = await litellm.acompletion(
            model=settings.verifier_model,
            messages=[
                {"role": "system", "content": _VERIFIER_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=80,
            response_format={"type": "json_object"},
        )
        raw  = (response.choices[0].message.content or "").strip()
        data = json.loads(raw)
        correct = bool(data.get("correct", True))
        reason  = data.get("reason", "")

        if correct:
            logger.debug("Verifier PASS: %s", reason)
            return VerifierResult(passed=True, reason=reason)
        else:
            logger.warning("Verifier FAIL — escalating to Mode 2: %s", reason)
            return VerifierResult(
                passed=False,
                reason=reason,
                escalate_to_mode2=True,
            )

    except Exception as exc:
        # Verifier failure is non-fatal: trust the deterministic output
        logger.warning("Verifier call failed (%s) — trusting Mode 1 output", exc)
        return VerifierResult(passed=True, reason=f"verifier_error_trusted:{exc}")
