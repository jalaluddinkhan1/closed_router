"""
app/layers/__init__.py
──────────────────────
Routing layer exports.

Active:
  layer1_rules — Safety gate (PII, blocked models, tier enforcement, token limits).
                 Called by node_safety_gate in app/routing_graph.py.

Superseded by MoE (app/experts/):
  Expert capability embeddings, WebSearchExpert, and PythonReplExpert in the
  MoE registry fully replace the old semantic and agentic layers.
"""

from app.layers import layer1_rules

__all__ = ["layer1_rules"]
