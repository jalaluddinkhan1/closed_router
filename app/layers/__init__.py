"""
app/layers/__init__.py
──────────────────────
Routing layer exports.

Active:
  layer1_rules — Safety gate (PII, blocked topics, profanity). Still used by
                 node_safety_gate in app/routing_graph.py.

Superseded by MoE (app/experts/):
  layer2_semantic — Qdrant-based semantic model selector. No longer called;
                    expert capability embeddings replace this function.
  layer3_agent    — ReAct agent runner. No longer called; WebSearchExpert and
                    PythonReplExpert in the MoE registry cover this function.
"""

from app.layers import layer1_rules

__all__ = ["layer1_rules"]
