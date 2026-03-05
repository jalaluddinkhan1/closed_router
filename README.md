# Tri-Modal Adaptive Orchestrator

A system-level Mixture of Experts (MoE) orchestrator that optimizes LLM workloads by scoring requests against specialized experts to ensure maximal efficiency, reduced latency, and enhanced accuracy.

```
Request → Safety Gate → MoE Router (Expert Scoring) → MoE Executor (Parallel Execution) → Aggregated Result
```

---

## System Architecture

The orchestrator utilizes a sparse Mixture of Experts architecture to route requests through the most efficient compute paradigm.

| Component | Technology | Functional Domain |
|-----------|------------|-------------------|
| **Safety Gate** | Presidio + Custom Rules | PII Sanitization, Tier Enforcement, Security |
| **MoE Router** | Qdrant Semantic Scoring | Capability-based Expert Selection |
| **MoE Executor** | Async IO Orchestrator | Parallel Execution & result aggregation |
| **Expert Layer** | Multi-Provider Models | Deterministic, Tool, and LLM specialized experts |
| **API Layer** | FastAPI | OpenAI-compatible REST Interface |
| **Observability** | Streamlit | Real-time performance and routing diagnostics |

---

## Directory Structure

```
Closed_Router/
├── app/
│   ├── main.py              # Application entry point & lifecycle
│   ├── config.py            # Type-safe configuration management
│   ├── models.py            # API schemas and metadata definitions
│   ├── proxy.py             # LiteLLM integration layer
│   ├── logger.py            # Structured telemetry & persistence
│   ├── routing_graph.py     # LangGraph MoE state machine
│   ├── experts/             # Specialized expert implementations
│   └── layers/              # Rule-based security logic
├── data/                    # SQLite and vector storage persistence
├── tests/                   # Integration and unit test suites
├── requirements.txt         # Dependency specification
└── dashboard.py             # Monitoring and analytics interface
```

---

## Quick Start

### 1. Environment Initialization

```bash
cd d:\Microsoft\Closed_Router
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

```bash
copy .env.example .env
# Configure necessary API keys in .env
```

### 3. Server Deployment

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Verification

```bash
# System Health Check
curl http://localhost:8000/health

# Standardized Chat Execution
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Analyze system latency."}]}'
```

---

## Telemetry & Logging

The system provides detailed metadata for every routing decision, enabling granular auditing of model selection and cost optimization.

### `POST /v1/chat/completions`

Response payload enriched with `routing_metadata`:

```json
{
  "id": "chatcmpl-0123456789",
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Analysis complete."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 4,
    "total_tokens": 16
  },
  "routing_metadata": {
    "decision_layer": "moe_router",
    "decision_reason": "moe_experts_used=fast_llm,math_evaluator",
    "model_selected": "gpt-4o-mini",
    "confidence": 0.942,
    "latency_ms": 142.5,
    "pii_detected": false,
    "pii_entities": [],
    "execution_mode": "mode3_agentic",
    "experts_selected": ["fast_llm", "math_evaluator"],
    "experts_scores": {
      "fast_llm": 0.942,
      "math_evaluator": 0.881
    }
  }
}
```

---

## Engineering Standards

- **Pydantic V2**: Rigorous data validation and serialization.
- **Type Safety**: Comprehensive type hinting across the entire codebase.
- **Asynchronous Architecture**: Non-blocking I/O for high-concurrency performance.
- **Resilient Integrity**: Automated failover mechanisms and graceful degradation.
- **Standardized Telemetry**: Structured logging and persistent decision tracking.
