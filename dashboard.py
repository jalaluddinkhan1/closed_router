"""
dashboard.py
────────────
Tri-Modal Adaptive Orchestrator — Streamlit Dashboard

Fetches data from the local FastAPI /v1/stats endpoints and renders:
  - Cost savings KPI (Mode 1 vs always-Mode-2 baseline)
  - Execution mode distribution (Mode 1 / Mode 2 / Mode 3)
  - Routing layer distribution (which classifier layer fired)
  - Model usage and cost breakdown
  - Traffic and latency over time
  - Recent request log with mode badges

Run with:
  streamlit run dashboard.py
"""

import time
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000/v1"

st.set_page_config(
    page_title="Tri-Modal Orchestrator | Enterprise Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    
    .stButton>button {
        border-radius: 6px;
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #334155;
        border-color: #475569;
    }

    /* Professional Dataframe Styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

MODE_LABELS = {
    "mode1_deterministic": "Deterministic",
    "mode2_probabilistic": "Probabilistic",
    "mode3_agentic":       "Agentic",
}
MODE_COLORS = {
    "mode1_deterministic": "#10b981",  # emerald-500
    "mode2_probabilistic": "#6366f1",  # indigo-500
    "mode3_agentic":       "#f59e0b",  # amber-500
}


@st.cache_data(ttl=5)
def fetch_stats() -> dict[str, Any] | None:
    try:
        r = httpx.get(f"{API_BASE_URL}/stats", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Failed to fetch stats: {exc}")
        return None


@st.cache_data(ttl=5)
def fetch_hourly() -> list[dict[str, Any]]:
    try:
        r = httpx.get(f"{API_BASE_URL}/stats/hourly?hours=24", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


@st.cache_data(ttl=5)
def fetch_recent() -> list[dict[str, Any]]:
    try:
        r = httpx.get(f"{API_BASE_URL}/stats/recent?limit=100", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def render_header() -> None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Tri-Modal Adaptive Orchestrator")
        st.markdown(
            "Enterprise-grade compute paradigm routing for optimized cost, latency, and performance."
        )
    with col2:
        if st.button("Refresh Dashboard", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def render_kpi_cards(stats: dict[str, Any]) -> None:
    total_reqs   = stats.get("total_requests", 0)
    total_cost   = stats.get("total_cost_usd", 0.0)
    total_saved  = stats.get("total_saved_usd", 0.0)
    pii_blocks   = stats.get("pii_stats", {}).get("total_blocks", 0)

    models = stats.get("requests_per_model", [])
    avg_lat = (
        sum(m["avg_latency_ms"] * m["count"] for m in models) / total_reqs
        if total_reqs > 0 and models else 0.0
    )

    # Mode 1 request count
    mode_dist  = stats.get("mode_distribution", [])
    mode1_count = next(
        (m["count"] for m in mode_dist if m["mode"] == "mode1_deterministic"), 0
    )
    mode1_pct = (mode1_count / total_reqs * 100) if total_reqs > 0 else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Traffic", f"{total_reqs:,}")
    with col2:
        st.metric("Operational Cost", f"${total_cost:.4f}")
    with col3:
        st.metric(
            "Efficiency Savings",
            f"${total_saved:.4f}",
            delta=f"{mode1_pct:.1f}% Opt",
            delta_color="normal",
        )
    with col4:
        st.metric("Mean Latency", f"{avg_lat:.0f} ms")
    with col5:
        st.metric("Security Blocks", f"{pii_blocks:,}")


def render_mode_distribution(stats: dict[str, Any]) -> None:
    """Show the tri-modal execution breakdown."""
    mode_dist = stats.get("mode_distribution", [])
    if not mode_dist:
        st.info("No mode distribution data yet.")
        return

    df = pd.DataFrame(mode_dist)
    df["label"] = df["mode"].map(lambda m: MODE_LABELS.get(m, m))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mode Distribution")
        fig = px.pie(
            df,
            values="count",
            names="label",
            hole=0.6,
            color="mode",
            color_discrete_map={
                m: MODE_COLORS[m] for m in df["mode"] if m in MODE_COLORS
            },
            template="plotly_dark",
        )
        fig.update_layout(
            margin=dict(t=30, b=0, l=0, r=0),
            font=dict(family="Inter, sans-serif"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Metric Comparison")
        fig = px.bar(
            df,
            x="label",
            y=["count", "avg_latency_ms"],
            barmode="group",
            labels={"value": "Quantity", "variable": "Dimension", "label": "Mode"},
            color_discrete_sequence=["#6366f1", "#10b981"],
            template="plotly_dark",
        )
        fig.update_layout(
            margin=dict(t=30, b=0, l=0, r=0),
            font=dict(family="Inter, sans-serif"),
            xaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cost savings table
    savings_df = df[["label", "count", "total_saved_usd", "avg_latency_ms"]].copy()
    savings_df.columns = ["Mode", "Requests", "Cost Saved ($)", "Avg Latency (ms)"]
    st.dataframe(savings_df, use_container_width=True, hide_index=True)


def render_routing_charts(stats: dict[str, Any], hourly: list[dict[str, Any]]) -> None:
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Routing Distribution")
        dist_data = stats.get("routing_distribution", [])
        if dist_data:
            df_dist = pd.DataFrame(dist_data)
            fig = px.pie(
                df_dist,
                values="count",
                names="layer",
                hole=0.6,
                color="layer",
                color_discrete_map={
                    "layer1_rules":         "#ef4444",
                    "layer2_semantic":      "#10b981",
                    "layer3_agent":         "#6366f1",
                    "heuristic_layer1":     "#f59e0b",
                    "semantic_layer2":      "#0ea5e9",
                    "llm_classifier_layer3":"#d946ef",
                },
                template="plotly_dark",
            )
            fig.update_layout(
                margin=dict(t=30, b=0, l=0, r=0),
                font=dict(family="Inter, sans-serif"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No routing data yet.")

    with col2:
        st.subheader("Model Utilization")
        model_data = stats.get("requests_per_model", [])
        if model_data:
            df_models = pd.DataFrame(model_data)
            fig = px.bar(
                df_models,
                x="model",
                y="count",
                color="total_cost",
                color_continuous_scale="Blues",
                labels={"count": "Requests", "total_cost": "Cost ($)"},
                template="plotly_dark",
            )
            fig.update_layout(
                margin=dict(t=30, b=0, l=0, r=0),
                font=dict(family="Inter, sans-serif"),
                xaxis_title=None,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model data yet.")

    st.markdown("---")
    st.subheader("Traffic & Latency Over Time (Last 24h)")
    if hourly:
        df_hourly = pd.DataFrame(hourly)
        df_hourly["time"] = pd.to_datetime(df_hourly["hour"], unit="s")

        col_time1, col_time2 = st.columns(2)
        with col_time1:
            fig_vol = px.line(
                df_hourly, x="time", y="count", markers=True, 
                title="Request Volume",
                template="plotly_dark"
            )
            fig_vol.update_layout(
                margin=dict(t=30, b=0, l=0, r=0),
                font=dict(family="Inter, sans-serif"),
                xaxis_title=None,
                yaxis_title="Requests"
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        with col_time2:
            fig_lat = px.line(
                df_hourly,
                x="time",
                y="avg_latency_ms",
                markers=True,
                title="Average Latency (ms)",
                color_discrete_sequence=["#38bdf8"],
                template="plotly_dark"
            )
            fig_lat.update_layout(
                margin=dict(t=30, b=0, l=0, r=0),
                font=dict(family="Inter, sans-serif"),
                xaxis_title=None,
                yaxis_title="Latency (ms)"
            )
            st.plotly_chart(fig_lat, use_container_width=True)
    else:
        st.info("No hourly time-series data yet.")


def render_recent_table(recent: list[dict[str, Any]]) -> None:
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Recent Requests Log")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)

    if recent:
        df_recent = pd.DataFrame(recent)
        df_recent["time"] = pd.to_datetime(
            df_recent["timestamp"], unit="s"
        ).dt.strftime("%H:%M:%S")
        df_recent["cost"]    = df_recent["estimated_cost"].apply(lambda x: f"${x:.6f}")
        df_recent["latency"] = df_recent["latency_ms"].apply(lambda x: f"{x:.0f}ms")

        mode_badge = {
            "mode1_deterministic": "M1: Deterministic",
            "mode2_probabilistic": "M2: Probabilistic",
            "mode3_agentic":       "M3: Agentic",
        }
        layer_badge = {
            "layer1_rules":          "Security",
            "layer2_semantic":       "Semantic",
            "layer3_agent":          "Agent",
            "heuristic_layer1":      "Heuristic",
            "semantic_layer2":       "Semantic",
            "llm_classifier_layer3": "Classifier",
        }

        df_recent["mode"]  = df_recent.get("execution_mode", pd.Series(dtype=str)).map(
            lambda x: mode_badge.get(x, x) if isinstance(x, str) else "—"
        )
        df_recent["layer"] = df_recent["decision_layer"].map(
            lambda x: layer_badge.get(x, x)
        )

        display_cols = [
            "time", "mode", "layer", "model_used",
            "user_tier", "latency", "cost", "pii_detected",
        ]
        # Only include columns that exist
        display_cols = [c for c in display_cols if c in df_recent.columns]

        st.dataframe(
            df_recent[display_cols],
            use_container_width=True,
            hide_index=True,
            height=400,
        )
    else:
        st.info("No recent requests.")

    if auto_refresh:
        time.sleep(5)
        st.rerun()


def main() -> None:
    render_header()

    stats = fetch_stats()
    if stats is None:
        st.warning("Cannot connect to the router API. Is it running on port 8000?")
        return

    hourly = fetch_hourly()
    recent = fetch_recent()

    render_kpi_cards(stats)

    st.markdown("---")
    st.subheader("Tri-Modal Execution Breakdown")
    render_mode_distribution(stats)

    render_routing_charts(stats, hourly)
    render_recent_table(recent)


if __name__ == "__main__":
    main()
