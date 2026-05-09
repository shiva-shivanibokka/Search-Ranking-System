"""
Gradio Demo — Neural Search Ranking System

Tabs:
  1. Search       — run a query, see ranked results with latency breakdown
  2. A/B Compare  — run same query through both LambdaRank and CrossEncoder, compare side-by-side
  3. System Stats — live Prometheus metrics (cache hit rate, RPS, latency)
  4. Eval Results — offline evaluation table (BM25 vs TwoTower vs LambdaRank vs CrossEncoder)
"""

import os
import json
import time
import httpx
import requests
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
FEEDBACK_URL = os.getenv("FEEDBACK_URL", "http://localhost:8004")

# ── Search helpers ────────────────────────────────────────────────────────────


def run_search(query: str, ranker: str = "auto") -> tuple:
    """Call the gateway /search endpoint and return formatted results."""
    if not query.strip():
        return "Please enter a query.", pd.DataFrame(), "{}"

    payload = {
        "query": query,
        "top_k": 10,
        "ranker": None if ranker == "auto (A/B)" else ranker,
    }

    try:
        resp = httpx.post(f"{GATEWAY_URL}/search", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Error: {e}", pd.DataFrame(), "{}"

    # Format results table
    rows = []
    for r in data["results"]:
        rows.append(
            {
                "Rank": r["rank"],
                "Doc ID": r["doc_id"],
                "Score": round(r["score"], 4),
                "Ranker": r["ranker"],
                "Text Preview": r["text"][:200] + "..."
                if len(r["text"]) > 200
                else r["text"],
            }
        )
    df = pd.DataFrame(rows)

    # Format metadata
    meta = {
        "request_id": data["request_id"],
        "intent": data.get("intent"),
        "rewritten_query": data.get("rewritten_query"),
        "latency": data["latency"],
    }

    summary = (
        f"**Request ID:** `{data['request_id']}`\n\n"
        f"**Intent:** {data.get('intent', 'N/A')}\n\n"
        f"**Rewritten Query:** {data.get('rewritten_query') or '_(no rewrite)_'}\n\n"
        f"**Latency Breakdown:**\n"
        f"- Query Understanding: {data['latency'].get('query_understanding_ms', 0):.1f}ms\n"
        f"- Retrieval (FAISS): {data['latency'].get('retrieval_ms', 0):.1f}ms "
        f"({'cache hit' if data['latency'].get('cache_hit') else 'cache miss'})\n"
        f"- Ranking: {data['latency'].get('ranking_ms', 0):.1f}ms\n"
        f"- **Total: {data['latency'].get('total_ms', 0):.1f}ms**"
    )

    return summary, df, data["request_id"]


def log_click(request_id: str, query: str, doc_id: str, rank: int, ranker: str) -> str:
    """Log a click event to the feedback service."""
    if not request_id or not doc_id:
        return "No click logged."
    try:
        resp = httpx.post(
            f"{FEEDBACK_URL}/click",
            json={
                "request_id": request_id,
                "query_text": query,
                "doc_id": int(doc_id),
                "rank_shown": rank,
                "ranker_version": ranker,
            },
            timeout=5.0,
        )
        return f"Click logged (id={resp.json().get('click_id')})"
    except Exception as e:
        return f"Click log failed: {e}"


def run_ab_compare(query: str) -> tuple:
    """Run same query through both rankers and show side-by-side comparison."""
    if not query.strip():
        return pd.DataFrame(), pd.DataFrame(), None

    def search_with(ranker):
        try:
            resp = httpx.post(
                f"{GATEWAY_URL}/search",
                json={"query": query, "top_k": 10, "ranker": ranker},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    lr_data = search_with("lambdarank")
    ce_data = search_with("crossencoder")

    def format_results(data, ranker_name):
        if data is None:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "Rank": r["rank"],
                    "Doc ID": r["doc_id"],
                    "Score": round(r["score"], 4),
                    "Text": r["text"][:150] + "..."
                    if len(r["text"]) > 150
                    else r["text"],
                }
                for r in data["results"]
            ]
        )

    lr_df = format_results(lr_data, "LambdaRank")
    ce_df = format_results(ce_data, "CrossEncoder")

    # Latency comparison bar chart
    fig = None
    if lr_data and ce_data:
        fig = go.Figure(
            data=[
                go.Bar(
                    name="LambdaRank",
                    x=["Retrieval", "Ranking", "Total"],
                    y=[
                        lr_data["latency"].get("retrieval_ms", 0),
                        lr_data["latency"].get("ranking_ms", 0),
                        lr_data["latency"].get("total_ms", 0),
                    ],
                    marker_color="steelblue",
                ),
                go.Bar(
                    name="CrossEncoder",
                    x=["Retrieval", "Ranking", "Total"],
                    y=[
                        ce_data["latency"].get("retrieval_ms", 0),
                        ce_data["latency"].get("ranking_ms", 0),
                        ce_data["latency"].get("total_ms", 0),
                    ],
                    marker_color="coral",
                ),
            ]
        )
        fig.update_layout(
            title="Latency Comparison (ms)",
            barmode="group",
            yaxis_title="Milliseconds",
            template="plotly_dark",
        )

    return lr_df, ce_df, fig


def load_eval_results() -> tuple:
    """Load offline evaluation results from data/processed/eval_results.json."""
    results_path = "data/processed/eval_results.json"
    try:
        with open(results_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return pd.DataFrame(), None

    rows = []
    for config, metrics in data.items():
        rows.append(
            {
                "Config": config,
                "NDCG@10": round(metrics.get("NDCG@10", 0), 4),
                "MAP@10": round(metrics.get("MAP@10", 0), 4),
                "MRR@10": round(metrics.get("MRR@10", 0), 4),
                "Recall@10": round(metrics.get("Recall@10", 0), 4),
                "Recall@100": round(metrics.get("Recall@100", 0), 4),
                "p50 lat (ms)": round(metrics.get("latency_p50_ms", 0), 1),
                "p95 lat (ms)": round(metrics.get("latency_p95_ms", 0), 1),
            }
        )

    df = pd.DataFrame(rows)

    # Latency vs quality scatter
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["p50 lat (ms)"]],
                y=[row["NDCG@10"]],
                mode="markers+text",
                text=[row["Config"]],
                textposition="top center",
                marker=dict(size=14),
                name=row["Config"],
            )
        )

    fig.update_layout(
        title="Quality vs Latency Tradeoff",
        xaxis_title="p50 Latency (ms)",
        yaxis_title="NDCG@10",
        template="plotly_dark",
        showlegend=False,
    )

    return df, fig


def get_system_stats() -> str:
    """Fetch click stats from feedback service."""
    try:
        resp = httpx.get(f"{FEEDBACK_URL}/stats", timeout=5.0)
        stats = resp.json()
        return (
            f"**Total Clicks Logged:** {stats['total_clicks']:,}\n\n"
            f"**Retraining Threshold:** {stats['retraining_threshold']:,}\n\n"
            f"**Threshold Reached:** {'Yes — retraining will trigger' if stats['threshold_reached'] else 'No'}"
        )
    except Exception as e:
        return f"Could not fetch stats: {e}"


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Neural Search Ranking System",
    theme=gr.themes.Base(primary_hue="blue"),
) as demo:
    gr.Markdown(
        """
        # Neural Search Ranking System
        **Two-stage pipeline:** Two-Tower retrieval (FAISS IVF+PQ) → LambdaRank / CrossEncoder reranking
        over MS MARCO (500K passages). Powered by Anthropic Claude for query understanding.
        """
    )

    with gr.Tabs():
        # ── Tab 1: Search ──────────────────────────────────────────────────────
        with gr.TabItem("Search"):
            with gr.Row():
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g. what causes inflation?",
                    scale=4,
                )
                ranker_select = gr.Dropdown(
                    choices=["auto (A/B)", "lambdarank", "crossencoder"],
                    value="auto (A/B)",
                    label="Ranker",
                    scale=1,
                )
                search_btn = gr.Button("Search", variant="primary", scale=1)

            search_meta = gr.Markdown(label="Query Analysis & Latency")
            search_results = gr.Dataframe(
                label="Results",
                wrap=True,
                interactive=False,
            )
            request_id_state = gr.State("")

            with gr.Row(visible=False) as click_row:
                click_docid = gr.Textbox(label="Doc ID to mark as clicked")
                click_rank = gr.Number(label="Rank", value=1, minimum=1, maximum=10)
                click_btn = gr.Button("Log Click (implicit feedback)")
                click_status = gr.Textbox(label="Status", interactive=False)

            search_btn.click(
                fn=run_search,
                inputs=[query_input, ranker_select],
                outputs=[search_meta, search_results, request_id_state],
            )
            click_btn.click(
                fn=log_click,
                inputs=[
                    request_id_state,
                    query_input,
                    click_docid,
                    click_rank,
                    ranker_select,
                ],
                outputs=[click_status],
            )

        # ── Tab 2: A/B Compare ─────────────────────────────────────────────────
        with gr.TabItem("A/B Compare"):
            gr.Markdown(
                "Run the same query through **LambdaRank** and **CrossEncoder** side by side."
            )
            with gr.Row():
                ab_query = gr.Textbox(
                    label="Query",
                    placeholder="e.g. machine learning vs deep learning",
                    scale=5,
                )
                ab_btn = gr.Button("Compare", variant="primary", scale=1)

            ab_chart = gr.Plot(label="Latency Comparison")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### LambdaRank Results")
                    lr_results = gr.Dataframe(wrap=True, interactive=False)
                with gr.Column():
                    gr.Markdown("### CrossEncoder Results")
                    ce_results = gr.Dataframe(wrap=True, interactive=False)

            ab_btn.click(
                fn=run_ab_compare,
                inputs=[ab_query],
                outputs=[lr_results, ce_results, ab_chart],
            )

        # ── Tab 3: Eval Results ────────────────────────────────────────────────
        with gr.TabItem("Offline Evaluation"):
            gr.Markdown(
                "Full evaluation on MS MARCO dev set (6,980 queries).\n"
                "Comparison: **BM25** → **Two-Tower** → **Two-Tower+LambdaRank** → **Two-Tower+CrossEncoder**"
            )
            eval_btn = gr.Button("Load Evaluation Results")
            eval_table = gr.Dataframe(label="Metrics", interactive=False)
            eval_chart = gr.Plot(label="Quality vs Latency Tradeoff")

            eval_btn.click(
                fn=load_eval_results,
                inputs=[],
                outputs=[eval_table, eval_chart],
            )

        # ── Tab 4: System Stats ────────────────────────────────────────────────
        with gr.TabItem("System Stats"):
            gr.Markdown(
                "Live system statistics. "
                "For full metrics, see **Grafana** at `http://localhost:3000`."
            )
            stats_btn = gr.Button("Refresh Stats")
            stats_display = gr.Markdown()

            gr.Markdown(
                """
                ### Service URLs
                | Service | URL |
                |---|---|
                | API Gateway | http://localhost:8000/docs |
                | Query Understanding | http://localhost:8001/docs |
                | Retrieval | http://localhost:8002/docs |
                | Ranking | http://localhost:8003/docs |
                | Feedback | http://localhost:8004/docs |
                | MLflow | http://localhost:5001 |
                | Airflow | http://localhost:8080 |
                | Prometheus | http://localhost:9090 |
                | Grafana | http://localhost:3000 |
                """
            )

            stats_btn.click(fn=get_system_stats, inputs=[], outputs=[stats_display])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
