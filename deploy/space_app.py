"""
Hugging Face Space entrypoint — consolidated single-process demo.

Runs the whole search pipeline in one process (see deploy/engine.py) behind a
Gradio UI, sized for a free Space. Optional integrations, all degrade gracefully:
  * LLM_PROVIDER (groq/gemini/openai/anthropic) -> HyDE + intent; default none
  * DATABASE_URL (Neon)                          -> click logging
Artifacts are pulled by scripts/bootstrap.py at container build/start.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.engine import SearchEngine, classify_intent  # noqa: E402
from services.shared.llm import get_llm_provider  # noqa: E402

print("Loading search engine (models + indexes)...", flush=True)
ENGINE = SearchEngine()
LLM = get_llm_provider()
print(f"Engine ready. LLM provider: {LLM.name} (available={LLM.available})", flush=True)


def _maybe_log_click(request_id: str, query: str, doc_id: int, rank: int, ranker: str) -> str:
    """Best-effort click logging to Postgres (Neon). No-op if DB not configured."""
    import os

    if not os.getenv("DATABASE_URL") and not os.getenv("POSTGRES_HOST"):
        return "Click logging disabled (no DATABASE_URL set)."
    try:
        from datetime import datetime

        from services.shared.database import ClickLog, create_tables, get_db_session

        create_tables()
        session = get_db_session()
        try:
            session.add(ClickLog(
                request_id=request_id, query_text=query, doc_id=int(doc_id),
                rank_shown=int(rank), ranker_version=ranker, clicked=True,
                created_at=datetime.utcnow(),
            ))
            session.commit()
        finally:
            session.close()
        return f"Logged click on doc {doc_id} (rank {rank})."
    except Exception as e:  # pragma: no cover - depends on live DB
        return f"Click logging failed: {e}"


def search(query: str, ranker: str, top_k: int):
    if not query or not query.strip():
        return "Enter a query.", [], ""

    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    intent = classify_intent(query)

    # Optional HyDE: embed a hypothetical answer instead of the raw query.
    embed_text = query
    hyde_used = False
    if LLM.available and intent == "informational":
        try:
            embed_text = LLM.complete(
                "Write a short, factual passage (2-4 sentences) that would be the "
                "ideal answer to the user's question. Be specific and factual.",
                f"Question: {query}",
                max_tokens=256,
            )
            hyde_used = True
        except Exception:
            embed_text = query

    chosen = "crossencoder" if ranker == "crossencoder" else "lambdarank"
    cands = ENGINE.retrieve(query, embed_text, top_k=100)
    results = ENGINE.rank(query, cands, top_k=int(top_k), ranker=chosen)
    elapsed = (time.perf_counter() - t0) * 1000

    rows = [[r["rank"], r["doc_id"], r["text"][:300], round(r["score"], 4), r["ranker"]]
            for r in results]
    meta = (
        f"**Intent:** {intent}  |  **Ranker:** {chosen}  |  "
        f"**HyDE:** {'yes' if hyde_used else 'no'}  |  "
        f"**Latency:** {elapsed:.0f} ms  |  **request_id:** `{request_id}`"
    )
    return meta, rows, request_id


with gr.Blocks(title="Neural Search Ranking System") as demo:
    gr.Markdown(
        "# Neural Search Ranking System\n"
        "Two-stage neural search over 500K MS MARCO passages: hybrid retrieval "
        "(FAISS + BM25 via Reciprocal Rank Fusion) then learned reranking. "
        "Running consolidated in a single process for the free demo."
    )
    with gr.Row():
        query = gr.Textbox(label="Search query", scale=4, placeholder="what causes inflation")
        ranker = gr.Dropdown(["auto (lambdarank)", "crossencoder"], value="auto (lambdarank)", label="Ranker")
        top_k = gr.Slider(1, 20, value=10, step=1, label="Top K")
    search_btn = gr.Button("Search", variant="primary")
    meta_out = gr.Markdown()
    results_out = gr.Dataframe(
        headers=["rank", "doc_id", "text", "score", "ranker"],
        label="Results", wrap=True,
    )
    rid_state = gr.State("")

    search_btn.click(
        lambda q, rk, k: search(q, "crossencoder" if "crossencoder" in rk else "lambdarank", k),
        [query, ranker, top_k], [meta_out, results_out, rid_state],
    )

    with gr.Accordion("Log a click (feeds the retraining signal)", open=False):
        click_doc = gr.Number(label="doc_id to log", precision=0)
        click_rank = gr.Number(label="rank shown", precision=0, value=1)
        click_btn = gr.Button("Log click")
        click_status = gr.Markdown()
        click_btn.click(
            lambda rid, q, d, r: _maybe_log_click(rid, q, d, r, "lambdarank"),
            [rid_state, query, click_doc, click_rank], [click_status],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
