"""
Feedback Service — Port 8004

Receives click events from the UI/client and stores them in PostgreSQL.
These clicks are the implicit relevance signal used to retrain LambdaRank.

The Airflow DAG polls click_logs row count and triggers retraining
when RETRAINING_CLICK_THRESHOLD new clicks are accumulated since last run.
"""

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import structlog

from services.shared.logger import configure_logging, bind_request_id
from services.shared.database import get_db_session, ClickLog, create_tables

configure_logging("feedback")
logger = structlog.get_logger()

CLICK_EVENTS = Counter(
    "feedback_clicks_total", "Total click events logged", ["ranker_version"]
)


class ClickEvent(BaseModel):
    request_id: str
    query_text: str
    doc_id: int
    rank_shown: int
    ranker_version: str = "unknown"


class ClickResponse(BaseModel):
    status: str
    click_id: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        create_tables()
        logger.info("feedback.db_ready")
    except Exception as e:
        logger.warning("feedback.db_init_failed", error=str(e))
    yield


app = FastAPI(title="Feedback Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "feedback"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/click", response_model=ClickResponse)
async def log_click(event: ClickEvent):
    bind_request_id(event.request_id)
    try:
        session = get_db_session()
        try:
            click = ClickLog(
                request_id=event.request_id,
                query_text=event.query_text,
                doc_id=event.doc_id,
                rank_shown=event.rank_shown,
                ranker_version=event.ranker_version,
                clicked=True,
                created_at=datetime.utcnow(),
            )
            session.add(click)
            session.commit()
            click_id = click.id
        finally:
            session.close()

        CLICK_EVENTS.labels(ranker_version=event.ranker_version).inc()
        logger.info(
            "click.logged",
            doc_id=event.doc_id,
            rank_shown=event.rank_shown,
            ranker_version=event.ranker_version,
        )
        return ClickResponse(status="ok", click_id=click_id)

    except Exception as e:
        logger.error("click.log_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to log click")


@app.get("/stats")
async def stats():
    """Return click count stats — used by Airflow to check retraining threshold."""
    try:
        session = get_db_session()
        try:
            total_clicks = session.query(ClickLog).count()
        finally:
            session.close()
        threshold = int(os.getenv("RETRAINING_CLICK_THRESHOLD", "1000"))
        return {
            "total_clicks": total_clicks,
            "retraining_threshold": threshold,
            "threshold_reached": total_clicks >= threshold,
        }
    except Exception as e:
        logger.error("stats.failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.feedback.main:app", host="0.0.0.0", port=8004, reload=False)
