"""
PostgreSQL connection and schema definitions.

Tables:
  query_logs    : every search request (for analytics and retraining)
  click_logs    : user clicks on results (implicit relevance signal)
  model_versions: model promotion history (audit trail)
"""

import os
from datetime import datetime
from functools import lru_cache

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    rewritten_query = Column(Text, nullable=True)
    intent = Column(
        String(32), nullable=True
    )  # navigational/informational/transactional
    ranker_version = Column(String(32), nullable=True)  # lambdarank / crossencoder
    ab_variant = Column(String(32), nullable=True)
    num_results = Column(Integer, nullable=True)
    total_latency_ms = Column(Float, nullable=True)
    retrieval_latency_ms = Column(Float, nullable=True)
    ranking_latency_ms = Column(Float, nullable=True)
    cache_hit = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_query_logs_created_at", "created_at"),)


class ClickLog(Base):
    __tablename__ = "click_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    doc_id = Column(Integer, nullable=False)  # passage pid
    rank_shown = Column(Integer, nullable=False)  # 1-indexed rank position
    ranker_version = Column(String(32), nullable=True)
    clicked = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_click_logs_created_at", "created_at"),
        Index("ix_click_logs_request_id", "request_id"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(
        String(64), nullable=False
    )  # lambdarank / crossencoder / two_tower
    version = Column(String(64), nullable=False)
    mlflow_run_id = Column(String(128), nullable=True)
    ndcg_at_10 = Column(Float, nullable=True)
    stage = Column(String(32), nullable=False)  # staging / production / archived
    promoted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def _build_dsn() -> str:
    """Build the SQLAlchemy DSN.

    DATABASE_URL takes precedence when set — this is what managed/serverless
    Postgres providers (Neon, Render, Railway) hand you, and it carries sslmode.
    Otherwise fall back to discrete POSTGRES_* parts for local docker-compose.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        # Normalize the scheme so SQLAlchemy uses the psycopg2 driver.
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "search_ranking")
    user = os.getenv("POSTGRES_USER", "searchuser")
    password = os.getenv("POSTGRES_PASSWORD", "searchpass")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


@lru_cache(maxsize=1)
def get_engine():
    """Process-wide pooled engine (created once, reused across requests).

    Previously a fresh engine + connection was built per request (NullPool),
    which churned connections under load. A pooled engine with pre-ping +
    recycle is also what serverless Postgres (Neon) needs, since it drops idle
    connections that a naive pool would hand out stale.
    """
    return create_engine(
        _build_dsn(),
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        pool_pre_ping=True,
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine())


def create_tables(engine=None):
    Base.metadata.create_all(engine or get_engine())


def get_db_session() -> Session:
    return get_session_factory()()
