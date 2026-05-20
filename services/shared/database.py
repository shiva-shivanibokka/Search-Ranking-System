"""
PostgreSQL connection and schema definitions.

Tables:
  query_logs    : every search request (for analytics and retraining)
  click_logs    : user clicks on results (implicit relevance signal)
  model_versions: model promotion history (audit trail)
"""

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Text,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import NullPool

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


def get_engine():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "search_ranking")
    user = os.getenv("POSTGRES_USER", "searchuser")
    password = os.getenv("POSTGRES_PASSWORD", "searchpass")
    dsn = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(dsn, poolclass=NullPool)


def get_session_factory(engine=None) -> sessionmaker:
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine)


def create_tables(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)


def get_db_session() -> Session:
    engine = get_engine()
    SessionFactory = sessionmaker(bind=engine)
    return SessionFactory()
