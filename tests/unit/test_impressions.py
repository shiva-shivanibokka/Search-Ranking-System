"""Unit tests for the ImpressionLog table (ORM + migration parity)."""

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def test_impressionlog_table_roundtrip():
    from services.shared.database import Base, ImpressionLog

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    row = ImpressionLog(
        request_id="req-123",
        query_text="what is machine learning",
        doc_id=42,
        rank_shown=3,
        ranker_version="lambdarank",
        created_at=datetime.utcnow(),
    )
    session.add(row)
    session.commit()

    fetched = session.query(ImpressionLog).one()
    assert fetched.request_id == "req-123"
    assert fetched.query_text == "what is machine learning"
    assert fetched.doc_id == 42
    assert fetched.rank_shown == 3
    assert fetched.ranker_version == "lambdarank"
    assert fetched.created_at is not None


def test_clicklog_has_single_request_id_index():
    from services.shared.database import ClickLog

    request_id_indexes = [
        idx for idx in ClickLog.__table__.indexes if "request_id" in idx.columns
    ]
    assert len(request_id_indexes) == 1
    assert request_id_indexes[0].name == "ix_click_logs_request_id"


def test_migration_matches_orm():
    import importlib

    from services.shared.database import ImpressionLog

    migration = importlib.import_module(
        "migrations.versions.0002_impression_logs"
    )

    assert migration.COLUMNS == list(ImpressionLog.__table__.columns.keys())
