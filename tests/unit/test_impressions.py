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


# ── build_impression_rows / insert_impressions ───────────────────────────────


def test_build_impression_rows():
    from services.shared.impressions import build_impression_rows

    results = [
        {"rank": 1, "doc_id": 10, "text": "a", "score": 0.9, "ranker": "lambdarank"},
        {"rank": 2, "doc_id": 20, "text": "b", "score": 0.8, "ranker": "lambdarank"},
    ]

    rows = build_impression_rows("req1", "q", "lambdarank", results)

    assert len(rows) == 2
    assert rows[0] == {
        "request_id": "req1",
        "query_text": "q",
        "doc_id": 10,
        "rank_shown": 1,
        "ranker_version": "lambdarank",
    }
    assert rows[1] == {
        "request_id": "req1",
        "query_text": "q",
        "doc_id": 20,
        "rank_shown": 2,
        "ranker_version": "lambdarank",
    }


def test_insert_impressions_writes_rows():
    from services.shared.database import Base, ImpressionLog
    from services.shared.impressions import insert_impressions

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    rows = [
        {
            "request_id": "req1",
            "query_text": "q",
            "doc_id": 10,
            "rank_shown": 1,
            "ranker_version": "lambdarank",
        },
        {
            "request_id": "req1",
            "query_text": "q",
            "doc_id": 20,
            "rank_shown": 2,
            "ranker_version": "lambdarank",
        },
    ]

    count = insert_impressions(session, rows)

    assert count == 2
    fetched = session.query(ImpressionLog).order_by(ImpressionLog.rank_shown).all()
    assert len(fetched) == 2
    assert fetched[0].rank_shown == 1
    assert fetched[1].rank_shown == 2


def test_gateway_logs_impressions(monkeypatch):
    import services.gateway.main as gw
    from services.shared.database import Base, ImpressionLog

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    monkeypatch.setattr(gw, "get_db_session", lambda: session)

    payload = dict(
        request_id="req1",
        query_text="q",
        ranker_version="lambdarank",
        results=[
            {"rank": 1, "doc_id": 10, "text": "a", "score": 0.9, "ranker": "lambdarank"},
            {"rank": 2, "doc_id": 20, "text": "b", "score": 0.8, "ranker": "lambdarank"},
        ],
    )

    gw._sync_log_impressions_to_db(payload)

    fetched = session.query(ImpressionLog).order_by(ImpressionLog.rank_shown).all()
    assert len(fetched) == 2
    assert fetched[0].doc_id == 10
    assert fetched[0].rank_shown == 1
    assert fetched[1].doc_id == 20
    assert fetched[1].rank_shown == 2
