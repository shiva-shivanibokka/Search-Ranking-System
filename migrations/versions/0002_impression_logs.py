"""impression_logs table

Revision ID: 0002_impression_logs
Revises: 0001_initial
Create Date: 2026-07-03

Adds `impression_logs`, recording every result shown to a user (not just
clicked ones). Mirrors the SQLAlchemy `ImpressionLog` model in
services.shared.database.

`COLUMNS` below lists the column names in declaration order and is asserted
against `ImpressionLog.__table__.columns.keys()` in
tests/unit/test_impressions.py::test_migration_matches_orm to keep this
migration and the ORM model in sync.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_impression_logs"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

COLUMNS = [
    "id",
    "request_id",
    "query_text",
    "doc_id",
    "rank_shown",
    "ranker_version",
    "created_at",
]


def upgrade() -> None:
    op.create_table(
        "impression_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.Integer(), nullable=False),
        sa.Column("rank_shown", sa.Integer(), nullable=False),
        sa.Column("ranker_version", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_impression_logs_request_id", "impression_logs", ["request_id"]
    )
    op.create_index(
        "ix_impression_logs_created_at", "impression_logs", ["created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_impression_logs_created_at", table_name="impression_logs")
    op.drop_index("ix_impression_logs_request_id", table_name="impression_logs")
    op.drop_table("impression_logs")
