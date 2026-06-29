"""initial schema: query_logs, click_logs, model_versions

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-28

Mirrors the SQLAlchemy models in services.shared.database. This is the baseline
migration; future schema changes should be added as new revisions via
`alembic revision --autogenerate -m "..."`.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "query_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("rewritten_query", sa.Text(), nullable=True),
        sa.Column("intent", sa.String(length=32), nullable=True),
        sa.Column("ranker_version", sa.String(length=32), nullable=True),
        sa.Column("ab_variant", sa.String(length=32), nullable=True),
        sa.Column("num_results", sa.Integer(), nullable=True),
        sa.Column("total_latency_ms", sa.Float(), nullable=True),
        sa.Column("retrieval_latency_ms", sa.Float(), nullable=True),
        sa.Column("ranking_latency_ms", sa.Float(), nullable=True),
        sa.Column("cache_hit", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_query_logs_request_id", "query_logs", ["request_id"])
    op.create_index("ix_query_logs_created_at", "query_logs", ["created_at"])

    op.create_table(
        "click_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.Integer(), nullable=False),
        sa.Column("rank_shown", sa.Integer(), nullable=False),
        sa.Column("ranker_version", sa.String(length=32), nullable=True),
        sa.Column("clicked", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_click_logs_request_id", "click_logs", ["request_id"])
    op.create_index("ix_click_logs_created_at", "click_logs", ["created_at"])

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_type", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=128), nullable=True),
        sa.Column("ndcg_at_10", sa.Float(), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("promoted_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("model_versions")
    op.drop_index("ix_click_logs_created_at", table_name="click_logs")
    op.drop_index("ix_click_logs_request_id", table_name="click_logs")
    op.drop_table("click_logs")
    op.drop_index("ix_query_logs_created_at", table_name="query_logs")
    op.drop_index("ix_query_logs_request_id", table_name="query_logs")
    op.drop_table("query_logs")
