"""Alembic environment.

Resolves the database URL from DATABASE_URL / POSTGRES_* via the same helper the
services use, and targets the SQLAlchemy models in services.shared.database so
`alembic revision --autogenerate` stays in sync with the ORM.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the project importable when alembic runs from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.shared.database import Base, _build_dsn  # noqa: E402

config = context.config
config.set_main_option("sqlalchemy.url", _build_dsn())

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=_build_dsn(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
