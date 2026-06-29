# 0005 — Serverless Postgres (Neon), not Supabase

**Status:** Accepted

## Context
The system needs a relational store for `query_logs`, `click_logs`, and
`model_versions` — the click data is the implicit relevance signal that drives
retraining, and the schema is genuinely relational. For the free hosted demo we
need a managed Postgres that costs nothing and works from an ephemeral Space.

## Decision
Use **Neon** (free serverless Postgres). The app reads a standard `DATABASE_URL`
(with `sslmode=require`), so it is provider-portable; the DB engine uses
`pool_pre_ping` + `pool_recycle`, which is exactly what serverless Postgres needs
since it drops idle connections. Schema is managed by **Alembic** migrations, not
ad-hoc `create_all()`.

## Consequences
- **Pro:** $0, fully managed, real Postgres (not a bespoke API) — portable to
  Render/Railway/RDS by changing one URL.
- **Pro:** serverless scale-to-zero suits bursty demo traffic.
- **Trade-off:** cold starts add latency to the first query after idle. Click
  logging is a background task, so it never blocks the user-facing response.
- **Why not Supabase:** explicitly avoided per project constraint. Supabase bundles
  auth/realtime/storage we don't need; plain Postgres keeps the data layer simple
  and vendor-neutral. Neon delivers the one thing we actually need (managed Postgres)
  with less surface area.

## At 10× scale
Move to a provisioned Postgres with read replicas for analytics queries over
`query_logs`/`click_logs`, and partition the log tables by time.
