# Architecture Decision Records (ADRs)

Short, durable records of the significant architecture/design decisions in this
project — what was decided, why, what was traded off, and what would change at
larger scale. ADRs make the reasoning behind the system legible without reading
all the code.

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-two-stage-retrieve-then-rank.md) | Two-stage retrieve-then-rank pipeline | Accepted |
| [0002](0002-hybrid-retrieval-rrf.md) | Hybrid retrieval (BM25 + dense) fused with RRF | Accepted |
| [0003](0003-provider-agnostic-llm.md) | Provider-agnostic LLM layer with a zero-key default | Accepted |
| [0004](0004-free-tier-deployment-topology.md) | Dual-mode deployment: microservices local, consolidated on free infra | Accepted (partially superseded by 0007) |
| [0005](0005-postgres-on-neon.md) | Serverless Postgres (Neon), not Supabase | Accepted |
| [0006](0006-orcas-calibrated-feedback.md) | ORCAS-calibrated click simulation, not raw human clicks | Accepted |
| [0007](0007-cloud-run-vercel-frontend.md) | Cloud Run API + SvelteKit/Vercel frontend with client-side BYOK RAG | Accepted |

Format: each ADR states **Context → Decision → Consequences (trade-offs) → At 10× scale**.
