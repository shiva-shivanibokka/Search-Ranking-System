# 0002 — Hybrid retrieval (BM25 + dense) fused with Reciprocal Rank Fusion

**Status:** Accepted

## Context
Dense (two-tower) retrieval captures semantic similarity but misses exact-term
matches (rare entities, IDs, acronyms). BM25 captures lexical overlap but misses
synonymy ("car" vs "automobile"). Each alone leaves recall on the table.

## Decision
Run both retrievers and fuse their ranked lists with **Reciprocal Rank Fusion**:
`score(d) = Σ_i 1 / (k + rank_i(d))`, with `k = 60`. RRF uses ranks, not raw
scores, so it needs no score normalization between the two very different scoring
systems, and documents both retrievers agree on get an additive boost.

## Consequences
- **Pro:** higher Recall@100 than either retriever alone, which directly lifts
  the ceiling for the reranker (see ADR-0001).
- **Pro:** RRF is parameter-light and robust — no per-system score calibration.
- **Trade-off:** runs two retrievers per query (more compute). Mitigated by
  Redis caching and the fact that BM25 over 500K docs is cheap.
- **Trade-off:** RRF ignores score magnitude, so a hugely confident single-system
  hit isn't weighted extra. Acceptable for this corpus; revisit with weighted RRF
  if one retriever proves consistently stronger.

## At 10× scale
BM25's full-corpus scoring becomes the bottleneck; move it to a real inverted
index (Elasticsearch/OpenSearch) and keep RRF as the fusion layer.
