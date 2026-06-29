# 0001 — Two-stage retrieve-then-rank pipeline

**Status:** Accepted

## Context
We rank relevant passages from a 500K-document corpus under a ~200ms latency
budget. Scoring every document with an accurate model per query is infeasible:
a cross-encoder at ~1ms/pair would need ~500s per query.

## Decision
Use the industry-standard two-stage pipeline:
1. **Retrieve** — a fast approximate stage returns ~100 candidates (FAISS ANN +
   BM25). Optimizes recall and speed.
2. **Rank** — a slower, more accurate reranker (LambdaRank or CrossEncoder)
   scores only those 100 and returns the top 10. Optimizes precision at the top.

## Consequences
- **Pro:** keeps p95 latency in budget while still using an expensive reranker.
- **Pro:** stages evolve independently (swap the reranker without touching retrieval).
- **Trade-off:** recall is capped by stage 1 — if a relevant doc isn't in the top
  100, the reranker can never surface it. We mitigate with hybrid retrieval
  (ADR-0002) to raise Recall@100.

## At 10× scale
The retrieval stage is the scaling pressure, not ranking. At 5M+ docs, shard the
FAISS index and move it behind a dedicated vector service; the reranker scales
horizontally since it only ever sees 100 candidates per query.
