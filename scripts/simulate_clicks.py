"""Click simulator: MS MARCO replay stream, ORCAS-weighted -> retrieve -> qrels
relevance -> position-based click model (PBM) -> impression/click logs.

Design (see docs/superpowers/plans/2026-07-03-orcas-real-feedback-loop.md):
  The REPLAY QUERY STREAM is MS MARCO queries that HAVE qrels (train, falling
  back to dev), so every replayed query is guaranteed to carry a real
  relevance label -- there is no coverage/starvation failure mode. ORCAS only
  WEIGHTS which queries get sampled (queries whose normalized text matches an
  ORCAS query are drawn proportionally to ORCAS popularity; everything else
  replays at a baseline weight) and calibrates click volume/propensity via
  scripts/calibrate_orcas.py. Relevance ALWAYS comes from qrels, never ORCAS.

Propensity key type (int vs. str) -- read this before wiring calibration in:
  `scripts.calibrate_orcas.propensity_curve()` returns a dict keyed by INT
  rank. `scripts.calibrate_orcas.calibrate()` re-keys that same dict to STRING
  ranks ("1".."10") before JSON-serializing it, so the calibration JSON on
  disk always has string keys. `sample_clicks()` below requires an INT-keyed
  dict (it looks the rank up directly as `propensity.get(rank, fallback)` and
  `rank` is always an int). Whenever a propensity dict is loaded from the
  calibration JSON (as `simulate()` and `main()` do), it MUST be converted
  back to int keys first: `{int(k): v for k, v in calibration["propensity"].items()}`.
"""

from __future__ import annotations

import random
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.shared.database import ClickLog  # noqa: E402
from services.shared.impressions import build_impression_rows, insert_impressions  # noqa: E402

# Retrieval-only replay: the simulator shows the fused retrieval set (no
# LambdaRank/CrossEncoder reranking applied), so logged rows are tagged with
# this instead of a real ranker version.
RANKER_VERSION = "orcas_replay_retrieval_only"


def sample_clicks(
    shown: list[dict],
    gold_pids: set[int],
    propensity: dict[int, float],
    rng: random.Random,
    irrelevant_ctr: float = 0.02,
) -> list[tuple[int, int, bool]]:
    """Sample clicks for a shown result list via a position-based click model.

    ``clicked ~ Bernoulli(propensity[rank] * (1.0 if doc_id in gold_pids else irrelevant_ctr))``

    Args:
        shown: list of `{"doc_id": int, "rank": int}` (1-indexed rank).
        gold_pids: passage ids judged relevant for this query (from qrels).
        propensity: INT-keyed rank -> propensity (see module docstring for the
            int-vs-str key note; convert calibration JSON's string keys before
            calling this).
        rng: caller-owned `random.Random` so trials are reproducible.
        irrelevant_ctr: click-through rate applied to non-relevant passages
            (models noise clicks; never the dominant signal).

    Returns:
        One `(doc_id, rank_shown, clicked)` tuple per shown passage.

    Ranks beyond the propensity curve's max rank (or any rank missing from
    the dict) fall back to the smallest known propensity value, since position
    bias only gets weaker further down the page. An empty propensity dict
    falls back to 0.0 (no clicks at all, rather than raising).
    """
    fallback = min(propensity.values()) if propensity else 0.0
    results: list[tuple[int, int, bool]] = []
    for result in shown:
        doc_id = result["doc_id"]
        rank = result["rank"]
        p_rank = propensity.get(rank, fallback)
        relevance_ctr = 1.0 if doc_id in gold_pids else irrelevant_ctr
        clicked = rng.random() < (p_rank * relevance_ctr)
        results.append((doc_id, rank, clicked))
    return results


def load_replay_workload(queries_df, qrels_df) -> tuple[dict[str, str], dict[str, set[int]]]:
    """Build the replay workload: MS MARCO queries that have qrels.

    Mirrors `training/evaluate.py`'s loading (`qid_to_gold =
    qrels_df.groupby("qid")["pid"].apply(set).to_dict()`, `qid_to_text =
    dict(zip(queries_df["qid"], queries_df["text"]))`). Only qids present in
    qrels are kept in *both* returned dicts, so every workload qid is
    guaranteed a non-empty gold set -- the simulator can never replay a query
    with no relevance label.
    """
    qid_to_gold = qrels_df.groupby("qid")["pid"].apply(set).to_dict()
    qid_to_text_all = dict(zip(queries_df["qid"], queries_df["text"]))
    qid_to_text = {qid: text for qid, text in qid_to_text_all.items() if qid in qid_to_gold}
    return qid_to_text, qid_to_gold


def build_sampling_weights(
    qid_to_text: dict[str, str],
    query_popularity: dict[str, int],
    baseline: float = 1.0,
) -> tuple[dict[str, float], int]:
    """Weight the replay stream by ORCAS query popularity.

    Per-qid weight = `float(query_popularity[norm(text)])` when the
    normalized query text (`strip().lower()`, matching
    `scripts.calibrate_orcas.query_popularity`'s normalization) matches an
    ORCAS query; otherwise `baseline`. Weights, never relevance -- an
    unmatched query still replays (at the baseline rate), it just isn't
    boosted by ORCAS frequency.

    Returns `(weights, matched_count)` where `matched_count` is how many qids
    matched an ORCAS query.
    """
    weights: dict[str, float] = {}
    matched = 0
    for qid, text in qid_to_text.items():
        normalized = text.strip().lower()
        if normalized in query_popularity:
            weights[qid] = float(query_popularity[normalized])
            matched += 1
        else:
            weights[qid] = baseline
    return weights, matched


def simulate(
    engine,
    qid_to_text: dict[str, str],
    qid_to_gold: dict[str, set[int]],
    weights: dict[str, float],
    calibration: dict,
    cfg,
    session,
    seed: int,
) -> dict:
    """Replay `cfg.queries` ORCAS-weighted queries through `engine.retrieve`.

    For each replayed qid: retrieve `cfg.top_k` candidates, log the full shown
    set as impressions, sample clicks via the calibrated PBM (relevance from
    `qid_to_gold`, never ORCAS), and write a `ClickLog` row for each clicked
    passage. `negatives = impressions - clicks` (shown-but-not-clicked rows
    are real negatives, recoverable via the impression/click join).

    `calibration` is the JSON dict from `scripts.calibrate_orcas.calibrate()`:
    its `propensity` (string-keyed) is converted to int keys here before being
    handed to `sample_clicks`; its `query_popularity` is used only to report
    how many *replayed* queries matched an ORCAS query (weighting was already
    decided by the caller via `weights`).
    """
    rng = random.Random(seed)
    propensity = {int(rank): value for rank, value in calibration.get("propensity", {}).items()}
    query_popularity = calibration.get("query_popularity", {})

    qids = list(weights.keys())
    qid_weights = [weights[qid] for qid in qids]

    impressions = 0
    clicks = 0
    queries_replayed = 0
    queries_matched_to_orcas = 0

    for _ in range(cfg.queries):
        (qid,) = rng.choices(qids, weights=qid_weights, k=1)
        query_text = qid_to_text[qid]
        gold_pids = qid_to_gold.get(qid, set())

        if query_text.strip().lower() in query_popularity:
            queries_matched_to_orcas += 1

        candidates = engine.retrieve(query_text, query_text, cfg.top_k)
        shown = [
            {"doc_id": candidate["doc_id"], "rank": rank}
            for rank, candidate in enumerate(candidates, start=1)
        ]

        request_id = uuid.uuid4().hex
        impression_rows = build_impression_rows(request_id, query_text, RANKER_VERSION, shown)
        impressions += insert_impressions(session, impression_rows)

        click_rows = sample_clicks(shown, gold_pids, propensity, rng, cfg.irrelevant_ctr)
        for doc_id, rank_shown, clicked in click_rows:
            if not clicked:
                continue
            session.add(
                ClickLog(
                    request_id=request_id,
                    query_text=query_text,
                    doc_id=doc_id,
                    rank_shown=rank_shown,
                    ranker_version=RANKER_VERSION,
                    clicked=True,
                )
            )
            clicks += 1
        session.commit()
        queries_replayed += 1

    return {
        "impressions": impressions,
        "clicks": clicks,
        "negatives": impressions - clicks,
        "queries_replayed": queries_replayed,
        "queries_matched_to_orcas": queries_matched_to_orcas,
    }


@dataclass
class ClickSimConfig:
    top_k: int = 10
    irrelevant_ctr: float = 0.02
    queries: int = 5000
    seed: int = 42
    calibration_path: str = "data/processed/orcas_calibration.json"
    workload: str = "train"


def _load_click_sim_config(config_path: str = "configs/config.yaml") -> ClickSimConfig:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    section = raw.get("click_sim", {})
    valid = {"top_k", "irrelevant_ctr", "queries", "seed", "calibration_path", "workload"}
    return ClickSimConfig(**{k: v for k, v in section.items() if k in valid})


def _load_workload_frames(workload: str) -> tuple:
    """Load (queries_df, qrels_df) for `workload` ("train"), falling back to
    dev if the train parquet files aren't present."""
    import pandas as pd

    processed = Path("data/processed")
    train_queries, train_qrels = processed / "train_queries.parquet", processed / "train_qrels.parquet"
    dev_queries, dev_qrels = processed / "dev_queries.parquet", processed / "dev_qrels.parquet"

    if workload == "train" and train_queries.exists() and train_qrels.exists():
        return pd.read_parquet(train_queries), pd.read_parquet(train_qrels)
    return pd.read_parquet(dev_queries), pd.read_parquet(dev_qrels)


def main() -> int:
    """Thin wiring: real config + SearchEngine + DB session + calibration ->
    simulate(). Best-effort; deliberately not unit-tested end-to-end (see
    tests/unit/test_click_sim.py for the unit-tested core helpers)."""
    import json

    from deploy.engine import SearchEngine
    from services.shared.database import get_db_session

    cfg = _load_click_sim_config()

    with open(cfg.calibration_path, "r", encoding="utf-8") as f:
        calibration = json.load(f)
    query_popularity = calibration.get("query_popularity", {})

    queries_df, qrels_df = _load_workload_frames(cfg.workload)
    qid_to_text, qid_to_gold = load_replay_workload(queries_df, qrels_df)
    weights, matched = build_sampling_weights(qid_to_text, query_popularity)
    print(
        f"Replay workload: {len(qid_to_text)} qids "
        f"({matched} matched an ORCAS query for popularity weighting)."
    )

    engine = SearchEngine()
    session = get_db_session()
    try:
        result = simulate(engine, qid_to_text, qid_to_gold, weights, calibration, cfg, session, cfg.seed)
    finally:
        session.close()

    print(f"Simulated {result['queries_replayed']} replayed queries.")
    print(
        f"impressions={result['impressions']} clicks={result['clicks']} "
        f"negatives={result['negatives']} "
        f"queries_matched_to_orcas={result['queries_matched_to_orcas']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
