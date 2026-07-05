"""Impression logging — records every result shown to a user (not just clicks).

Combined with ClickLog, this gives real negatives (shown-but-not-clicked) for
retraining, instead of the sampled/random negatives used previously.
"""

from services.shared.database import ImpressionLog


def build_impression_rows(
    request_id: str,
    query_text: str,
    ranker_version: str | None,
    results: list[dict],
) -> list[dict]:
    """Build one impression row per shown result.

    `rank_shown` comes from `result["rank"]` and `doc_id` from `result["doc_id"]`.
    """
    return [
        {
            "request_id": request_id,
            "query_text": query_text,
            "doc_id": result["doc_id"],
            "rank_shown": result["rank"],
            "ranker_version": ranker_version,
        }
        for result in results
    ]


def insert_impressions(session, rows: list[dict]) -> int:
    """Bulk-insert impression rows and return the number of rows written."""
    for row in rows:
        session.add(ImpressionLog(**row))
    session.commit()
    return len(rows)
