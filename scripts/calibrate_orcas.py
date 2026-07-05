"""Calibrate position-bias propensity and query popularity from ORCAS.

This module produces:
  - query_popularity: normalized-query-text -> frequency in ORCAS
  - mean_clicks_per_query: average distinct documents clicked per query
  - propensity_curve: position-bias monotonic curve (literature-based eta)
  - calibrate: produces the full calibration JSON for the simulator

Note: ORCAS has no rank/position column, so eta (position-bias exponent) is a
literature-based assumption, not data-driven from ORCAS itself.
"""

from __future__ import annotations

import json
from pathlib import Path


def propensity_curve(max_rank: int, eta: float) -> dict[int, float]:
    """Compute position-bias propensity curve.

    Args:
        max_rank: Maximum rank (e.g., 10).
        eta: Position-bias exponent. For eta=1.0, propensity[rank] = 1/rank.

    Returns:
        Dictionary {rank: (1/rank)**eta} for rank in 1..max_rank.
        Strictly decreasing for eta > 0.
    """
    return {rank: (1.0 / rank) ** eta for rank in range(1, max_rank + 1)}


def query_popularity(rows: list[dict]) -> dict[str, int]:
    """Count query frequency by normalized text.

    Normalization: strip and lowercase.

    Args:
        rows: List of dicts with 'query' key (from Task 4's parse_orcas_line).

    Returns:
        Dictionary {normalized_query_text: count}.
    """
    popularity = {}
    for row in rows:
        normalized = row["query"].strip().lower()
        popularity[normalized] = popularity.get(normalized, 0) + 1
    return popularity


def mean_clicks_per_query(rows: list[dict]) -> float:
    """Compute mean distinct documents clicked per query.

    Args:
        rows: List of dicts with 'qid' and 'did' keys.

    Returns:
        Mean number of distinct documents per unique query ID.
    """
    if not rows:
        return 0.0

    # Group by qid and count distinct dids per qid
    qid_to_dids = {}
    for row in rows:
        qid = row["qid"]
        did = row["did"]
        if qid not in qid_to_dids:
            qid_to_dids[qid] = set()
        qid_to_dids[qid].add(did)

    # Compute mean
    total_dids = sum(len(dids) for dids in qid_to_dids.values())
    return total_dids / len(qid_to_dids)


def calibrate(
    rows: list[dict], max_rank: int = 10, eta: float = 1.0
) -> dict:
    """Calibrate propensity and popularity from ORCAS rows.

    Args:
        rows: List of dicts from Task 4's parse_orcas_line.
        max_rank: Maximum rank for propensity curve (default 10).
        eta: Position-bias exponent (default 1.0, literature assumption).

    Returns:
        JSON-serializable dict with keys:
          - eta: Position-bias exponent (literature assumption).
          - propensity: {str(rank): propensity_value} for JSON serialization.
          - mean_clicks_per_query: Float mean clicks per query.
          - query_popularity: {normalized_text: count}.
          - source: "ORCAS".
          - notes: Explanation of what is calibrated vs. assumed.

        Note: propensity keys are strings in the output dict (for JSON
        serialization) but are computed as ints internally via propensity_curve.
    """
    # Compute components
    propensity_int = propensity_curve(max_rank, eta)
    # Convert propensity keys to strings for JSON serialization
    propensity_str = {str(k): v for k, v in propensity_int.items()}
    popularity = query_popularity(rows)
    mean_clicks = mean_clicks_per_query(rows)

    notes = (
        "eta (position-bias exponent) is a literature-based assumption; "
        "ORCAS has no rank/position column. "
        "query_popularity and mean_clicks_per_query are calibrated from ORCAS data."
    )

    return {
        "eta": eta,
        "propensity": propensity_str,
        "mean_clicks_per_query": mean_clicks,
        "query_popularity": popularity,
        "source": "ORCAS",
        "notes": notes,
    }


def main():
    """Load ORCAS calibration data and write calibration JSON.

    Reads from data/raw/orcas_sample.tsv (output of Task 4: download_orcas.py)
    and writes calibration to data/processed/orcas_calibration.json.
    """
    from scripts.download_orcas import parse_orcas_line

    raw_path = Path("data/raw/orcas_sample.tsv")
    out_path = Path("data/processed/orcas_calibration.json")

    if not raw_path.exists():
        print(f"Error: {raw_path} not found. Run Task 4 first (download_orcas.py).")
        return 1

    # Read and parse ORCAS sample
    rows = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            row = parse_orcas_line(line)
            if row is not None:
                rows.append(row)

    if not rows:
        print(f"Error: No valid rows found in {raw_path}")
        return 1

    # Calibrate
    calibration = calibrate(rows, max_rank=10, eta=1.0)

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"Wrote calibration for {len(rows):,} rows -> {out_path}")
    print(f"  Query popularity: {len(calibration['query_popularity'])} unique queries")
    print(f"  Mean clicks/query: {calibration['mean_clicks_per_query']:.2f}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
