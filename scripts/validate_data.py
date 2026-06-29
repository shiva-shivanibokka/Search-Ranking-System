"""
Data quality checks for the processed dataset.

Runs after preprocessing (or in CI/scheduled jobs) to catch silent data
corruption before it reaches training or serving — empty passages, duplicate
ids, null keys, implausible row counts. Exits non-zero on any failure so it can
gate a pipeline.

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --data-dir data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Check:
    """Collects pass/fail results so we can report all problems at once."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passed = 0

    def expect(self, condition: bool, message: str) -> None:
        if condition:
            self.passed += 1
        else:
            self.failures.append(message)


def validate_passages(df: pd.DataFrame, c: Check) -> None:
    c.expect({"pid", "text", "token_count"}.issubset(df.columns),
             "passages: missing required columns {pid, text, token_count}")
    c.expect(df["pid"].notna().all(), "passages: null pid values found")
    c.expect(df["pid"].is_unique, "passages: duplicate pid values found")
    c.expect(df["text"].notna().all(), "passages: null text values found")
    empty = (df["text"].fillna("").str.strip() == "").sum()
    c.expect(empty == 0, f"passages: {empty} empty passage texts")
    c.expect((df["token_count"] > 0).all(), "passages: non-positive token_count values")
    c.expect(len(df) > 1000, f"passages: suspiciously few rows ({len(df)})")


def validate_qrels(df: pd.DataFrame, name: str, c: Check) -> None:
    c.expect({"qid", "pid"}.issubset(df.columns), f"{name}: missing required columns {{qid, pid}}")
    c.expect(df["qid"].notna().all(), f"{name}: null qid values found")
    c.expect(df["pid"].notna().all(), f"{name}: null pid values found")
    c.expect(len(df) > 0, f"{name}: empty qrels")


def validate_queries(df: pd.DataFrame, name: str, c: Check) -> None:
    c.expect({"qid", "text"}.issubset(df.columns), f"{name}: missing required columns {{qid, text}}")
    c.expect(df["qid"].is_unique, f"{name}: duplicate qid values found")
    empty = (df["text"].fillna("").str.strip() == "").sum()
    c.expect(empty == 0, f"{name}: {empty} empty query texts")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate processed data quality.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    c = Check()
    checked_files = 0

    files = {
        "passages.parquet": validate_passages,
        "dev_qrels.parquet": lambda df, ck: validate_qrels(df, "dev_qrels", ck),
        "train_qrels.parquet": lambda df, ck: validate_qrels(df, "train_qrels", ck),
        "dev_queries.parquet": lambda df, ck: validate_queries(df, "dev_queries", ck),
        "train_queries.parquet": lambda df, ck: validate_queries(df, "train_queries", ck),
    }

    for fname, validator in files.items():
        path = data_dir / fname
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            # An unreadable parquet is itself a data-quality failure (corruption
            # or a writer/reader version mismatch) — report it, don't crash.
            c.expect(False, f"{fname}: failed to read parquet ({type(e).__name__}: {e})")
            checked_files += 1
            print(f"  [unreadable] {fname}")
            continue
        validator(df, c)
        checked_files += 1
        print(f"  [checked] {fname}")

    if checked_files == 0:
        print("No processed data files found — run scripts/preprocess.py first.", file=sys.stderr)
        return 2

    print(f"\n{c.passed} checks passed, {len(c.failures)} failed.")
    if c.failures:
        for f in c.failures:
            print(f"  FAIL: {f}", file=sys.stderr)
        return 1
    print("All data quality checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
