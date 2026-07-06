"""
Quick tokenizer speed check — run this FIRST in any new environment.

The anaconda-base env tokenizes at ~1,200 sequences/sec (≈100x too slow),
which is what forced SP-1's first run down to a 20K-query subset. A healthy
environment should hit tens of thousands of sequences/sec. Use this to confirm
a clean venv actually fixed the problem before launching the full retrain.

    python scripts/tokenizer_speed_test.py
"""

import time


def main() -> None:
    import os

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print("fast tokenizer:", tok.is_fast)

    n = 50_000
    texts = [
        "the primary causes of inflation include money supply growth and demand"
    ] * n

    t0 = time.time()
    tok(texts, max_length=180, padding="max_length", truncation=True, return_tensors="pt")
    dt = time.time() - t0
    rate = n / dt
    print(f"tokenized {n:,} sequences in {dt:.1f}s  ->  {rate:,.0f} seq/sec")
    if rate < 5_000:
        print(
            "SLOW (<5k/s): this env has the tokenizer problem — do NOT run the full "
            "retrain here. Use a clean venv (see docs/SP1-FULL-RERUN.md)."
        )
    else:
        print("HEALTHY (>=5k/s): safe to run the full-scale retrain in this env.")


if __name__ == "__main__":
    main()
