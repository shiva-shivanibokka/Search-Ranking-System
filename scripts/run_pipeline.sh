#!/bin/bash
# Full training pipeline — run steps in order after activating your venv.
# Usage: bash scripts/run_pipeline.sh

set -e

echo "=== Step 1: Download MS MARCO ==="
python scripts/download_msmarco.py

echo "=== Step 2: Preprocess + BM25 index + Hard negatives ==="
python scripts/preprocess.py --max-passages 500000 --max-triples 500000

echo "=== Step 3: Train Two-Tower ==="
python training/train_two_tower.py

echo "=== Step 4: Build FAISS Index ==="
python training/build_faiss_index.py

echo "=== Step 5: Train CrossEncoder ==="
python training/train_cross_encoder.py

echo "=== Step 6: Train LambdaRank ==="
python training/train_lambdarank.py

echo "=== Step 7: Full Offline Evaluation (includes Hybrid+RRF configs) ==="
python training/evaluate.py

echo "=== Pipeline complete. Start services with: docker-compose up ==="
