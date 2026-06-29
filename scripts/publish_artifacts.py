"""
Publish serving artifacts to Hugging Face Hub (run once, or after retraining).

This is the upload counterpart to ``scripts/bootstrap.py``. It pushes the model
weights / FAISS index / BM25 index / passages that are too large for git to a
free Hugging Face model repo, so that any fresh clone or deploy can pull them.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login          # or set HF_TOKEN

Usage:
    HF_ARTIFACTS_REPO=your-username/search-ranking-artifacts \
        python scripts/publish_artifacts.py

    # Include optional artifacts too:
    python scripts/publish_artifacts.py --optional

Only files that exist locally are uploaded; missing optional artifacts are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.artifacts_manifest import (  # noqa: E402
    HF_ARTIFACTS_REPO,
    OPTIONAL_ARTIFACTS,
    SERVING_ARTIFACTS,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload serving artifacts to HF Hub.")
    parser.add_argument(
        "--optional",
        action="store_true",
        help="Also upload optional artifacts (cross-encoder, difficulty classifier).",
    )
    args = parser.parse_args()

    repo_id = HF_ARTIFACTS_REPO
    if repo_id.startswith("REPLACE_ME"):
        print(
            "ERROR: set HF_ARTIFACTS_REPO=<your-username>/search-ranking-artifacts first.",
            file=sys.stderr,
        )
        return 2

    token = os.getenv("HF_TOKEN")
    repo_type = os.getenv("HF_ARTIFACTS_REPO_TYPE", "model")

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(repo_id, repo_type=repo_type, exist_ok=True, token=token)
    print(f"Target repo: {repo_id} ({repo_type})")

    targets = list(SERVING_ARTIFACTS)
    if args.optional:
        targets += OPTIONAL_ARTIFACTS

    uploaded, skipped = 0, 0
    for rel_path in targets:
        local = PROJECT_ROOT / rel_path
        if not local.exists():
            print(f"  [skip — not found locally] {rel_path}")
            skipped += 1
            continue
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=rel_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"  [uploaded] {rel_path}")
        uploaded += 1

    print(f"\nDone. Uploaded {uploaded}, skipped {skipped}.")
    print(f"Consumers should set: export HF_ARTIFACTS_REPO={repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
