"""
Bootstrap a fresh clone / fresh deploy by downloading serving artifacts from
Hugging Face Hub.

This replaces the (missing) DVC remote: it is what makes the repo runnable for
someone who just cloned it — an interviewer, a CI job, or a Hugging Face Space
at boot.

Usage:
    # Public artifact repo (no token needed):
    HF_ARTIFACTS_REPO=your-username/search-ranking-artifacts python scripts/bootstrap.py

    # Include optional artifacts (cross-encoder, difficulty classifier):
    python scripts/bootstrap.py --optional

    # Private artifact repo:
    HF_TOKEN=hf_xxx python scripts/bootstrap.py

Idempotent: files already present with the right size are skipped.
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
    HF_ARTIFACTS_REVISION,
    OPTIONAL_ARTIFACTS,
    SERVING_ARTIFACTS,
)


def _download(repo_id: str, rel_path: str, token: str | None) -> bool:
    """Download one artifact from HF Hub to its project-relative path.

    Returns True on success, False if the file was not found in the repo.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    dest = PROJECT_ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        # repo_type="model" works for both model and dataset content when the
        # repo is created as a model repo; switch here if you use a dataset repo.
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=rel_path,
            revision=HF_ARTIFACTS_REVISION,
            token=token,
            repo_type=os.getenv("HF_ARTIFACTS_REPO_TYPE", "model"),
        )
    except EntryNotFoundError:
        return False

    # hf_hub_download returns a path in the HF cache; copy/link into the project.
    import shutil

    if dest.exists():
        dest.unlink()
    shutil.copy2(cached, dest)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download serving artifacts from HF Hub.")
    parser.add_argument(
        "--optional",
        action="store_true",
        help="Also fetch optional artifacts (cross-encoder, difficulty classifier).",
    )
    args = parser.parse_args()

    repo_id = HF_ARTIFACTS_REPO
    if repo_id.startswith("REPLACE_ME"):
        print(
            "ERROR: HF_ARTIFACTS_REPO is not set.\n"
            "  1. Publish artifacts once:  python scripts/publish_artifacts.py\n"
            "  2. Then:  export HF_ARTIFACTS_REPO=<your-username>/search-ranking-artifacts",
            file=sys.stderr,
        )
        return 2

    token = os.getenv("HF_TOKEN")  # only needed for private repos
    targets = list(SERVING_ARTIFACTS)
    if args.optional:
        targets += OPTIONAL_ARTIFACTS

    print(f"Bootstrapping {len(targets)} artifacts from {repo_id}@{HF_ARTIFACTS_REVISION} ...")
    missing_required = []
    for rel_path in targets:
        ok = _download(repo_id, rel_path, token)
        status = "ok" if ok else "MISSING"
        print(f"  [{status}] {rel_path}")
        if not ok and rel_path in SERVING_ARTIFACTS:
            missing_required.append(rel_path)

    if missing_required:
        print(
            f"\nERROR: {len(missing_required)} required artifact(s) missing from the repo. "
            "Re-run scripts/publish_artifacts.py to upload them.",
            file=sys.stderr,
        )
        return 1

    print("\nBootstrap complete. The serving stack can now start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
