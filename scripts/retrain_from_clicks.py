"""
Click-feedback retraining — the free (GitHub Actions) replacement for the
Airflow DAG. Mirrors airflow_dags/retraining_dag.py but runs as a single CLI
so it can be scheduled on free CI instead of an always-on Airflow.

Flow:
  1. Read labeled impressions from Postgres (Neon via DATABASE_URL): every
     SHOWN passage, left-joined against click_logs. If fewer than
     RETRAINING_CLICK_THRESHOLD rows, exit 0 (nothing to do).
  2. Build the shared 7-feature LambdaRank vectors (BM25 + two-tower +
     passage stats) via services.shared.features.build_lambdarank_features
     — the SAME builder the live serve path uses, so train == serve.
  3. Label each row `clicked` (1.0) or shown-not-clicked (0.0) — NEVER all-1
     — and IPS-weight clicked rows by `1/propensity[rank]` to correct for
     position bias (top-ranked clicks are "easier" to get and would
     otherwise be over-counted).
  4. Retrain XGBoost rank:ndcg. Abort (no train, no publish) if the labels
     that came out are degenerate (fewer than 2 distinct values) — this is
     the guard for the bug this module exists to fix.
  5. Save models/lambdarank/lambdarank.json and (if HF creds are set)
     publish it back to the HF Hub artifact repo so the live Space picks it
     up on restart.

Prereqs: serving artifacts present (run scripts/bootstrap.py first) and
DATABASE_URL pointing at the clicks/impressions DB.

Note: the full NDCG@10 promotion gate runs in the local Airflow pipeline (it
needs the dev-set eval harness). This CI job is threshold-gated retrain +
publish; treat the Airflow DAG as the gated path and this as the lightweight
scheduled path.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from services.shared.database import get_engine
from services.shared.features import FEATURE_NAMES, Candidate, build_lambdarank_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

THRESHOLD = int(os.getenv("RETRAINING_CLICK_THRESHOLD", "1000"))

DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "data" / "processed" / "orcas_calibration.json"


def load_labeled_impressions(engine) -> pd.DataFrame:
    """Every shown passage, labeled with whether it was clicked.

    `impression_logs LEFT JOIN click_logs ON (request_id, doc_id)`: rows with
    no matching click are real negatives (shown-but-not-clicked), fixing the
    previous bug where every retrain label was hard-coded to `1`.

    Returns columns: request_id, query_text, doc_id, rank_shown, clicked(bool).
    """
    from sqlalchemy import text

    query = text(
        """
        SELECT
            i.request_id AS request_id,
            i.query_text AS query_text,
            i.doc_id AS doc_id,
            i.rank_shown AS rank_shown,
            (c.doc_id IS NOT NULL) AS clicked
        FROM impression_logs i
        LEFT JOIN click_logs c
            ON i.request_id = c.request_id AND i.doc_id = c.doc_id
        ORDER BY i.request_id, i.rank_shown
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df["clicked"] = df["clicked"].astype(bool)
    return df


class TwoTowerRetriever:
    """Lazily-loaded two-tower scorer + passage-text lookup.

    Rebuilds `Candidate` rows (doc_id, text, score, retrieval_rank) for
    historic impressions at retrain time, mirroring the serve-time
    fusion/two-tower score so `build_lambdarank_features` sees the same
    shape of input it sees live. This class is intentionally NOT covered by
    unit tests (it loads a real torch model) — tests inject a stub retriever
    exposing the same `get_candidates` method instead.
    """

    def __init__(self, pid_to_text: dict[int, str], tt_model, tok, device: str):
        self.pid_to_text = pid_to_text
        self.tt_model = tt_model
        self.tok = tok
        self.device = device

    def get_candidates(
        self, query_text: str, doc_ids: list[int], ranks: list[int]
    ) -> list[Candidate]:
        import torch

        enc_q = self.tok(query_text, max_length=64, truncation=True, return_tensors="pt")
        with torch.no_grad():
            q_emb = (
                self.tt_model.encode_query(
                    enc_q["input_ids"].to(self.device), enc_q["attention_mask"].to(self.device)
                )
                .cpu()
                .numpy()
            )

        candidates = []
        for doc_id, rank in zip(doc_ids, ranks):
            doc_text = self.pid_to_text.get(doc_id, "")
            enc_d = self.tok(doc_text, max_length=180, truncation=True, return_tensors="pt")
            with torch.no_grad():
                d_emb = (
                    self.tt_model.encode_doc(
                        enc_d["input_ids"].to(self.device), enc_d["attention_mask"].to(self.device)
                    )
                    .cpu()
                    .numpy()
                )
            score = float((d_emb @ q_emb.T).squeeze())
            candidates.append(
                Candidate(doc_id=doc_id, text=doc_text, score=score, retrieval_rank=rank)
            )
        return candidates


def _load_retriever() -> TwoTowerRetriever:
    import torch

    from training.two_tower_model import load_two_tower

    data = PROJECT_ROOT / "data"
    models = PROJECT_ROOT / "models"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tt_model, tok = load_two_tower(str(models / "two_tower"), device=device)

    pdf = pd.read_parquet(data / "processed" / "passages.parquet")
    pid_to_text = dict(zip(pdf["pid"], pdf["text"]))
    return TwoTowerRetriever(pid_to_text, tt_model, tok, device)


def _load_bm25():
    data = PROJECT_ROOT / "data"
    with open(data / "indexes" / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(data / "indexes" / "bm25_pid_list.pkl", "rb") as f:
        bm25_pid_list = pickle.load(f)

    pdf = pd.read_parquet(data / "processed" / "passages.parquet")
    pid_to_len = dict(zip(pdf["pid"], pdf["token_count"]))
    return bm25, bm25_pid_list, pid_to_len


def _load_propensity() -> dict[int, float]:
    """Load the ORCAS-calibrated propensity curve (see
    scripts/calibrate_orcas.py). Its `propensity` dict is string-keyed once
    round-tripped through JSON, so keys are re-cast to int here (see
    scripts/simulate_clicks.py module docstring for the same caveat). Falls
    back to a documented default decay curve if calibration hasn't been run
    yet, so retraining never crashes for lack of a calibration file.
    """
    if not DEFAULT_CALIBRATION_PATH.exists():
        return {rank: 1.0 / rank for rank in range(1, 11)}
    with open(DEFAULT_CALIBRATION_PATH) as f:
        calibration = json.load(f)
    return {int(k): v for k, v in calibration.get("propensity", {}).items()}


def build_training_matrix(
    labeled: pd.DataFrame,
    retriever,
    bm25,
    bm25_pid_list,
    pid_to_len,
    propensity: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Build (X, y, weights, groups) from labeled impressions.

    - `y` = 1.0 for clicked rows, 0.0 for shown-but-not-clicked rows. Both
      values appear whenever a group has at least one click and one
      non-click — this is NEVER all-1 (the bug this module fixes).
    - `weights` = inverse-propensity-score (IPS) weight `1/propensity[rank]`
      for clicked rows (corrects for position bias: a rank-1 click is a
      "weaker" positive signal than an equally clicked rank-10 result, so it
      gets down-weighted relative to it), `1.0` for non-clicked rows.
    - `groups` = number of shown passages per request_id, for
      `xgb.DMatrix.set_group`.

    Features come from `services.shared.features.build_lambdarank_features`
    — the same builder the live serve path uses — so retrain features never
    skew from serve features.
    """
    fallback = min(propensity.values()) if propensity else 1.0

    X_parts: list[np.ndarray] = []
    y_list: list[float] = []
    w_list: list[float] = []
    groups: list[int] = []

    for _request_id, group in labeled.groupby("request_id", sort=False):
        group = group.sort_values("rank_shown")
        query_text = str(group["query_text"].iloc[0])
        doc_ids = group["doc_id"].astype(int).tolist()
        ranks = group["rank_shown"].astype(int).tolist()

        candidates = retriever.get_candidates(query_text, doc_ids, ranks)
        feats = build_lambdarank_features(query_text, candidates, bm25, bm25_pid_list, pid_to_len)
        X_parts.append(np.asarray(feats, dtype=np.float32))

        for clicked, rank in zip(group["clicked"].tolist(), ranks):
            clicked = bool(clicked)
            y_list.append(1.0 if clicked else 0.0)
            w_list.append((1.0 / propensity.get(rank, fallback)) if clicked else 1.0)
        groups.append(len(group))

    X = (
        np.vstack(X_parts).astype(np.float32)
        if X_parts
        else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    )
    y = np.asarray(y_list, dtype=np.float32)
    weights = np.asarray(w_list, dtype=np.float32)
    return X, y, weights, groups


def is_degenerate(y: np.ndarray) -> bool:
    """True if `y` has fewer than 2 distinct values — no ranking signal to
    learn from. Guards against ever training/publishing on all-1 (or all-0)
    labels again."""
    return len(np.unique(y)) < 2


def train_and_save(X: np.ndarray, y: np.ndarray, weights: np.ndarray, groups: list[int]) -> Path:
    import xgboost as xgb

    dtrain = xgb.DMatrix(X, label=y, weight=weights)
    dtrain.set_group(groups if groups else [len(X)])
    params = {
        "objective": "rank:ndcg", "eval_metric": "ndcg@10",
        "eta": 0.05, "max_depth": 6, "subsample": 0.8, "tree_method": "hist", "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=300)
    out = PROJECT_ROOT / "models" / "lambdarank" / "lambdarank.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out))
    return out


def _publish(model_path: Path) -> None:
    repo = os.getenv("HF_ARTIFACTS_REPO", "")
    token = os.getenv("HF_TOKEN")
    if not repo or repo.startswith("REPLACE_ME") or not token:
        print("HF creds not set — skipping publish (model saved locally only).")
        return
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="models/lambdarank/lambdarank.json",
        repo_id=repo,
        repo_type=os.getenv("HF_ARTIFACTS_REPO_TYPE", "model"),
    )
    print(f"Published updated LambdaRank model to {repo}.")


def main() -> int:
    engine = get_engine()
    labeled = load_labeled_impressions(engine)
    n = len(labeled)
    print(f"Loaded {n} labeled impression rows (threshold {THRESHOLD}).")
    if n < THRESHOLD:
        print("Below threshold — nothing to retrain.")
        return 0

    retriever = _load_retriever()
    bm25, bm25_pid_list, pid_to_len = _load_bm25()
    propensity = _load_propensity()

    X, y, weights, groups = build_training_matrix(
        labeled, retriever, bm25, bm25_pid_list, pid_to_len, propensity
    )
    if len(X) == 0:
        print("No usable impression features — exiting.")
        return 0
    print(f"Built {len(X)} feature rows across {len(groups)} request groups.")

    if is_degenerate(y):
        print("Degenerate labels — abort")
        return 0

    model_path = train_and_save(X, y, weights, groups)
    print(f"Saved retrained model to {model_path}.")

    # Record what happened for the workflow log.
    summary = {"impressions": int(n), "feature_rows": int(len(X)), "groups": int(len(groups))}
    print("RETRAIN_SUMMARY " + json.dumps(summary))

    _publish(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
