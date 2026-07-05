# 0006 — ORCAS-calibrated click simulation, not raw human clicks

**Status:** Accepted
**Date:** 2026-07-03

## Context
A real feedback loop needs a click signal to retrain against. This project has
no real end users, so it has no real human click logs. Two public datasets
exist that are relevant but neither is sufficient alone:

- **MS MARCO qrels** give real, human-judged passage-level relevance, but no
  query popularity or click behavior — they are static labels, not a stream.
- **ORCAS** (Microsoft's public click dataset) gives real Bing query
  frequencies and real query→clicked-document pairs at scale, but it maps to
  MS MARCO *documents*, not passages, carries no click *position*, and is
  released under a **non-commercial, research-only license**.

Using either alone would either be static (qrels-only, no simulated click
volume/popularity) or overclaim (presenting ORCAS clicks as if they were
labeled, positioned clicks on the passages this system actually serves).

## Decision
Build a click **simulator** (`scripts/simulate_clicks.py`) that combines both
datasets honestly, each for what it actually provides:

- The **replay query stream** is MS MARCO queries that have qrels (train,
  falling back to dev) — every replayed query is guaranteed a real relevance
  label, so there is no coverage/starvation failure mode.
- **ORCAS calibrates, never labels**: query popularity (`query_popularity`)
  and mean click volume (`mean_clicks_per_query`) come from real ORCAS
  frequencies (`scripts/calibrate_orcas.py`), and are used only to *weight*
  which replayed queries are sampled more often and how many clicks to expect
  — never to decide what is relevant.
- **Relevance always comes from MS MARCO qrels**, via a position-based click
  model: `clicked ~ Bernoulli(propensity[rank] * relevance_ctr)`, where
  `relevance_ctr = 1.0` for passages in the query's qrels gold set and a small
  noise rate otherwise.
- **Position-bias propensity `eta`** is a documented **literature assumption**
  (the standard `1/rank^eta` position-bias curve from counterfactual
  learning-to-rank research), because ORCAS has no rank/position column and
  therefore cannot calibrate this term from data.
- **Every shown passage is logged as an impression**, not just clicked ones,
  so shown-but-not-clicked passages are recoverable as real negatives
  (`impressions - clicks`), fixing the earlier all-positive-label bug.
- Retraining (`scripts/retrain_from_clicks.py`, and the mirrored Airflow DAG)
  IPS-weights clicked rows by `1/propensity[rank]` to correct for position
  bias, and aborts if the resulting labels are degenerate (fewer than 2
  distinct values).
- Promotion (`scripts/promote.py::evaluate_and_gate`) evaluates the current
  production model and the staging candidate under the exact same evaluation
  harness, back-to-back, so the nDCG@10 comparison that gates promotion is
  never apples-to-oranges.
- `scripts/download_orcas.py` refuses to download ORCAS data unless the
  caller explicitly passes `--accept-noncommercial-license`, and prints the
  license terms first; clicked documents/URLs are never redistributed.

## Consequences
- **Pro:** the retraining pipeline exercises a genuinely realistic pipeline —
  real impressions, real negatives, real qrels-grounded relevance, a
  documented position-bias model, propensity-weighted labels, and a real
  one-harness promotion gate — end to end, without ever claiming to have
  collected human clicks that don't exist publicly.
- **Trade-off:** `eta` (position-bias strength) is an assumption, not
  measured; if the true position bias on this system's UI differs materially,
  the IPS weights would be miscalibrated. This is called out explicitly in
  the calibration output (`notes` field) rather than hidden.
- **Trade-off:** because relevance is always MS MARCO qrels rather than
  ORCAS-observed clicks, the simulation cannot surface relevance judgments
  that differ from the static qrels (e.g. a passage that real users would
  find useful but that qrels didn't judge as relevant to that query).
- **Non-commercial only:** because ORCAS's license is research-only, this
  project (and anything built on this calibration) must not be used
  commercially without separately licensing or replacing the ORCAS-derived
  calibration inputs.

## At 10× scale
Replace the simulator with real production click logs (this system already
logs `impression_logs`/`click_logs` in the right shape to support that
transition with zero schema change) and re-estimate `eta` from randomized
result-swapping experiments (e.g. RandTop-N / intervention-based propensity
estimation) instead of the literature-assumed curve.
