# SP-1 Runbook — retrain the two-tower (run these on your own machine/GPU)

The code is done and committed (gold-inclusive corpus builder, `bm25s` hard-neg
mining, config, recall-based checkpointing). These are the long-running RUN steps
you execute in **your own terminal** (stable env + RTX 4060 + your HF credentials).
Every Python command is prefixed with `USE_TF=0 TRANSFORMERS_NO_TF=1` to stop
`transformers` from importing TensorFlow (which destabilizes this stack).

## 0. Safety — back up the current working artifacts first
```bash
mkdir -p data/_backup_pre_sp1
cp data/processed/passages.parquet        data/_backup_pre_sp1/ 2>/dev/null || true
cp data/processed/hard_negatives.parquet  data/_backup_pre_sp1/ 2>/dev/null || true
cp data/indexes/faiss_ivfpq.index         data/_backup_pre_sp1/ 2>/dev/null || true
cp data/indexes/docid_map.pkl             data/_backup_pre_sp1/ 2>/dev/null || true
cp -r models/two_tower                    data/_backup_pre_sp1/two_tower_old 2>/dev/null || true
```
If anything below goes wrong, restore from `data/_backup_pre_sp1/`.

## 1. Rebuild the gold-inclusive ~1M corpus + mine real hard negatives (~30–90 min)
```bash
# remove stale 500K-subset artifacts so preprocess regenerates them
rm -f data/processed/passages.parquet data/processed/hard_negatives.parquet \
      data/processed/train_triples.parquet \
      data/indexes/bm25_index.pkl data/indexes/bm25_pid_list.pkl
mkdir -p logs
USE_TF=0 TRANSFORMERS_NO_TF=1 python scripts/preprocess.py 2>&1 | tee logs/preprocess_sp1.log
```
**Acceptance (must pass before continuing):**
```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python -c "
import pandas as pd
p=pd.read_parquet('data/processed/passages.parquet')
tr=pd.read_parquet('data/processed/train_qrels.parquet'); dv=pd.read_parquet('data/processed/dev_qrels.parquet')
gold=set(tr['pid'])|set(dv['pid']); idx=set(p['pid'])
print('corpus:',len(p),'gold coverage:', round(len(gold&idx)/len(gold),4))"
```
Expect `gold coverage: 1.0` (was 0.02). If not ~1.0, stop and check the qrels union.

## 2. Retrain the two-tower (hours; unattended)
```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python training/train_two_tower.py 2>&1 | tee logs/train_sp1.log
```
- Watch `eval_recall_at_10` climb across epochs; `model_best.pt` is saved on the
  best-recall epoch (not loss).
- **If it OOMs on the 8 GB VRAM at `batch_size: 64`:** set `two_tower.batch_size: 32`
  in `configs/config.yaml` and rerun.

## 3. Rebuild the FAISS index + doc embeddings on the new corpus (~20–40 min)
```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python training/build_faiss_index.py 2>&1 | tee logs/faiss_sp1.log
```

## 4. Re-measure honestly (regenerates the committed JSON)
```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python scripts/eval_recall.py    # -> data/processed/two_tower_recall.json
USE_TF=0 TRANSFORMERS_NO_TF=1 python scripts/eval_beir.py      # -> data/processed/beir_results.json (uses your GPU now)
```
**Acceptance:** answerable Recall@100 materially above the current 0.32 (target 0.7–0.85).

## 5. Update the README + publish (honesty gate + your credentials)
- **README:** paste me the two regenerated JSON files (or just say "update the README")
  and I'll rewrite §11/§14 + the BEIR section from the real numbers — no hand-typed values.
- **Publish to HF Hub** (needs your account):
  ```bash
  # one-time: create a model repo at https://huggingface.co/new, then:
  export HF_TOKEN=hf_xxx
  export HF_ARTIFACTS_REPO=<your-hf-username>/search-ranking-artifacts
  python scripts/publish_artifacts.py
  ```

## After SP-1
Come back and we start **SP-2** (FastAPI retrieval API + Cloud Run) and **SP-3**
(SvelteKit frontend + client-side BYOK RAG). The specs/plans for those are next.
