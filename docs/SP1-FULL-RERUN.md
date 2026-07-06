# SP-1 Full-Scale Rerun — in a clean venv (do this when the GPU is free)

The first SP-1 run was crippled by the **anaconda-base env's tokenizer running
~100× too slow** (~1,200 seq/sec), which forced a 20K-query / 2-epoch subset —
too little training to show the hard-negatives benefit, and it slightly hurt
out-of-domain transfer. A **clean pip venv** almost always fixes the tokenizer
speed (the slowness is an anaconda/MKL-threading issue, not the code). This
runbook does the full-scale run properly.

**Nothing here is destructive to your current results** — the reduced-run model
+ numbers stay committed, and your pre-SP1 baseline is backed up at
`data/_backup_pre_sp1/`. Back up the reduced-run model first if you want to keep it:
`cp -r models/two_tower data/_backup_reduced_run_two_tower`

Every command uses `PYTHONIOENCODING=utf-8` (Windows cp1252 console) and
`USE_TF=0 TRANSFORMERS_NO_TF=1` (keeps TensorFlow out of the process).

---

## Step 0 — Create the clean venv (NOT anaconda base)

In **PowerShell**, from the repo root:
```powershell
py -3.11 -m venv .venv                       # a fresh, isolated Python 3.11
.\.venv\Scripts\Activate.ps1                  # prompt should now show (.venv)
python -m pip install --upgrade pip
# GPU torch (CUDA 12.1) FIRST, from the CUDA index:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -r requirements-dev.txt
pip install bm25s==0.3.9
```
(Bash/git-bash equivalent activation: `source .venv/Scripts/activate`.)

## Step 1 — VERIFY the tokenizer is fast (the whole point)
```powershell
python scripts/tokenizer_speed_test.py
```
- **HEALTHY (>=5k seq/sec):** proceed. The full run will be hours, not the
  pathological startup we hit before.
- **SLOW (<5k/sec):** stop — the venv didn't fix it. Options: `pip install -U
  tokenizers`, or set `OMP_NUM_THREADS=8` / `TOKENIZERS_PARALLELISM=true`, then
  re-test. Don't launch the full run until this is healthy.

Also confirm the GPU is visible:
```powershell
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Step 2 — Restore full-scale config
Edit `configs/config.yaml` `two_tower:` block back to full scale (the reduced-run
values were an env workaround):
```yaml
  hard_negatives_per_query: 5
  batch_size: 16               # keep 16 for 8GB VRAM (7 seqs/query). See AMP note below.
  epochs: 3
  warmup_steps: 1000
  eval_max_distractors: 100000
```
Leave the `data:` block as-is (`hard_neg_max_queries: 150000`, `target_corpus_size: 1000000`).

## Step 3 — Re-mine the full 150K hard negatives
The reduced run subset `hard_negatives.parquet` to 20K, so re-mine (the 1M
corpus + qrels/queries already exist and are reused):
```powershell
Remove-Item data/processed/hard_negatives.parquet -Force -ErrorAction SilentlyContinue
PYTHONIOENCODING=utf-8 USE_TF=0 TRANSFORMERS_NO_TF=1 python scripts/preprocess.py
# acceptance: hard_negatives.parquet ~150,000 rows
```

## Step 4 — Full retrain (hours, unattended)
```powershell
PYTHONIOENCODING=utf-8 USE_TF=0 TRANSFORMERS_NO_TF=1 python training/train_two_tower.py 2>&1 | Tee-Object logs/train_full.log
```
Watch `eval_recall_at_10` climb across the 3 epochs; `model_best.pt` is saved on
the best-recall epoch. With a fast tokenizer this is ~real training time, not
the ~9-min stall per launch we hit before.

## Step 5 — Rebuild the index (delete stale embeddings FIRST — this bit us once)
```powershell
Remove-Item data/embeddings/doc_embeddings.npy -Force -ErrorAction SilentlyContinue
Remove-Item data/indexes/faiss_ivfpq.index -Force -ErrorAction SilentlyContinue
Remove-Item data/indexes/docid_map.pkl -Force -ErrorAction SilentlyContinue
PYTHONIOENCODING=utf-8 USE_TF=0 TRANSFORMERS_NO_TF=1 python training/build_faiss_index.py
```
(`build_faiss_index` silently reuses `doc_embeddings.npy` if it exists — always delete it before a rerun.)

## Step 6 — Re-measure + document + publish
```powershell
PYTHONIOENCODING=utf-8 USE_TF=0 TRANSFORMERS_NO_TF=1 python scripts/eval_recall.py   # -> two_tower_recall.json
PYTHONIOENCODING=utf-8 USE_TF=0 TRANSFORMERS_NO_TF=1 python scripts/eval_beir.py     # -> beir_results.json
```
Then paste me the two JSONs (or say "update the README") and I'll write §11/§14/BEIR
from the real numbers. Publish to HF Hub needs your `HF_TOKEN` + `HF_ARTIFACTS_REPO`.

---

## Recommended enhancement (ask me to implement when the GPU is free): mixed precision
The training ran ~1.5 s/step at ~20% GPU util (overhead-bound). Adding
**AMP (autocast + GradScaler)** to `train_two_tower.py` would roughly halve step
time AND free VRAM to raise `batch_size` to 32 — cutting a full 150K/3-epoch run
from ~6 h toward ~2 h, with more in-batch negatives (better quality). It's a
~15-line, well-understood change I can make + test in the clean venv. Say the
word and I'll add it before the rerun.
