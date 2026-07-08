# Dialect-Robust Explainable Bangla Hate Speech Detection

Research pipeline for binary Bangla hate speech detection (BD-SHS Task A)
built around three evaluation axes and three proposed dialect-robustness
methods, with journal-grade statistics.

**Evaluation axes (protocol v2):**

1. **Clean test** — the official BD-SHS test split (near-duplicates of the
   training data flagged via a char-3-gram audit).
2. **Synthetic noisy test** — the test set perturbed with rule-based
   social-media noise (`test_augmented.csv`). *Synthetic — see
   [DATASETS.md](DATASETS.md).*
3. **Real dialect test** — `bidwesh_heldout`: the held-out half of BIDWESH
   (Chittagong/Noakhali/Barishal), grouped by source sentence so no
   sentence appears in both adaptation and evaluation, near-duplicates of
   BD-SHS train excluded.

**Proposed methods:**

- **DIA — Dialect-Informed Augmentation**: a standard→dialect word lexicon
  (~600 entries, e.g. আমার→মোর/আঁর, আপনি→আমনে) mined automatically from
  BIDWESH *adapt-split* parallel sentences (`src/data/dialect_lexicon.py`)
  and applied as training-time augmentation.
- **NCT — Noise-Consistency Training**: symmetric-KL consistency loss
  between clean and noised views of each training example
  (`src/models/consistency_trainer.py`).
- **Few-shot dialect adaptation**: continue fine-tuning on N ∈ {250, 500,
  1000, full} BIDWESH adapt examples (`src/models/adapt.py`) — data-
  efficiency curve + catastrophic-forgetting check.

**Rigor:** 3 seeds (42/43/44) for every main row, per-example prediction
dumps, bootstrap 95% CIs, paired-bootstrap + McNemar significance tests,
ERASER-style faithfulness metrics for LIME/SHAP, near-duplicate leakage
audit. Model families: TF-IDF LR/SVM, BanglaBERT, XLM-R, MuRIL, mBERT,
Qwen2.5-7B few-shot.

⚠️ Content warning: the datasets contain highly offensive Bangla text; they
are used exclusively to build detection systems.

## Setup

Python ≥ 3.10 (`pip install -r requirements.txt`). CPU auto-detected; GPU
needed only for full transformer training. On Windows the scripts force
UTF-8 output themselves.

## Data pipeline

```bash
python -m src.data.download          # 4 datasets, anonymous endpoints
python -m src.data.preprocess        # standard splits + BIDWESH 40/10/50 grouped split
python -m src.data.leakage_audit     # near-duplicate flags + reports/leakage_audit.md
python -m src.data.dialect_lexicon   # DIA lexicon from BIDWESH adapt split
python -m src.data.augment           # test_augmented, train_augmented,
                                     # train_dialect_aug, train_aug_dia
```

## Training

```bash
# Classical (CPU, minutes) — every variant x seed:
python -m src.models.baselines --model tfidf_lr --train-file data/processed/train_aug_dia.csv --seed 42

# Transformers (GPU) — variants via --train-file, NCT via --method:
python -m src.models.train_transformer --model-name csebuetnlp/banglabert --seed 42
python -m src.models.train_transformer --model-name csebuetnlp/banglabert \
    --train-file data/processed/train_aug_dia.csv --seed 42
python -m src.models.train_transformer --model-name csebuetnlp/banglabert \
    --method consistency --seed 42

# Few-shot dialect adaptation (short GPU runs from a trained checkpoint):
python -m src.models.adapt --from-dir outputs/models/banglabert_augmented_s42 --n 250

# LLM baseline (GPU, 4-bit):
python -m src.models.llm_baseline
```

Every run auto-evaluates on all v2 test sets, upserts
`outputs/results/results_summary.csv` (keyed by model × variant × test set
× seed) and dumps per-example predictions to `outputs/predictions/`.

### The full GPU experiment queue (Colab/Kaggle)

All 49 GPU jobs live in `scripts/experiments_manifest.json` (priority
order). On Colab (T4), run ONE cell and re-run it each session until it
prints `ALL DONE` — state and checkpoints persist on Drive:

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/Nafis878/ICCIT-1.git 2>/dev/null; %cd ICCIT-1
!git pull
!pip install -q lime accelerate sentencepiece bitsandbytes
!python -m src.data.download && python -m src.data.preprocess && \
 python -m src.data.leakage_audit && python -m src.data.dialect_lexicon && \
 python -m src.data.augment
!python scripts/colab_runner.py --state-dir /content/drive/MyDrive/iccit_q1_state --time-budget-min 200
```

When done: `!cd /content/drive/MyDrive/iccit_q1_state && zip -r q1_results.zip mirror`

## Analysis (local, CPU)

```bash
python -m src.evaluation.evaluate --all --skip-existing   # rebuild summaries
python -m src.evaluation.stats                            # mean±std, CIs, significance
python -m src.evaluation.slices                           # per-dialect/type/length slices
```

Outputs: `stats_summary.csv`, `significance_tests.csv`,
`robustness_summary.csv`, `slice_analysis.csv`, confusion matrices in
`outputs/figures/`. (v1 single-seed results are archived under
`outputs/results/archive_v1/`.)

## Explainability

```bash
python -m src.explainability.explain --model-dir outputs/models/<dir>        # LIME examples
python -m src.explainability.faithfulness --model-dir outputs/models/<dir> \
    --method lime            # comprehensiveness / sufficiency / deletion-AUC
```

Whitespace tokenization is used for both LIME and SHAP (python's `\W`
regex splits inside Bangla words — combining vowel signs are non-word
characters — which corrupts attributions).

## Repository layout

```
data/raw|processed/     datasets; dialect_lexicon.json; augmented train files
src/data/               download, preprocess, augment, dialect_lexicon, leakage_audit
src/models/             baselines, train_transformer, consistency_trainer, adapt, llm_baseline
src/evaluation/         metrics, evaluate, stats, slices
src/explainability/     explain, faithfulness
scripts/                colab_runner.py, experiments_manifest.json, make_manifest.py
outputs/                results/, predictions/, models/, figures/, explainability/
reports/                RESULTS.md, PAPER_NOTES.md, data_stats.md, leakage_audit.md
```

## Notes & caveats

- Synthetic augmentation ≠ dialect data; BIDWESH is the only real dialect
  set, and its adapt/dev halves never touch the held-out benchmark.
- The DIA lexicon is mined *only* from adapt-split sentence pairs.
- Transformers code targets the v5 API (`eval_strategy`,
  `processing_class`, collator-level column removal); also runs on ≥4.46.
- HS-BAN has no public download (see DATASETS.md §5).

### If Hugging Face model downloads stall

On some networks the HF Hub python downloader stalls while plain `curl`
works. Fetch the model files with curl and pass the local folder as
`--model-name`:

```powershell
$m = "hf-local\banglabert"
mkdir $m
foreach ($f in @("config.json","pytorch_model.bin","vocab.txt","tokenizer_config.json","special_tokens_map.json")) {
  curl.exe -SL -o "$m\$f" "https://huggingface.co/csebuetnlp/banglabert/resolve/main/$f"
}
python -m src.models.train_transformer --model-name "$m" --smoke
```
