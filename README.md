# Dialect-Robust Explainable Bangla Hate Speech Detection

Research pipeline for binary Bangla hate speech detection (BD-SHS Task A)
with three evaluation axes:

1. **Clean test** — the official BD-SHS test split.
2. **Synthetic noisy test** — the same test set perturbed with rule-based
   social-media noise / informal spellings (`test_augmented.csv`).
   *Synthetic, not verified dialect data* — see the disclaimer in
   [DATASETS.md](DATASETS.md).
3. **Real dialect test** — BIDWESH: BD-SHS sentences manually translated
   into Chittagong, Noakhali and Barishal dialects.

Models: majority baseline, TF-IDF + Logistic Regression, TF-IDF + Linear
SVM, and fine-tuned transformers (BanglaBERT, XLM-RoBERTa-base). The
robustness method is **augmentation training** (train on
`train_augmented.csv`, compare against clean-trained). Explainability via
LIME (SHAP fallback).

⚠️ Content warning: the datasets contain highly offensive Bangla text; they
are used exclusively to build detection systems.

## Setup

Python ≥ 3.10. On Windows the scripts force UTF-8 output themselves.

```bash
pip install -r requirements.txt
```

CPU is auto-detected; a GPU is only needed for *full* transformer training.

## Pipeline

Run everything from the repository root.

```bash
# 1. Download datasets (BD-SHS, BIDWESH, 30K, bn_hate_speech; ~15 MB total)
python -m src.data.download
#    -> data/raw/..., data/raw/download_status.json; details in DATASETS.md

# 2. Preprocess -> standardized splits (text, text_clean, label, target,
#    hate_type, source_dataset) + label_mapping.json + reports/data_stats.md
python -m src.data.preprocess

# 3. Synthetic noisy/dialect-style augmentation
python -m src.data.augment                     # test_augmented.csv, train_augmented.csv
```

### Smoke tests (fast end-to-end checks, CPU-friendly)

```bash
python -m src.models.baselines --model all --smoke
python -m src.models.train_transformer --model-name csebuetnlp/banglabert --smoke
python -m src.models.train_transformer --model-name xlm-roberta-base --smoke
```

Smoke runs train on tiny subsamples, evaluate on 500-row samples of each
test set, and record rows tagged `smoke` under model ids ending in
`_smoke` — they never overwrite full results.

### Classical baselines (full data, minutes on CPU)

```bash
python -m src.models.baselines --model all                # clean-trained
python -m src.models.baselines --model all \
    --train-file data/processed/train_augmented.csv       # robustness-trained
```

### Transformers (full fine-tuning; GPU strongly recommended)

```bash
python -m src.models.train_transformer --model-name csebuetnlp/banglabert \
    --epochs 3 --batch-size 16 --lr 2e-5 --max-len 128
python -m src.models.train_transformer --model-name xlm-roberta-base \
    --epochs 3 --batch-size 16 --lr 2e-5 --max-len 128
# robustness-trained variants:
python -m src.models.train_transformer --model-name csebuetnlp/banglabert \
    --train-file data/processed/train_augmented.csv
```

All knobs: `--model-name --task --batch-size --lr --epochs --max-len
--output-dir --train-file --smoke --seed`. On this repo's CPU-only dev
machine a full run takes many hours — the scripts are smoke-verified
locally and meant to be run on GPU (Colab/Kaggle) for full results.

### Evaluation

Training scripts evaluate automatically. To (re-)evaluate saved models:

```bash
python -m src.evaluation.evaluate --all              # every outputs/models/*/
python -m src.evaluation.evaluate --model-dir outputs/models/tfidf_lr
```

Outputs:

- `outputs/results/results_summary.csv` — one row per model × train-variant
  × test set (accuracy, macro/weighted P/R/F1, per-class F1)
- `outputs/results/classification_report_*.json` — full reports incl.
  confusion matrices
- `outputs/results/robustness_summary.csv` — macro-F1 drops clean→noisy and
  clean→BIDWESH (full, non-overlapping subset, per dialect)
- `outputs/figures/confusion_matrix_*.png`

BIDWESH rows whose Standard-Bangla source also appears in BD-SHS train/val
are flagged and excluded from the `bidwesh_clean` numbers (exact-match
check — a lower bound on overlap; only ~0.3% of rows).

### Explainability

```bash
python -m src.explainability.explain --model-dir outputs/models/tfidf_lr
# transformer model / SHAP variant:
python -m src.explainability.explain --model-dir outputs/models/banglabert --method shap
```

Explains 10 correct + 10 incorrect test predictions → per-example HTML,
`explanations_summary.json`, and an index in `outputs/explainability/`.

## Repository layout

```
data/raw/            unmodified downloads (+ download_status.json)
data/processed/      standardized splits, augmented sets, label_mapping.json
src/data/            download.py, preprocess.py, augment.py
src/models/          baselines.py, train_transformer.py
src/evaluation/      metrics.py, evaluate.py
src/explainability/  explain.py
src/utils/           common.py, normalize.py
outputs/             results/, models/, figures/, explainability/
reports/             data_stats.md, RESULTS.md
```

## Running on Google Colab (T4 GPU)

Runtime → Change runtime type → **T4 GPU**, then run these cells in order:

```python
# 1. Get the code
!git clone https://github.com/Nafis878/ICCIT-1.git
%cd ICCIT-1

# 2. Dependencies (Colab already ships torch/transformers; this adds the rest)
!pip install -q lime accelerate sentencepiece

# 3. Data pipeline (~2 min)
!python -m src.data.download
!python -m src.data.preprocess
!python -m src.data.augment

# 4. Full transformer training on T4 (fp16 auto-enabled on GPU)
#    ~40-60 min each for BanglaBERT, ~1-1.5 h each for XLM-R
!python -m src.models.train_transformer --model-name csebuetnlp/banglabert --epochs 3 --batch-size 32 --lr 2e-5 --max-len 128
!python -m src.models.train_transformer --model-name csebuetnlp/banglabert --epochs 3 --batch-size 32 --lr 2e-5 --max-len 128 --train-file data/processed/train_augmented.csv
!python -m src.models.train_transformer --model-name xlm-roberta-base --epochs 3 --batch-size 16 --lr 2e-5 --max-len 128
!python -m src.models.train_transformer --model-name xlm-roberta-base --epochs 3 --batch-size 16 --lr 2e-5 --max-len 128 --train-file data/processed/train_augmented.csv

# 5. Rebuild the robustness summary over everything trained so far
!python -m src.evaluation.evaluate --all --skip-existing

# 6. Explainability for the best transformer (LIME; use --method shap as fallback)
!python -m src.explainability.explain --model-dir outputs/models/banglabert_aug --num-samples 500

# 7. Save results before the runtime dies (models are big; results are small)
!zip -r results.zip outputs/results outputs/figures outputs/explainability reports
from google.colab import files; files.download("results.zip")
# optional, large (~450 MB per model): zip outputs/models/banglabert* too
```

Each training run auto-evaluates on the clean, synthetic-noisy and BIDWESH
dialect test sets and appends to `outputs/results/results_summary.csv`.
If a Colab session dies mid-way, just rerun from step 3 — training scripts
overwrite their own model dir, and evaluation rows are upserted, never
duplicated.

### If Hugging Face model downloads stall

On some networks the HF Hub python downloader stalls while plain `curl`
works. Workaround — fetch the model files with curl and pass the local
folder as `--model-name`:

```powershell
$m = "hf-local\banglabert"
mkdir $m
foreach ($f in @("config.json","pytorch_model.bin","vocab.txt","tokenizer_config.json","special_tokens_map.json")) {
  curl.exe -SL -o "$m\$f" "https://huggingface.co/csebuetnlp/banglabert/resolve/main/$f"
}
python -m src.models.train_transformer --model-name "$m" --smoke
```

(For `xlm-roberta-base` the files are `config.json`, `model.safetensors`,
`sentencepiece.bpe.model`, `tokenizer.json`, `tokenizer_config.json`.)

## Notes & caveats

- **Synthetic augmentation ≠ dialect data.** `augment.py` output is
  rule-based noise imitating informal Bangla; only BIDWESH is real,
  human-produced dialect data.
- BD-SHS official train/val/test splits are kept as published.
- Transformers code targets the v5 API (`eval_strategy`,
  `processing_class`); it also runs on transformers ≥ 4.46.
- If `csebuetnlp/banglabert` ever breaks on a new transformers major
  version, `sagorsarker/bangla-bert-base` is a drop-in `--model-name`
  alternative.
- HS-BAN could not be included: no official public download exists (see
  DATASETS.md §5).
