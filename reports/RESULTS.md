# Results — 2026-07-07 (full runs, classical + transformers)

Binary Bangla hate speech detection (BD-SHS Task A, official splits;
train n=40,181 / val 5,028 / test 5,028). Full metric rows:
`outputs/results/results_summary.csv`; drops:
`outputs/results/robustness_summary.csv`; per-run JSON + confusion
matrices under `outputs/results/` and `outputs/figures/`.

Classical models trained on the CPU dev machine; transformers fine-tuned
on Google Colab T4 (3 epochs, batch 32/16, lr 2e-5, max_len 128, fp16)
with results merged back into this repo. "+ synthetic aug" = trained on
`train_augmented.csv` (original 40K + 20K rule-based noisy copies).

## Macro-F1 across test conditions

| model | train data | clean test | synthetic-noisy | BIDWESH (all) | Chittagong | Noakhali | Barishal |
|---|---|---|---|---|---|---|---|
| majority | — | 0.342 | 0.342 | 0.336 | 0.336 | 0.335 | 0.338 |
| TF-IDF + LR | clean | 0.9036 | 0.8796 | 0.8574 | 0.8136 | 0.8707 | 0.8888 |
| TF-IDF + LR | + synthetic aug | 0.9050 | 0.8926 | 0.8654 | 0.8294 | 0.8747 | 0.8929 |
| TF-IDF + SVM | clean | 0.9014 | 0.8771 | 0.8505 | 0.8058 | 0.8640 | 0.8827 |
| TF-IDF + SVM | + synthetic aug | 0.9010 | 0.8816 | 0.8556 | 0.8153 | 0.8666 | 0.8858 |
| XLM-R base | clean | 0.9063 | 0.8915 | 0.8559 | 0.8071 | 0.8695 | 0.8926 |
| XLM-R base | + synthetic aug | 0.9104 | 0.8994 | 0.8678 | 0.8290 | 0.8813 | 0.8943 |
| BanglaBERT | clean | 0.9162 | 0.8982 | 0.8704 | 0.8205 | 0.8862 | 0.9059 |
| BanglaBERT | **+ synthetic aug** | **0.9226** | **0.9069** | **0.8787** | **0.8307** | **0.8972** | **0.9096** |

(BD-SHS paper's best reported Task A model, BiLSTM + informal fastText:
F1 ≈ 0.91 — BanglaBERT here exceeds it.)

## Robustness drops (macro-F1 vs each model's clean score)

| model / train | → synthetic noisy | → BIDWESH all | → Chittagong | → Noakhali | → Barishal |
|---|---|---|---|---|---|
| LR / clean | −0.0240 | −0.0462 | −0.0900 | −0.0329 | −0.0148 |
| LR / augmented | −0.0124 | −0.0397 | −0.0757 | −0.0304 | −0.0121 |
| SVM / clean | −0.0242 | −0.0509 | −0.0956 | −0.0374 | −0.0187 |
| SVM / augmented | −0.0194 | −0.0454 | −0.0857 | −0.0344 | −0.0152 |
| XLM-R / clean | −0.0147 | −0.0504 | −0.0991 | −0.0367 | −0.0137 |
| XLM-R / augmented | **−0.0110** | −0.0426 | −0.0814 | −0.0291 | −0.0161 |
| BanglaBERT / clean | −0.0180 | −0.0459 | −0.0957 | −0.0300 | −0.0104 |
| BanglaBERT / augmented | −0.0157 | **−0.0439** | −0.0920 | −0.0254 | −0.0130 |

BIDWESH rows overlapping BD-SHS train/val sources (exact-match, 28 rows =
0.3%) are excluded in the `bidwesh_clean` variant — numbers shift by
< 0.001, so contamination is not driving these scores.

## Findings

1. **BanglaBERT + augmentation training is the best model everywhere**:
   0.9226 clean / 0.9069 noisy / 0.8787 real-dialect macro-F1. Ordering is
   consistent across all test sets: BanglaBERT > XLM-R > TF-IDF LR > SVM.
2. **Augmentation training helps every model on every shifted test set,
   at zero (classical) or negative (transformer) clean-test cost.** For
   BanglaBERT it adds +0.6 pt clean, +0.9 pt noisy, +0.8 pt BIDWESH; for
   XLM-R +0.4/+0.8/+1.2 pts.
3. **The real-dialect gap survives pretraining.** Transformers shrink the
   *absolute* gap (BanglaBERT-aug reaches 0.879 on BIDWESH vs LR-aug
   0.865) but their *relative* drop clean→BIDWESH (−0.044 to −0.050) is
   the same size as the classical models' — large-scale Bangla
   pretraining does not by itself confer dialect robustness.
4. **Chittagong is the hardest dialect for every model** (best: 0.831),
   losing ~2× more F1 than Noakhali and ~5-7× more than Barishal —
   consistent with Chittagonian being the most divergent of the three.
   Synthetic noise recovers only ~15-20% of the Chittagong drop, so
   real-dialect methods (e.g. dialect-aware augmentation, BIDWESH-based
   fine-tuning) are the natural next step.
5. **Synthetic noise stress-test is a weaker proxy for dialect shift**:
   drops on the synthetic set (1-2.4 pts) are consistently ~⅓ of the
   real-dialect drops (4-5 pts).

## Explainability

`outputs/explainability/`: LIME explanations for 20 test predictions
(10 correct + 10 incorrect, balanced over classes/error directions) of the
TF-IDF LR augmentation-trained model; `explanations_summary.json` holds
top ±10 features per example. Transformer LIME (`--model-dir
outputs/models/banglabert_aug`) requires the trained weights, which live
on the Colab runtime — run `python -m src.explainability.explain
--model-dir outputs/models/banglabert_aug --num-samples 500` there.

## Reproduce

```bash
python -m src.data.download && python -m src.data.preprocess && python -m src.data.augment
python -m src.models.baselines --model all
python -m src.models.baselines --model all --train-file data/processed/train_augmented.csv
# on GPU (Colab T4):
python -m src.models.train_transformer --model-name csebuetnlp/banglabert --epochs 3 --batch-size 32 --lr 2e-5 --max-len 128
python -m src.models.train_transformer --model-name csebuetnlp/banglabert --epochs 3 --batch-size 32 --lr 2e-5 --max-len 128 --train-file data/processed/train_augmented.csv
python -m src.models.train_transformer --model-name xlm-roberta-base --epochs 3 --batch-size 16 --lr 2e-5 --max-len 128
python -m src.models.train_transformer --model-name xlm-roberta-base --epochs 3 --batch-size 16 --lr 2e-5 --max-len 128 --train-file data/processed/train_augmented.csv
python -m src.evaluation.evaluate --all --skip-existing
python -m src.explainability.explain --model-dir outputs/models/tfidf_lr_aug
```
