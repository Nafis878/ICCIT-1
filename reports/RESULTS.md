# Results snapshot — 2026-07-06

Binary Bangla hate speech detection (BD-SHS Task A, official splits;
train n=40,181 / val 5,028 / test 5,028). Full metric rows:
`outputs/results/results_summary.csv`; drops:
`outputs/results/robustness_summary.csv`; per-run JSON + confusion
matrices under `outputs/results/` and `outputs/figures/`.

Machine: CPU-only (torch 2.9.1+cpu, transformers 5.8.1). Classical models
= full-data results. Transformers = **smoke-verified scripts only** (tiny
subsample; full fine-tuning is meant for a GPU box — commands in README).

## Full-data results (classical models), macro-F1

| model | train data | clean test | synthetic-noisy | BIDWESH (all) | Chittagong | Noakhali | Barishal |
|---|---|---|---|---|---|---|---|
| majority | — | 0.342 | 0.342 | 0.336 | 0.336 | 0.335 | 0.338 |
| TF-IDF + LR | clean | **0.9036** | 0.8796 | 0.8574 | 0.8136 | 0.8707 | 0.8888 |
| TF-IDF + LR | **+ synthetic aug** | **0.9050** | **0.8926** | **0.8654** | **0.8294** | **0.8747** | **0.8929** |
| TF-IDF + SVM | clean | 0.9014 | 0.8771 | 0.8505 | 0.8058 | 0.8640 | 0.8827 |
| TF-IDF + SVM | + synthetic aug | 0.9010 | 0.8816 | 0.8556 | 0.8153 | 0.8666 | 0.8858 |

(For reference: the BD-SHS paper's best full model, BiLSTM + informal
fastText embeddings, reports F1 ≈ 0.91 on Task A.)

## Robustness drops (macro-F1, relative to each model's clean test score)

| model / train | → synthetic noisy | → BIDWESH all | → Chittagong | → Noakhali | → Barishal |
|---|---|---|---|---|---|
| LR / clean | −0.0240 | −0.0462 | −0.0900 | −0.0329 | −0.0148 |
| LR / augmented | **−0.0124** | **−0.0397** | **−0.0757** | −0.0304 | −0.0121 |
| SVM / clean | −0.0242 | −0.0509 | −0.0956 | −0.0374 | −0.0187 |
| SVM / augmented | −0.0194 | −0.0454 | −0.0857 | −0.0344 | −0.0152 |

BIDWESH rows overlapping BD-SHS train/val sources (exact-match, 28 rows =
0.3%) are also excluded in a `bidwesh_clean` variant — numbers shift by
< 0.001, so contamination is not driving these scores.

## Findings so far

1. **Strong classical baseline.** TF-IDF (word 1–2g + char 2–5g) + LR
   reaches 0.904 macro-F1 on the official BD-SHS test — within ~0.6 pt of
   the paper's best deep model, at ~6 s training cost on CPU.
2. **Real dialect shift hurts 2–4× more than synthetic noise.** Clean→
   noisy costs the clean-trained LR 2.4 pts; clean→BIDWESH costs 4.6 pts,
   and Chittagong alone costs 9.0 pts (Chittagonian is the most divergent
   of the three dialects — consistent with linguistics).
3. **Synthetic augmentation training helps robustness at zero clean-test
   cost.** For LR it halves the synthetic-noise drop (2.4→1.2 pts) and
   recovers ~1.4 pts on Chittagong, +0.8 pts on BIDWESH overall, while
   clean F1 *improves* slightly (0.9036→0.9050). Same direction for SVM.
   I.e. cheap rule-based noise transfers measurably to *real* dialect
   robustness, but closes only ~15–20% of the dialect gap — headroom for
   dialect-aware methods.
4. **Per-class behavior** (see `classification_report_*.json`): hate-class
   F1 is consistently a few points below not-hate; augmentation training
   narrows the gap on noisy/dialect sets.

## Transformer smoke runs (NOT full results — sanity checks only)

| run | val macro-F1 | note |
|---|---|---|
| BanglaBERT smoke (200 train rows, 1 epoch, CPU, 33 s) | 0.363 | pipeline verified end-to-end on transformers 5.8.1; barely trained by design |
| XLM-R smoke (200 train rows, 1 epoch, CPU, 35 s) | 0.306 | same — script verified, model saved + auto-evaluated |

Smoke rows are tagged `smoke` in `results_summary.csv` (evaluated on
500-row samples of each test set) and never overwrite full results.

## Explainability

`outputs/explainability/`: LIME explanations for 20 test predictions of
the best model (TF-IDF LR, augmentation-trained) — 10 correct + 10
incorrect, balanced over classes and error directions (5 false positives,
5 false negatives). `explanations_summary.json` holds top ±10 features per
example (weights toward the *hate* class); per-example HTML files are
interactive. Example: a correct hate prediction driven by "মরল"
(die/kill stem) with +0.10 weight.

## Reproduce

```bash
python -m src.data.download
python -m src.data.preprocess
python -m src.data.augment
python -m src.models.baselines --model all
python -m src.models.baselines --model all --train-file data/processed/train_augmented.csv
python -m src.evaluation.evaluate --all --skip-existing
python -m src.explainability.explain --model-dir outputs/models/tfidf_lr_aug
```
