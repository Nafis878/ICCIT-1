# Results — protocol v2 (Q1 package)

Updated 2026-07-06. Binary Bangla hate speech detection (BD-SHS Task A,
official splits). **Protocol v2**: BIDWESH grouped 40/10/50
adapt/dev/heldout split by source sentence; held-out dialect benchmark
`bidwesh_heldout` (n=4,408 after excluding char-3-gram near-duplicates of
BD-SHS train/val, threshold 0.9 — `reports/leakage_audit.md`); 3 seeds per
configuration; per-example prediction dumps; bootstrap 95% CIs; paired
bootstrap + McNemar significance. v1 single-seed results archived in
`outputs/results/archive_v1/`.

Tables: `outputs/results/stats_summary.csv` (mean±std + CI),
`significance_tests.csv`, `robustness_summary.csv` (per-seed drops),
`slice_analysis.csv`. Figures: `outputs/figures/robustness_bars_*.png`,
`faithfulness_bars.png` (+ `adaptation_curve.png` after GPU phase).

## Methods under test

- **synthetic aug** — rule-based social-media noise (train_augmented.csv)
- **DIA** (proposed) — Dialect-Informed Augmentation via a 601-entry
  standard→dialect lexicon mined from BIDWESH adapt-split parallel
  sentences (train_dialect_aug.csv)
- **synthetic+DIA** — both (train_aug_dia.csv)
- **NCT** (proposed) — noise-consistency training (transformers only;
  GPU phase)
- **few-shot adaptation** (proposed) — continued fine-tuning on N BIDWESH
  adapt rows (transformers only; GPU phase)

## Classical results (complete, local; macro-F1, 3 seeds, 95% bootstrap CI)

| model | train variant | clean test | synthetic-noisy | BIDWESH held-out |
|---|---|---|---|---|
| majority | — | 0.342 | 0.342 | 0.336 |
| TF-IDF LR | clean | 0.9036 [.895–.912] | 0.8796 | 0.8538 [.843–.864] |
| TF-IDF LR | synthetic aug | 0.9050 [.897–.913] | 0.8926 | 0.8631 [.853–.873] |
| TF-IDF LR | **DIA** | 0.9039 [.895–.912] | 0.8835 | **0.8779 [.869–.888]** |
| TF-IDF LR | synthetic+DIA | 0.9047 [.896–.913] | 0.8892 | **0.8782 [.868–.888]** |
| TF-IDF SVM | clean | 0.9014 [.893–.909] | 0.8771 | 0.8447 [.834–.856] |
| TF-IDF SVM | synthetic aug | 0.9010 [.893–.909] | 0.8816 | 0.8510 [.841–.861] |
| TF-IDF SVM | DIA | 0.9004 [.892–.909] | 0.8781 | 0.8614 [.851–.872] |
| TF-IDF SVM | synthetic+DIA | 0.8996 [.891–.908] | 0.8874 | 0.8675 [.857–.878] |

(Classical seed-to-seed std ≈ 0 — LR/SVM training is deterministic given
the data; uncertainty is carried by the example-level bootstrap CIs.)

### Significance (paired bootstrap + McNemar, seed-pooled)

| comparison (on BIDWESH held-out) | ΔF1 | p (bootstrap) | p (McNemar) |
|---|---|---|---|
| LR synthetic-aug vs clean | +0.0093 | <0.001 | 9.1e-4 |
| LR **DIA vs clean** | **+0.0241** | <0.001 | 3.1e-11 |
| LR **synthetic+DIA vs clean** | **+0.0244** | <0.001 | 3.8e-11 |

## Key findings so far

1. **DIA (proposed) beats synthetic augmentation on real dialects by
   ~2.6×** for LR: +2.41 pts vs +0.93 pts macro-F1 on the held-out dialect
   benchmark, both highly significant, at zero clean-test cost
   (0.9039 vs 0.9036). Same ordering for SVM (+1.7 vs +0.6 pts).
2. **DIA recovers about half of the dialect gap**: LR's clean→dialect drop
   shrinks from 4.98 pts (clean-trained) to 2.60 pts (DIA-trained).
3. Synthetic noise mainly helps the synthetic-noisy test (as expected —
   it matches that distribution); DIA transfers to real dialect text
   because its substitutions come from real dialect parallel data.
4. **Explanations are faithful, quantitatively**: on 200 test examples,
   LIME comprehensiveness 0.341 / sufficiency −0.095 / deletion-AUC 0.571
   vs random-attribution 0.053 / 0.209 / 0.766 (LIME ≥ SHAP on all three;
   `faithfulness_summary.csv`). LIME/SHAP now use whitespace tokenization —
   python's `\W` splits Bangla combining vowel signs and corrupts
   attributions (fixed; the v1 single-character artifact is gone).
5. Leakage rigor: near-duplicate audit found 188 BIDWESH rows (vs 28
   exact-match) and 2.1% of BD-SHS test near-duplicated in train at
   cosine ≥0.9; the dialect benchmark excludes them, and slice analysis
   covers the test-set flag.

## Transformer phase (GPU queue — in progress)

49 jobs in `scripts/experiments_manifest.json`: BanglaBERT ×
{clean, synthetic aug, DIA, synthetic+DIA, NCT} × 3 seeds, few-shot
adaptation curve (N ∈ {250, 500, 1000, full} × 3 seeds), XLM-R (3 variants
× 3 seeds), MuRIL (2 × 3), mBERT (2 × 1), Qwen2.5-7B few-shot baseline,
transformer faithfulness (LIME/SHAP/random). Run via
`scripts/colab_runner.py` (resume-safe; see README). Analysis after each
returned batch: `evaluate --all --skip-existing && stats && slices &&
figures`.

## Reproduce (local part)

```bash
python -m src.data.download && python -m src.data.preprocess
python -m src.data.leakage_audit && python -m src.data.dialect_lexicon
python -m src.data.augment
for s in 42 43 44; do for f in train train_augmented train_dialect_aug train_aug_dia; do
  python -m src.models.baselines --model tfidf_lr  --train-file data/processed/$f.csv --seed $s
  python -m src.models.baselines --model tfidf_svm --train-file data/processed/$f.csv --seed $s
done; done
python -m src.models.baselines --model majority
python -m src.evaluation.stats && python -m src.evaluation.slices && python -m src.evaluation.figures
python -m src.explainability.faithfulness --model-dir outputs/models/tfidf_lr_aug_dia_s42 --method lime
python -m src.explainability.explain --model-dir outputs/models/tfidf_lr_aug_dia_s42
```
