# Paper notes — claims → evidence map (Q1 package)

Status: skeleton — numbers filled in as experiment phases complete.
Protocol v2: BIDWESH grouped 40/10/50 adapt/dev/heldout split, near-dup
leakage exclusion, 3 seeds (42/43/44), bootstrap CIs + McNemar/paired
bootstrap, prediction dumps in `outputs/predictions/`.

## Proposed contributions

1. **DIA — Dialect-Informed Augmentation**: standard→dialect lexicon
   (601 entries) mined automatically from BIDWESH adapt-split parallel
   sentences (`src/data/dialect_lexicon.py`), applied as training-time
   augmentation. Evidence: `stats_summary.csv` rows `*:dialect` and
   `*:aug_dia` vs `*:clean`/`*:augmented`; significance rows in
   `significance_tests.csv`.
2. **NCT — Noise-Consistency Training**: symmetric-KL consistency between
   clean and noised views (`src/models/consistency_trainer.py`). Evidence:
   `banglabert-nct:clean` vs `banglabert:augmented` (same augmentation
   knowledge, different objective).
3. **Few-shot dialect adaptation curve**: N ∈ {250, 500, 1000, ~3.6K}
   BIDWESH adapt examples → macro-F1 on bidwesh_heldout + clean
   (forgetting). Evidence: `banglabert-adapt*` rows; figure
   `figures/adaptation_curve.png` (Phase C).

## Supporting claims (fill in)

- C1 Dialect gap: clean → bidwesh_heldout drop per model family (mean±std).
- C2 Synthetic noise is a weak proxy: drop(augmented) vs drop(bidwesh).
- C3 Pretraining alone doesn't close the gap: relative drops similar
  across TF-IDF/mBERT/MuRIL/XLM-R/BanglaBERT.
- C4 DIA/NCT close X% of the gap at zero clean cost; adaptation with
  N=250 closes Y%.
- C5 LLM few-shot (Qwen2.5-7B) underperforms fine-tuned models by Z pts
  (or not — report either way).
- C6 Faithfulness: LIME vs SHAP vs random comprehensiveness/sufficiency/
  deletion-AUC for TF-IDF LR + BanglaBERT.
- C7 Leakage rigor: near-dup rates (BIDWESH↔train 188 rows @0.9;
  BD-SHS test↔train 2.1% @0.9); results insensitive (slice analysis).

## Tables planned

- T1 main results: model × {clean, augmented, dialect, aug_dia, nct} ×
  {clean, synthetic-noisy, bidwesh_heldout} mean±std + CI.
- T2 per-dialect breakdown (chittagong/noakhali/barishal).
- T3 significance matrix (key claims).
- T4 adaptation curve values.
- T5 faithfulness metrics.
- T6 slice analysis highlights (hate_type, target, length, latin).

## Figures planned

- F1 pipeline/method diagram (manuscript stage).
- F2 robustness bars with CI whiskers per model family.
- F3 adaptation data-efficiency curve (3-seed band).
- F4 faithfulness deletion curves.
- F5 confusion matrices (reference seed).

## Limitations to state

- BIDWESH covers 3 of many Bangla dialect regions; translated (not
  in-the-wild) dialect text.
- Synthetic augmentation is rule-based; lexicon mined from the same
  distribution used for adaptation experiments (but disjoint sentences).
- Single dataset family for training (BD-SHS); 30K corpus available for
  cross-dataset generalization if a reviewer asks.
- LLM baseline evaluated on subsets (cost); subset row-ids stored for
  exact reproduction.
- Human plausibility of explanations: rating sheet provided
  (`outputs/explainability/human_eval_sheet.csv`, optional native-speaker
  annotation) — faithfulness metrics are the primary XAI evidence.

## Target venues (Q1)

Information Processing & Management; Computer Speech & Language; IEEE
Access (fallback, fast); Language Resources and Evaluation (resource
angle). Extended-conference-paper policies allow the ICCIT version.
