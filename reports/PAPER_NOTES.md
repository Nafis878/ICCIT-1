# Paper notes — claims → evidence map (Q1 package)

Status: **complete except one job** (Qwen QLoRA row; 47/48 queue jobs
done). All numbers below are final, from `outputs/results/`.
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

## Supporting claims (final numbers)

- C1 Dialect gap (clean → heldout drop, clean-trained): LR −4.98 pts,
  SVM −5.67, XLM-R −4.24, MuRIL −4.48, BanglaBERT −4.35 — every family
  loses 4–6 pts to real dialects.
- C2 Synthetic noise is a weak proxy: clean→noisy drops (1.9–2.5 pts) are
  ~half the real-dialect drops for every model.
- C3 Pretraining alone doesn't close the gap: relative drops nearly
  identical for TF-IDF and all encoders (C1); ranking stable across
  conditions.
- C4 Proposed methods: DIA closes 34% (BanglaBERT, +1.58 pts, p=2.3e-5)
  to 48% (LR, +2.41 pts, p=3e-11) of the dialect gap at zero clean cost;
  NCT is best on synthetic noise (0.9138) but n.s. on real dialects —
  clean dissociation. Full-split adaptation (3.6K rows) on top of aug:
  best dialect (0.9010) AND best clean (0.9317) — no forgetting;
  N≤500 adaptation is flat (data-efficiency curve).
- C5 LLM comparison: Qwen2.5-7B 5-shot = 0.703 clean / 0.689 dialect
  macro-F1 — ~22 pts below fine-tuned encoders. (QLoRA fine-tuned row
  pending: job `llm_qwen25_7b_qlora`, one short GPU session.) Scope
  note: mBERT dropped (MuRIL is the stronger generic-multilingual
  representative for Bangla).
- C6 Faithfulness (200 examples; comprehensiveness↑ / sufficiency↓ /
  deletion-AUC↓): TF-IDF LR — LIME 0.341 / −0.095 / 0.571, SHAP 0.327 /
  −0.094 / 0.581, random 0.053 / 0.209 / 0.766. BanglaBERT(aug_dia) —
  LIME 0.375 / −0.019 / 0.615, SHAP 0.364 / −0.020 / 0.623, random
  0.044 / 0.328 / 0.827. LIME ≥ SHAP on both models; both far above
  random → explanations are quantitatively faithful.
- C7 Leakage rigor: near-dups at cosine ≥0.9 — 188 BIDWESH rows (vs 28
  exact) excluded from the benchmark; 2.1% of BD-SHS test flagged vs
  train (slice `near_dup_train` in slice_analysis.csv).

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
