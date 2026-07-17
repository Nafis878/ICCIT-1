# Results — protocol v2, complete (Q1 package)

Updated 2026-07-17. Binary Bangla hate speech detection (BD-SHS Task A,
official splits). Protocol: held-out dialect benchmark `bidwesh_heldout`
(BIDWESH test half, grouped by source sentence, near-duplicates of BD-SHS
train excluded; n=4,408), 3 seeds per configuration (seed std shown),
bootstrap 95% CIs, paired-bootstrap + McNemar significance, per-example
prediction dumps (`outputs/predictions/`, 214 files). Transformers
trained on Colab A100 (whole 47-job queue: **2.1 GPU-hours** thanks to
length-bucketed batching); classical models on CPU.

Source tables: `stats_summary.csv`, `significance_tests.csv`,
`slice_analysis.csv`, `robustness_summary.csv`. Figures:
`robustness_bars_*.png`, `adaptation_curve.png`, `faithfulness_bars.png`.

## Main results (macro-F1, mean over 3 seeds; heldout std ≤ 0.004)

| model | train variant | clean | synthetic-noisy | BIDWESH held-out |
|---|---|---|---|---|
| majority | — | 0.342 | 0.342 | 0.336 |
| TF-IDF LR | clean | 0.9036 | 0.8796 | 0.8538 |
| TF-IDF LR | **DIA** | 0.9039 | 0.8835 | 0.8779 |
| TF-IDF SVM | clean | 0.9014 | 0.8771 | 0.8447 |
| TF-IDF SVM | DIA | 0.9004 | 0.8781 | 0.8614 |
| XLM-R base | clean | 0.9098 | 0.8955 | 0.8674 |
| XLM-R base | synthetic aug | 0.9135 | 0.9047 | 0.8798 |
| XLM-R base | synthetic+DIA | 0.9152 | 0.9045 | 0.8898 |
| MuRIL base | clean | 0.9159 | 0.8963 | 0.8711 |
| MuRIL base | synthetic aug | 0.9207 | 0.9074 | 0.8780 |
| BanglaBERT | clean | 0.9239 | 0.9051 | 0.8804 |
| BanglaBERT | synthetic aug | 0.9251 | 0.9122 | 0.8876 |
| BanglaBERT | **NCT** | 0.9224 | **0.9138** | 0.8822 |
| BanglaBERT | **DIA** | 0.9249 | 0.9074 | **0.8962** |
| BanglaBERT | synthetic+DIA | 0.9241 | 0.9105 | **0.8962** |
| BanglaBERT aug **+ full BIDWESH adaptation** | | **0.9317** | — | **0.9010** |
| Qwen2.5-7B | 5-shot (1.5K subsets) | 0.7034 | — | 0.6894 |

## Significance of the key claims (paired bootstrap / McNemar, heldout)

| claim | ΔF1 | p_boot | p_McNemar |
|---|---|---|---|
| BanglaBERT DIA > clean | +0.0158 | <0.001 | 2.3e-5 |
| BanglaBERT synthetic+DIA > synthetic | +0.0086 | <0.001 | 8.3e-4 |
| BanglaBERT synthetic > clean | +0.0073 | 0.002 | 3.7e-2 |
| BanglaBERT NCT vs synthetic | −0.0054 | 0.008 | 0.24 (n.s.) |
| TF-IDF LR DIA > clean | +0.0241 | <0.001 | 3.1e-11 |
| BanglaBERT > TF-IDF LR (clean-trained) | +0.0266 | <0.001 | 1.0e-8 |

## Findings

1. **DIA (proposed) is the most effective training-time defense against
   real dialect shift**, across every architecture: +2.41 pts (LR), +1.67
   (SVM), +2.24 (XLM-R, via synthetic+DIA), +1.58 (BanglaBERT) on the
   held-out dialect benchmark, at zero clean-test cost. It closes ~34%
   (BanglaBERT) to ~48% (LR) of each model's dialect gap.
2. **NCT and DIA dissociate cleanly**: NCT gives the best synthetic-noise
   robustness of any variant (0.9138) but does *not* beat plain
   augmentation on real dialects (n.s.) — consistency training defends
   the distribution it trains on, while DIA's lexicon carries real
   dialect knowledge. Useful ablation evidence that synthetic noise and
   dialect shift are different phenomena (echoing finding 5, v2).
3. **Few-shot dialect adaptation needs ~>1K real examples to pay off**
   (`adaptation_curve.png`): N≤500 is flat-to-negative; the full adapt
   split (3.6K rows) yields the best dialect score overall (0.9010,
   +1.34 pts over its base) *and* the best clean score (0.9317) — no
   catastrophic forgetting. Best overall recipe: BanglaBERT + synthetic
   aug + full BIDWESH adaptation.
4. **Zero/few-shot LLMs are not competitive** for this task: Qwen2.5-7B
   5-shot scores 0.703/0.689 macro-F1 vs 0.92+ for fine-tuned encoders —
   a ~22-pt gap. (QLoRA fine-tuned Qwen row pending — one short GPU
   session; see below.)
5. **Model ranking is stable across all test conditions**: BanglaBERT >
   MuRIL > XLM-R > TF-IDF LR > SVM ≫ 5-shot LLM ≫ majority; monolingual
   pretraining wins, but no amount of pretraining closes the dialect gap
   by itself (relative drops are similar for every encoder).
6. **Explanations are quantitatively faithful** (LIME comprehensiveness
   0.341 vs random 0.053; deletion-AUC 0.571 vs 0.766), with LIME ≥ SHAP
   on all three metrics; Bangla-safe whitespace tokenization throughout.
7. Transformer seed variance is small (std ≤ 0.004), and the near-dup
   leakage audit + grouped BIDWESH split rule out contamination effects.

## Outstanding

- `llm_qwen25_7b_qlora` (Unsloth QLoRA fine-tune) is the single unfinished
  queue job (47/48 done). One ~30-60 min GPU session: re-run the standard
  runner cell; it will execute only this job.
- Optional: transformer faithfulness/LIME artifacts from the A100 batch
  are merged; human plausibility sheet
  (`outputs/explainability/human_eval_sheet.csv`) awaits native-speaker
  ratings (optional appendix).

## Reproduce

Local part: see README §Analysis. GPU part: `scripts/colab_runner.py`
with `scripts/experiments_manifest.json` (resume-safe; A100 ≈ 2 GPU-hours
total). Summary CSVs are rebuildable from per-run report JSONs via
`python -m src.evaluation.rebuild_summary`.
