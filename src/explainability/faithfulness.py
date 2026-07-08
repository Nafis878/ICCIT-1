"""Quantitative (ERASER-style) faithfulness evaluation of explanations.

For N stratified test examples and an attribution method (LIME or SHAP):
- comprehensiveness@k: p(class) drop after REMOVING the top-k attributed
  words (higher = explanation captured what mattered)
- sufficiency@k:       p(class) drop when KEEPING ONLY the top-k words
  (lower = the top words alone support the prediction)
- deletion AUC:        p(class) as top words are deleted one at a time,
  normalized area under the curve (lower = more faithful)

k defaults to 20% of the example's word count (>=1). Scores are computed
w.r.t. the model's predicted class. Random-attribution baseline included
for calibration.

Output: appends one row per (model, method) to
outputs/results/faithfulness_summary.csv and saves per-example scores.

Usage:
    python -m src.explainability.faithfulness --model-dir outputs/models/<dir> \
        --method lime [--num-examples 200] [--num-samples 500]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.evaluate import load_predictor, load_test_set
from src.explainability.explain import make_proba_fn
from src.utils.common import RESULTS_DIR, ensure_dirs, set_seed, setup_utf8_stdout

REMOVED = ""  # words are deleted outright (empty-string masking)


def attribution_weights(method: str, text: str, proba_fn, pred_class: int,
                        num_samples: int, seed: int) -> dict[str, float]:
    """Word -> weight toward the predicted class."""
    if method == "lime":
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(split_expression=r"\s+", bow=True,
                                      random_state=seed)
        exp = explainer.explain_instance(
            text, proba_fn, num_features=len(text.split()),
            num_samples=num_samples, labels=(pred_class,))
        return dict(exp.as_list(label=pred_class))
    if method == "shap":
        import shap

        masker = shap.maskers.Text(r"\s+")
        explainer = shap.Explainer(
            lambda x: proba_fn(x)[:, pred_class], masker, silent=True)
        sv = explainer([text])
        return {t.strip(): float(v)
                for t, v in zip(sv.data[0], sv.values[0]) if t.strip()}
    if method == "random":
        rng = np.random.RandomState(seed)
        words = text.split()
        return {w: float(r) for w, r in zip(words, rng.rand(len(words)))}
    raise ValueError(method)


def rank_words(text: str, weights: dict[str, float]) -> list[int]:
    """Word positions sorted by descending attribution."""
    words = text.split()
    scored = [(weights.get(w, 0.0), i) for i, w in enumerate(words)]
    return [i for _, i in sorted(scored, key=lambda x: -x[0])]


def drop_words(text: str, positions: set[int], keep: bool) -> str:
    words = text.split()
    if keep:
        kept = [w for i, w in enumerate(words) if i in positions]
    else:
        kept = [w for i, w in enumerate(words) if i not in positions]
    return " ".join(kept) if kept else "…"


def evaluate_example(text: str, proba_fn, method: str, num_samples: int,
                     seed: int, deletion_steps: int = 10) -> dict:
    p0 = proba_fn([text])[0]
    pred = int(p0.argmax())
    weights = attribution_weights(method, text, proba_fn, pred,
                                  num_samples, seed)
    order = rank_words(text, weights)
    n_words = len(text.split())
    k = max(1, round(0.2 * n_words))
    topk = set(order[:k])

    p_compr = proba_fn([drop_words(text, topk, keep=False)])[0][pred]
    p_suff = proba_fn([drop_words(text, topk, keep=True)])[0][pred]

    # deletion curve: remove top words cumulatively
    steps = np.linspace(0, min(n_words, 2 * k), deletion_steps + 1,
                        dtype=int)[1:]
    texts = [drop_words(text, set(order[:s]), keep=False) for s in steps]
    curve = proba_fn(texts)[:, pred]
    del_auc = float(np.trapz(np.concatenate([[p0[pred]], curve]))
                    / (len(curve) + 1))

    return {
        "pred_class": pred,
        "p_orig": float(p0[pred]),
        "comprehensiveness": float(p0[pred] - p_compr),
        "sufficiency": float(p0[pred] - p_suff),
        "deletion_auc": del_auc,
        "n_words": n_words,
        "k": k,
    }


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--method", default="lime",
                        choices=["lime", "shap", "random"])
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=500,
                        help="LIME perturbation samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    predictor, meta = load_predictor(args.model_dir)
    proba_fn = make_proba_fn(predictor)
    text_col = predictor.input_column

    df = load_test_set("clean")
    df[text_col] = df[text_col].fillna("")
    df = df[df[text_col].str.split().str.len() >= 5]  # need enough words
    sample = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(
                    n=min(len(g), args.num_examples // 2),
                    random_state=args.seed)))
    print(f"faithfulness: {meta['model_id']} x {args.method} on "
          f"{len(sample)} examples")

    records = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        try:
            records.append(evaluate_example(
                row[text_col], proba_fn, args.method, args.num_samples,
                args.seed))
        except Exception as e:  # noqa: BLE001 - skip pathological examples
            print(f"  skip example ({e})")
        if i % 25 == 0:
            print(f"  {i}/{len(sample)}")
    res = pd.DataFrame(records)

    ensure_dirs(RESULTS_DIR / "faithfulness")
    detail_path = (RESULTS_DIR / "faithfulness" /
                   f"{meta['model_id']}_{meta.get('train_variant', 'clean')}"
                   f"_{args.method}.csv")
    res.to_csv(detail_path, index=False)

    row = {
        "model": meta["model_id"],
        "train_variant": meta.get("train_variant", "clean"),
        "method": args.method,
        "n_examples": len(res),
        "comprehensiveness_mean": round(res["comprehensiveness"].mean(), 4),
        "sufficiency_mean": round(res["sufficiency"].mean(), 4),
        "deletion_auc_mean": round(res["deletion_auc"].mean(), 4),
        "num_samples": args.num_samples,
    }
    summary_path = RESULTS_DIR / "faithfulness_summary.csv"
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        key = ((s["model"] == row["model"])
               & (s["train_variant"] == row["train_variant"])
               & (s["method"] == row["method"]))
        s = s[~key]
        s = pd.concat([s, pd.DataFrame([row])], ignore_index=True)
    else:
        s = pd.DataFrame([row])
    s.to_csv(summary_path, index=False)
    print({k: v for k, v in row.items() if k not in ("model",)})
    print(f"wrote {detail_path} and faithfulness_summary.csv")


if __name__ == "__main__":
    main()
