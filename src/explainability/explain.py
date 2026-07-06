"""Explain model predictions on the clean test set.

Selects 10 correct and 10 incorrect predictions (balanced over classes as
far as possible) and explains each with LIME (default) or SHAP. Works with
both saved model kinds (sklearn joblib pipeline / HF transformer dir) via
the predictors in src.evaluation.evaluate.

Outputs (in outputs/explainability/<model-dir-name>/):
- lime_<row>_<correct|wrong>.html    per-example interactive report
- explanations_summary.json          top +/- features per example
- README.md                          index of generated files

Usage:
    python -m src.explainability.explain --model-dir outputs/models/tfidf_lr
    python -m src.explainability.explain --model-dir outputs/models/tfidf_lr --method shap
"""
import argparse
import html as html_mod
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.evaluate import HFPredictor, SklearnPredictor, load_predictor
from src.utils.common import (
    DATA_PROCESSED,
    EXPLAIN_DIR,
    ensure_dirs,
    save_json,
    set_seed,
    setup_utf8_stdout,
)

LABEL_NAMES = ["not_hate", "hate"]


def make_proba_fn(predictor):
    """Return f(list[str]) -> ndarray[n, 2] of class probabilities."""
    if isinstance(predictor, SklearnPredictor):
        pipeline = predictor.pipeline

        def proba(texts):
            return pipeline.predict_proba(list(texts))

        return proba
    if isinstance(predictor, HFPredictor):
        torch = predictor.torch

        def proba(texts):
            texts = list(texts)
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), predictor.batch_size):
                    enc = predictor.tokenizer(
                        texts[i:i + predictor.batch_size], truncation=True,
                        max_length=predictor.max_len, padding=True,
                        return_tensors="pt").to(predictor.device)
                    logits = predictor.model(**enc).logits
                    out.append(torch.softmax(logits, dim=-1).cpu().numpy())
            return np.concatenate(out)

        return proba
    raise TypeError(type(predictor))


def pick_examples(df: pd.DataFrame, preds: np.ndarray, n_each: int,
                  seed: int) -> pd.DataFrame:
    """n_each correct + n_each incorrect rows, balanced over true labels."""
    rng = np.random.RandomState(seed)
    df = df.assign(pred=preds, correct=(preds == df["label"].values))
    picked = []
    for correct in (True, False):
        pool = df[df["correct"] == correct]
        half = n_each // 2
        chosen_idx = []
        for lbl in (0, 1):
            sub = pool[pool["label"] == lbl]
            take = min(half, len(sub))
            chosen_idx += list(rng.choice(sub.index, size=take, replace=False))
        # top up if a class had too few examples
        remaining = pool.index.difference(chosen_idx)
        if len(chosen_idx) < n_each and len(remaining) > 0:
            extra = rng.choice(remaining,
                               size=min(n_each - len(chosen_idx),
                                        len(remaining)), replace=False)
            chosen_idx += list(extra)
        picked.append(df.loc[chosen_idx])
    return pd.concat(picked)


def explain_lime(texts_col: str, row: pd.Series, proba_fn, num_features: int,
                 num_samples: int, seed: int):
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=LABEL_NAMES, random_state=seed)
    exp = explainer.explain_instance(
        row[texts_col], proba_fn, num_features=num_features,
        num_samples=num_samples, labels=(1,))
    weights = exp.as_list(label=1)  # feature weight toward class "hate"
    return weights, exp.as_html(labels=(1,))


def explain_shap(texts_col: str, row: pd.Series, proba_fn, num_features: int):
    import shap

    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(lambda x: proba_fn(x)[:, 1], masker,
                               silent=True)
    sv = explainer([row[texts_col]])
    tokens = [t.strip() for t in sv.data[0]]
    vals = sv.values[0]
    pairs = sorted(zip(tokens, vals), key=lambda p: abs(p[1]), reverse=True)
    weights = [(t, float(v)) for t, v in pairs if t][:num_features]
    body = "".join(
        f"<li><code>{html_mod.escape(t)}</code>: {v:+.4f}</li>"
        for t, v in weights)
    page = (f"<html><head><meta charset='utf-8'></head><body>"
            f"<h3>SHAP weights toward 'hate'</h3>"
            f"<p>{html_mod.escape(row[texts_col])}</p><ul>{body}</ul>"
            f"</body></html>")
    return weights, page


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--method", default="lime", choices=["lime", "shap"])
    parser.add_argument("--num-each", type=int, default=10,
                        help="number of correct AND incorrect examples")
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="LIME perturbation samples (lower = faster)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    # Per-model subdir so explanations for several models can coexist.
    out_dir = EXPLAIN_DIR / Path(args.model_dir).name
    ensure_dirs(out_dir)

    predictor, meta = load_predictor(args.model_dir)
    proba_fn = make_proba_fn(predictor)
    text_col = predictor.input_column

    df = pd.read_csv(DATA_PROCESSED / "test.csv")
    df[text_col] = df[text_col].fillna("")
    preds = predictor.predict(df)
    picked = pick_examples(df, preds, args.num_each, args.seed)
    print(f"explaining {len(picked)} examples "
          f"({int(picked['correct'].sum())} correct, "
          f"{int((~picked['correct']).sum())} incorrect) "
          f"with {args.method} for model {meta['model_id']}")

    summary = []
    for row_idx, row in picked.iterrows():
        kind = "correct" if row["correct"] else "wrong"
        if args.method == "lime":
            weights, page = explain_lime(text_col, row, proba_fn,
                                         args.num_features, args.num_samples,
                                         args.seed)
        else:
            weights, page = explain_shap(text_col, row, proba_fn,
                                         args.num_features)
        fname = f"{args.method}_{row_idx}_{kind}.html"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            f.write(page)
        proba = proba_fn([row[text_col]])[0]
        summary.append({
            "test_row": int(row_idx),
            "file": fname,
            "text": row["text"],
            "true_label": LABEL_NAMES[int(row["label"])],
            "predicted_label": LABEL_NAMES[int(row["pred"])],
            "p_hate": round(float(proba[1]), 4),
            "correct": bool(row["correct"]),
            "top_features_toward_hate": [
                {"token": t, "weight": round(float(w), 4)}
                for t, w in weights],
        })
        print(f"  {fname}: true={summary[-1]['true_label']} "
              f"pred={summary[-1]['predicted_label']} "
              f"p_hate={summary[-1]['p_hate']}")

    save_json(
        {"model": meta["model_id"],
         "train_variant": meta.get("train_variant", "clean"),
         "method": args.method,
         "note": ("weights are contributions toward the 'hate' class; "
                  "positive pushes toward hate, negative toward not_hate"),
         "examples": summary},
        out_dir / "explanations_summary.json",
    )

    index_lines = [
        f"# Explainability outputs — {meta['model_id']} ({args.method})", "",
        f"{len(summary)} examples from the clean test set "
        f"({args.num_each} correct + {args.num_each} incorrect predictions).",
        "Weights are toward the **hate** class.", "",
    ]
    for s in summary:
        mark = "OK " if s["correct"] else "MISS"
        index_lines.append(
            f"- [{mark}] [`{s['file']}`]({s['file']}) — true "
            f"{s['true_label']}, pred {s['predicted_label']} "
            f"(p_hate={s['p_hate']})")
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")
    print(f"wrote {out_dir} (HTML + explanations_summary.json + README.md)")


if __name__ == "__main__":
    main()
