"""Classical baselines: majority class, TF-IDF + Logistic Regression,
TF-IDF + Linear SVM (calibrated so it exposes predict_proba for LIME).

Each trained model is saved to outputs/models/<model_id>[_aug][_smoke]/ and
immediately evaluated on the clean, synthetic-noisy and BIDWESH dialect
test sets via src.evaluation.evaluate.

Usage:
    python -m src.models.baselines --model all
    python -m src.models.baselines --model tfidf_lr --train-file data/processed/train_augmented.csv
    python -m src.models.baselines --model all --smoke
"""
import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.evaluation.evaluate import evaluate_saved_model
from src.utils.common import (
    DATA_PROCESSED,
    DEFAULT_SEED,
    MODELS_DIR,
    ensure_dirs,
    save_json,
    set_seed,
    setup_utf8_stdout,
)

INPUT_COLUMN = "text_clean"


def make_features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                 min_df=2, max_features=200_000,
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=2, max_features=300_000,
                                 sublinear_tf=True)),
    ])


def build_model(model_id: str, seed: int) -> Pipeline:
    if model_id == "majority":
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    if model_id == "tfidf_lr":
        clf = LogisticRegression(max_iter=2000, C=4.0,
                                 class_weight="balanced", random_state=seed)
    elif model_id == "tfidf_svm":
        svm = LinearSVC(C=1.0, class_weight="balanced", random_state=seed)
        clf = CalibratedClassifierCV(svm, cv=3)  # adds predict_proba
    else:
        raise ValueError(model_id)
    return Pipeline([("features", make_features()), ("clf", clf)])


def train_one(model_id: str, train_df: pd.DataFrame, train_variant: str,
              tag: str, seed: int) -> Path:
    # Smoke runs get a distinct id so they never overwrite full-run rows
    # in results_summary.csv.
    full_id = model_id + ("_smoke" if tag == "smoke" else "")
    out_dir = MODELS_DIR / f"{full_id}_{train_variant}_s{seed}"
    ensure_dirs(out_dir)

    texts = train_df[INPUT_COLUMN].fillna("").tolist()
    labels = train_df["label"].values
    model = build_model(model_id, seed)
    t0 = time.time()
    model.fit(texts, labels)
    train_secs = time.time() - t0
    print(f"  trained {model_id} on {len(texts)} rows "
          f"({train_variant}) in {train_secs:.1f}s")

    joblib.dump(model, out_dir / "model.joblib")
    save_json(
        {
            "model_id": full_id,
            "model_type": "sklearn",
            "input_column": INPUT_COLUMN,
            "train_variant": train_variant,
            "train_rows": int(len(texts)),
            "train_seconds": round(train_secs, 1),
            "tag": tag,
            "seed": seed,
        },
        out_dir / "meta.json",
    )
    return out_dir


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all",
                        choices=["majority", "tfidf_lr", "tfidf_svm", "all"])
    parser.add_argument("--train-file", type=Path,
                        default=DATA_PROCESSED / "train.csv",
                        help="swap in train_augmented.csv for the "
                             "robustness-trained variant")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny subsample, quick end-to-end check")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    set_seed(args.seed)

    from src.utils.common import VARIANT_OF_FILE

    train_df = pd.read_csv(args.train_file)
    train_variant = VARIANT_OF_FILE.get(args.train_file.name,
                                        args.train_file.stem)
    tag = "smoke" if args.smoke else "full"
    if args.smoke:
        train_df = train_df.sample(n=min(500, len(train_df)),
                                   random_state=args.seed)

    models = (["majority", "tfidf_lr", "tfidf_svm"]
              if args.model == "all" else [args.model])
    for model_id in models:
        print(f"[{model_id}] train_variant={train_variant} tag={tag}")
        out_dir = train_one(model_id, train_df, train_variant, tag, args.seed)
        evaluate_saved_model(out_dir)


if __name__ == "__main__":
    main()
