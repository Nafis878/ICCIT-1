"""Slice-based error analysis from prediction dumps.

For each configuration (model, variant; seed-pooled majority vote) and
test set, reports macro-F1 / accuracy on interpretable slices:
- clean:           hate_type, target, length bucket, has-Latin-chars,
                   near-duplicate-of-train vs not
- bidwesh_heldout: dialect, hate_type, target, length bucket

Output: outputs/results/slice_analysis.csv

Usage: python -m src.evaluation.slices [--models banglabert,tfidf_lr]
"""
import argparse

import numpy as np
import pandas as pd

from src.evaluation.evaluate import load_test_set
from src.evaluation.stats import load_dumps, macro_f1
from src.utils.common import RESULTS_DIR, setup_utf8_stdout


def add_slice_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_words = df["text"].fillna("").str.split().str.len()
    df["len_bucket"] = pd.cut(n_words, bins=[0, 5, 15, 30, 10_000],
                              labels=["1-5w", "6-15w", "16-30w", ">30w"])
    df["has_latin"] = df["text"].fillna("").str.contains(r"[a-zA-Z]")
    # first (primary) hate type / target for hate rows
    df["hate_type_1"] = df["hate_type"].fillna("").str.split("_").str[0]
    df["target_1"] = df["target"].fillna("").str.split("_").str[0]
    return df


SLICES = {
    "clean": ["len_bucket", "has_latin", "hate_type_1", "target_1",
              "near_dup_train"],
    "bidwesh_heldout": ["dialect", "len_bucket", "hate_type_1", "target_1"],
}


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None,
                        help="comma-separated model filter")
    args = parser.parse_args()

    index = load_dumps()
    if args.models:
        keep = set(args.models.split(","))
        index = index[index["model"].isin(keep)]

    rows = []
    for test_name, slice_cols in SLICES.items():
        base = add_slice_columns(load_test_set(test_name))
        for (model, variant), grp in index[index["testset"] == test_name] \
                .groupby(["model", "variant"]):
            preds = [pd.read_csv(r["path"]).set_index("row_id")["y_pred"]
                     for _, r in grp.iterrows()]
            vote = (pd.concat(preds, axis=1).mean(axis=1) >= 0.5).astype(int)
            df = base.join(vote.rename("y_pred"), how="inner")
            for col in slice_cols:
                if col not in df.columns:
                    continue
                for value, sub in df.groupby(col, observed=True):
                    if len(sub) < 30 or str(value) == "":
                        continue
                    rows.append({
                        "model": model, "train_variant": variant,
                        "test_set": test_name, "slice": col,
                        "value": str(value), "n": len(sub),
                        "accuracy": round(
                            float((sub["y_pred"] == sub["label"]).mean()), 4),
                        "f1_macro": round(
                            macro_f1(sub["label"].values,
                                     sub["y_pred"].values), 4),
                    })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "slice_analysis.csv", index=False)
    print(f"{len(out)} slice rows -> {RESULTS_DIR / 'slice_analysis.csv'}")
    if len(out):
        worst = (out[out["slice"].isin(["dialect", "hate_type_1"])]
                 .sort_values("f1_macro").head(12))
        print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
