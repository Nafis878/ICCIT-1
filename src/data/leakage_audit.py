"""Near-duplicate leakage audit (char-3-gram TF-IDF cosine similarity).

Checks two contamination channels that exact-match misses:
1. BIDWESH source sentences vs BD-SHS train+val  -> flags rows in
   bidwesh_test.csv (column ``near_dup_bdshs_train``). Evaluation excludes
   flagged rows from the held-out dialect benchmark.
2. BD-SHS test vs BD-SHS train+val               -> flags rows in
   test.csv (column ``near_dup_train``). The official split is kept for
   comparability; slice analysis reports scores with/without the flag.

Writes reports/leakage_audit.md with counts at several thresholds.

Usage: python -m src.data.leakage_audit [--threshold 0.9]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.common import DATA_PROCESSED, REPORTS_DIR, setup_utf8_stdout

THRESHOLDS = (0.80, 0.85, 0.90, 0.95)


def max_similarity(queries: list[str], corpus: list[str]) -> np.ndarray:
    """Max cosine similarity of each query against the corpus
    (char 3-gram TF-IDF), computed in query chunks to bound memory."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1)
    corpus_m = vec.fit_transform(corpus)
    out = np.zeros(len(queries))
    chunk = 500
    for i in range(0, len(queries), chunk):
        q = vec.transform(queries[i:i + chunk])
        sims = cosine_similarity(q, corpus_m, dense_output=False)
        out[i:i + chunk] = sims.max(axis=1).toarray().ravel()
    return out


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    val = pd.read_csv(DATA_PROCESSED / "val.csv")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    bidwesh = pd.read_csv(DATA_PROCESSED / "bidwesh_test.csv")
    corpus = train["text"].fillna("").tolist() + val["text"].fillna("").tolist()

    lines = ["# Near-duplicate leakage audit", "",
             "Char-3-gram TF-IDF cosine similarity, max over BD-SHS "
             "train+val.", ""]

    print("Auditing BIDWESH source sentences vs BD-SHS train/val ...")
    src_texts = bidwesh["standard_bangla"].fillna("").unique().tolist()
    src_sims = pd.Series(max_similarity(src_texts, corpus), index=src_texts)
    bidwesh["near_dup_bdshs_train"] = (
        bidwesh["standard_bangla"].map(src_sims) >= args.threshold)
    lines.append("## BIDWESH sources vs BD-SHS train+val")
    lines.append(f"{len(src_texts)} unique sources; exact matches previously "
                 f"found: {int(bidwesh['overlaps_bdshs_train'].sum())} rows.")
    for t in THRESHOLDS:
        n_src = int((src_sims >= t).sum())
        n_rows = int((bidwesh["standard_bangla"].map(src_sims) >= t).sum())
        in_test = int(((bidwesh["standard_bangla"].map(src_sims) >= t)
                       & (bidwesh["bidwesh_split"] == "test")).sum())
        lines.append(f"- cosine >= {t:.2f}: {n_src} sources / {n_rows} rows "
                     f"({in_test} in the held-out test half)")
        print(lines[-1])
    n_flag = int((bidwesh["near_dup_bdshs_train"]
                  & (bidwesh["bidwesh_split"] == "test")).sum())
    lines.append(f"\nFlag applied at {args.threshold:.2f}; evaluation excludes "
                 f"{n_flag} flagged rows from bidwesh_heldout.")
    bidwesh.to_csv(DATA_PROCESSED / "bidwesh_test.csv", index=False,
                   encoding="utf-8")

    print("Auditing BD-SHS test vs train/val ...")
    test_sims = max_similarity(test["text"].fillna("").tolist(), corpus)
    test["near_dup_train"] = test_sims >= args.threshold
    lines.append("\n## BD-SHS test vs train+val (official split kept)")
    for t in THRESHOLDS:
        n = int((test_sims >= t).sum())
        lines.append(f"- cosine >= {t:.2f}: {n} test rows "
                     f"({n / len(test):.1%})")
        print(lines[-1])
    lines.append(f"\nFlag column `near_dup_train` written at "
                 f"{args.threshold:.2f}; official test kept intact for "
                 f"comparability, slice analysis reports both subsets.")
    test.to_csv(DATA_PROCESSED / "test.csv", index=False, encoding="utf-8")

    with open(REPORTS_DIR / "leakage_audit.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {REPORTS_DIR / 'leakage_audit.md'}")


if __name__ == "__main__":
    main()
