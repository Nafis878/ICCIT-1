"""Standardize raw datasets into data/processed/.

Standard schema: ``text`` (normalized, transformer-ready), ``text_clean``
(aggressively cleaned, for TF-IDF), ``label`` (0 = not_hate, 1 = hate),
``target`` / ``hate_type`` (optional, BD-SHS Task B/C strings),
``source_dataset`` and, for BIDWESH, ``dialect`` + ``overlaps_bdshs_train``.

Outputs
-------
- train.csv / val.csv / test.csv           BD-SHS official splits, cleaned
- bidwesh_test.csv                          9,183 dialect rows (eval only)
- extra_bengali_hs_30k.csv                  standardized, unused by default
- label_mapping.json                        label id <-> name maps
- reports/data_stats.md                     row counts + label distributions

Usage: python -m src.data.preprocess
"""
import pandas as pd

from src.utils.common import (
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_SEED,
    REPORTS_DIR,
    ensure_dirs,
    read_csv_any,
    save_json,
    set_seed,
    setup_utf8_stdout,
)
from src.utils.normalize import clean_for_tfidf, normalize_bangla

LABEL_NAMES = {0: "not_hate", 1: "hate"}
TARGET_VOCAB = ["ind", "male", "female", "group"]
TYPE_VOCAB = ["slander", "gender", "religion", "callToViolence"]


def _standardize(df: pd.DataFrame, text_col: str, label_col: str, source: str,
                 target_col: str | None = None, type_col: str | None = None,
                 extra: dict | None = None) -> pd.DataFrame:
    out = pd.DataFrame()
    out["text"] = df[text_col].map(normalize_bangla)
    out["label"] = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
    out["target"] = df[target_col] if target_col else pd.NA
    out["hate_type"] = df[type_col] if type_col else pd.NA
    out["source_dataset"] = source
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


def _clean(df: pd.DataFrame, name: str, dedup: bool = True) -> pd.DataFrame:
    n0 = len(df)
    df = df[df["label"].isin([0, 1])]
    df = df[df["text"].str.len() > 0]
    n_empty = n0 - len(df)
    n_dup = 0
    if dedup:
        before = len(df)
        df = df.drop_duplicates(subset=["text"], keep="first")
        n_dup = before - len(df)
    df = df.reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    print(f"  {name}: {n0} -> {len(df)} rows "
          f"(dropped {n_empty} empty/invalid, {n_dup} duplicates)")
    return df


def _add_text_clean(df: pd.DataFrame) -> pd.DataFrame:
    df["text_clean"] = df["text"].map(clean_for_tfidf)
    # TF-IDF text may become empty (e.g. emoji-only comments); keep the row —
    # vectorizers handle empty strings — but count them.
    n_empty_clean = int((df["text_clean"].str.len() == 0).sum())
    if n_empty_clean:
        print(f"    note: {n_empty_clean} rows have empty text_clean (emoji/punct-only)")
    return df


def process_bdshs() -> dict[str, pd.DataFrame]:
    splits = {}
    for split in ("train", "val", "test"):
        raw = read_csv_any(DATA_RAW / "bdshs" / f"{split}.csv")
        df = _standardize(raw, "sentence", "hate speech", "bdshs",
                          target_col="target", type_col="type")
        df = _clean(df, f"bdshs/{split}")
        df = _add_text_clean(df)
        splits[split] = df
    return splits


def process_bidwesh(bdshs_train_val_texts: set[str]) -> pd.DataFrame:
    labels = read_csv_any(DATA_RAW / "bidwesh" / "BIDWESH Dataset.csv")
    sources = read_csv_any(DATA_RAW / "bidwesh" / "Regional Translated Texts.csv")
    # Files are row-aligned (verified: dialect columns identical across both).
    labels = labels.copy()
    labels["standard_bangla"] = sources["Standard Bangla"]

    frames = []
    for dialect in ("Chittagong", "Noakhali", "Barishal"):
        part = _standardize(
            labels, dialect, "hate speech", "bidwesh",
            target_col="target", type_col="type",
            extra={"dialect": dialect.lower()},
        )
        part["standard_bangla"] = labels["standard_bangla"].map(normalize_bangla)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["overlaps_bdshs_train"] = df["standard_bangla"].isin(bdshs_train_val_texts)
    df = _clean(df, "bidwesh (3 dialects melted)")
    df = _add_text_clean(df)
    n_overlap = int(df["overlaps_bdshs_train"].sum())
    print(f"    {n_overlap}/{len(df)} rows have a source sentence present in "
          f"BD-SHS train/val (marked overlaps_bdshs_train=True)")
    return df


def process_30k() -> pd.DataFrame:
    raw = read_csv_any(DATA_RAW / "bengali_hs_30k" / "Bengali hate speech .csv")
    df = _standardize(raw, "sentence", "hate", "bengali_hs_30k",
                      extra={"category": raw["category"]})
    df = _clean(df, "bengali_hs_30k")
    df = _add_text_clean(df)
    # Stratified 80/10/10 split recorded as a column (unused by default).
    from sklearn.model_selection import train_test_split

    idx_train, idx_rest = train_test_split(
        df.index, test_size=0.2, stratify=df["label"], random_state=DEFAULT_SEED)
    idx_val, idx_test = train_test_split(
        idx_rest, test_size=0.5, stratify=df.loc[idx_rest, "label"],
        random_state=DEFAULT_SEED)
    df["split"] = "train"
    df.loc[idx_val, "split"] = "val"
    df.loc[idx_test, "split"] = "test"
    return df


def label_dist(df: pd.DataFrame) -> str:
    c = df["label"].value_counts().sort_index()
    total = len(df)
    parts = [f"{LABEL_NAMES[k]}={v} ({v / total:.1%})" for k, v in c.items()]
    return f"n={total} | " + ", ".join(parts)


def main() -> None:
    setup_utf8_stdout()
    set_seed()
    ensure_dirs(DATA_PROCESSED, REPORTS_DIR)

    print("Processing BD-SHS (official splits kept) ...")
    bdshs = process_bdshs()

    train_val_texts = set(bdshs["train"]["text"]) | set(bdshs["val"]["text"])
    print("Processing BIDWESH (dialect eval set) ...")
    bidwesh = process_bidwesh(train_val_texts)

    print("Processing Bengali HS 30K (extra, unused by default) ...")
    extra30k = process_30k()

    stats_lines = ["# Processed data statistics", ""]
    for name, df in [("train", bdshs["train"]), ("val", bdshs["val"]),
                     ("test", bdshs["test"])]:
        path = DATA_PROCESSED / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        line = f"- `{path.name}` (BD-SHS {name}): {label_dist(df)}"
        stats_lines.append(line)
        print(line)

    bidwesh.to_csv(DATA_PROCESSED / "bidwesh_test.csv", index=False,
                   encoding="utf-8")
    line = f"- `bidwesh_test.csv`: {label_dist(bidwesh)}"
    stats_lines.append(line)
    for d in ("chittagong", "noakhali", "barishal"):
        sub = bidwesh[bidwesh["dialect"] == d]
        stats_lines.append(f"  - {d}: {label_dist(sub)}")
    n_ov = int(bidwesh["overlaps_bdshs_train"].sum())
    stats_lines.append(
        f"  - rows whose Standard-Bangla source appears in BD-SHS train/val: "
        f"{n_ov} ({n_ov / len(bidwesh):.1%}) -> excluded in 'bidwesh-clean' eval")
    print(line)

    extra30k.to_csv(DATA_PROCESSED / "extra_bengali_hs_30k.csv", index=False,
                    encoding="utf-8")
    line = f"- `extra_bengali_hs_30k.csv`: {label_dist(extra30k)} (not used by default)"
    stats_lines.append(line)
    print(line)

    save_json(
        {
            "taskA": {str(k): v for k, v in LABEL_NAMES.items()},
            "taskB_target_vocab": TARGET_VOCAB,
            "taskC_type_vocab": TYPE_VOCAB,
            "multilabel_separator": "_",
        },
        DATA_PROCESSED / "label_mapping.json",
    )
    stats_lines.append("")
    stats_lines.append("Label mapping: 0 = not_hate, 1 = hate "
                       "(`label_mapping.json`).")

    with open(REPORTS_DIR / "data_stats.md", "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines) + "\n")
    print(f"\nWrote {REPORTS_DIR / 'data_stats.md'}")


if __name__ == "__main__":
    main()
