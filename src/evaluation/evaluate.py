"""Evaluate saved models on the clean / synthetic-noisy / real-dialect test
sets and build the robustness summary.

Works with both model kinds saved under outputs/models/<name>/:
- sklearn pipelines:  model.joblib + meta.json
- HF transformers:    save_pretrained dir + meta.json

Usage:
    python -m src.evaluation.evaluate --model-dir outputs/models/tfidf_lr
    python -m src.evaluation.evaluate --all          # every dir with meta.json
    python -m src.evaluation.evaluate --all --skip-existing
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import record_results
from src.utils.common import (
    DATA_PROCESSED,
    MODELS_DIR,
    RESULTS_DIR,
    load_json,
    setup_utf8_stdout,
)

# Main test sets (figures generated). Subset rows (no figures) are derived
# from bidwesh: bidwesh_clean (non-overlapping) + one per dialect.
MAIN_TEST_SETS = {
    "clean": DATA_PROCESSED / "test.csv",
    "augmented": DATA_PROCESSED / "test_augmented.csv",
    "bidwesh": DATA_PROCESSED / "bidwesh_test.csv",
}


class SklearnPredictor:
    def __init__(self, model_dir: Path, meta: dict):
        import joblib

        self.pipeline = joblib.load(model_dir / "model.joblib")
        self.input_column = meta.get("input_column", "text_clean")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(df[self.input_column].fillna("").tolist())


class HFPredictor:
    def __init__(self, model_dir: Path, meta: dict, batch_size: int = 32):
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir).to(self.device).eval()
        self.max_len = int(meta.get("max_len", 128))
        self.batch_size = batch_size
        self.input_column = meta.get("input_column", "text")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        texts = df[self.input_column].fillna("").tolist()
        preds = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(
                    texts[i:i + self.batch_size], truncation=True,
                    max_length=self.max_len, padding=True,
                    return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                preds.append(logits.argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)


def load_predictor(model_dir: Path):
    meta = load_json(model_dir / "meta.json")
    if (model_dir / "model.joblib").exists():
        return SklearnPredictor(model_dir, meta), meta
    return HFPredictor(model_dir, meta), meta


def evaluate_saved_model(model_dir: Path, skip_existing: bool = False) -> None:
    model_dir = Path(model_dir)
    predictor, meta = load_predictor(model_dir)
    model_id = meta["model_id"]
    train_variant = meta.get("train_variant", "clean")
    tag = meta.get("tag", "full")
    print(f"  evaluating {model_id} (train={train_variant}, tag={tag})")

    if skip_existing:
        summary_path = RESULTS_DIR / "results_summary.csv"
        if summary_path.exists():
            s = pd.read_csv(summary_path)
            done = s[(s["model"] == model_id) &
                     (s["train_variant"] == train_variant)]["test_set"]
            if set(MAIN_TEST_SETS) <= set(done):
                print("    all main test sets already recorded, skipping")
                return

    # Smoke-tagged models (CPU sanity runs) are evaluated on a 500-row
    # sample per test set so transformer smoke tests stay fast.
    limit = 500 if tag == "smoke" else None

    for test_name, path in MAIN_TEST_SETS.items():
        df = pd.read_csv(path)
        if limit and len(df) > limit:
            df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        preds = predictor.predict(df)
        record_results(model_id, train_variant, test_name,
                       df["label"].values, preds, tag=tag)
        if test_name == "bidwesh":
            subsets = {"bidwesh_clean": df[~df["overlaps_bdshs_train"]]}
            for dialect in ("chittagong", "noakhali", "barishal"):
                subsets[f"bidwesh_{dialect}"] = df[df["dialect"] == dialect]
            for sub_name, sub in subsets.items():
                if len(sub) == 0:
                    continue
                record_results(model_id, train_variant, sub_name,
                               sub["label"].values, preds[sub.index.values],
                               tag=tag, make_figure=False)


def build_robustness_summary() -> pd.DataFrame:
    """Pivot results_summary.csv into per-model robustness drops."""
    summary = pd.read_csv(RESULTS_DIR / "results_summary.csv")
    rows = []
    for (model, variant), grp in summary.groupby(["model", "train_variant"]):
        f1 = grp.set_index("test_set")["f1_macro"]
        if "clean" not in f1.index:
            continue
        clean = f1["clean"]
        row = {
            "model": model,
            "train_variant": variant,
            "tag": grp["tag"].iloc[0],
            "f1_clean": round(clean, 4),
        }
        for ts in ("augmented", "bidwesh", "bidwesh_clean", "bidwesh_chittagong",
                   "bidwesh_noakhali", "bidwesh_barishal"):
            if ts in f1.index:
                row[f"f1_{ts}"] = round(f1[ts], 4)
                row[f"drop_{ts}"] = round(clean - f1[ts], 4)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["model", "train_variant"])
    out.to_csv(RESULTS_DIR / "robustness_summary.csv", index=False,
               encoding="utf-8")
    return out


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--all", action="store_true",
                        help="evaluate every outputs/models/*/meta.json")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.all:
        dirs = sorted(p.parent for p in MODELS_DIR.glob("*/meta.json"))
    elif args.model_dir:
        dirs = [args.model_dir]
    else:
        parser.error("pass --model-dir or --all")

    for d in dirs:
        evaluate_saved_model(d, skip_existing=args.skip_existing)

    out = build_robustness_summary()
    print("\nRobustness summary (macro-F1, drop vs clean):")
    print(out.to_string(index=False))
    print(f"\nWrote {RESULTS_DIR / 'robustness_summary.csv'}")


if __name__ == "__main__":
    main()
