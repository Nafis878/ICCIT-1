"""Evaluate saved models on the v2 protocol test sets and dump per-example
predictions for statistical analysis.

Test sets (v2 protocol):
- clean            BD-SHS official test split
- augmented        synthetic noisy copy of the test split
- bidwesh_heldout  BIDWESH held-out half (bidwesh_split == 'test'),
                   near-duplicates of BD-SHS train/val excluded
  (+ per-dialect subset rows derived from bidwesh_heldout)
- bidwesh_dev      adaptation dev set (on demand, --test-set bidwesh_dev)

Every evaluation writes outputs/predictions/<model>_<variant>_s<seed>_
<testset>.csv (row_id, y_true, y_pred, p_hate) — all statistics
(bootstrap CIs, McNemar) are computed from these dumps by
src.evaluation.stats, so GPU machines only ever train + predict.

Usage:
    python -m src.evaluation.evaluate --model-dir outputs/models/<dir>
    python -m src.evaluation.evaluate --all [--skip-existing]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import record_results
from src.utils.common import (
    DATA_PROCESSED,
    MODELS_DIR,
    OUTPUTS,
    RESULTS_DIR,
    ensure_dirs,
    load_json,
    setup_utf8_stdout,
)

PREDICTIONS_DIR = OUTPUTS / "predictions"
MAIN_TEST_SETS = ("clean", "augmented", "bidwesh_heldout")
DIALECTS = ("chittagong", "noakhali", "barishal")


def load_test_set(name: str) -> pd.DataFrame:
    if name == "clean":
        return pd.read_csv(DATA_PROCESSED / "test.csv")
    if name == "augmented":
        return pd.read_csv(DATA_PROCESSED / "test_augmented.csv")
    if name in ("bidwesh_heldout", "bidwesh_dev"):
        df = pd.read_csv(DATA_PROCESSED / "bidwesh_test.csv")
        split = "test" if name == "bidwesh_heldout" else "dev"
        df = df[df["bidwesh_split"] == split]
        if name == "bidwesh_heldout" and "near_dup_bdshs_train" in df.columns:
            df = df[~df["near_dup_bdshs_train"]]
        return df.reset_index(drop=True)
    raise ValueError(name)


class SklearnPredictor:
    def __init__(self, model_dir: Path, meta: dict):
        import joblib

        self.pipeline = joblib.load(model_dir / "model.joblib")
        self.input_column = meta.get("input_column", "text_clean")

    def predict_with_proba(self, df: pd.DataFrame):
        texts = df[self.input_column].fillna("").tolist()
        preds = self.pipeline.predict(texts)
        proba = None
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(texts)[:, 1]
        return np.asarray(preds), proba


class HFPredictor:
    def __init__(self, model_dir: Path, meta: dict, batch_size: int = 64):
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

    def predict_with_proba(self, df: pd.DataFrame):
        texts = df[self.input_column].fillna("").tolist()
        probs = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(
                    texts[i:i + self.batch_size], truncation=True,
                    max_length=self.max_len, padding=True,
                    return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                probs.append(
                    self.torch.softmax(logits.float(), dim=-1).cpu().numpy())
        probs = np.concatenate(probs)
        return probs.argmax(axis=1), probs[:, 1]


def load_predictor(model_dir: Path):
    meta = load_json(model_dir / "meta.json")
    if (model_dir / "model.joblib").exists():
        return SklearnPredictor(model_dir, meta), meta
    return HFPredictor(model_dir, meta), meta


def dump_predictions(model_id: str, variant: str, seed: int, test_name: str,
                     df: pd.DataFrame, preds, proba) -> None:
    ensure_dirs(PREDICTIONS_DIR)
    out = pd.DataFrame({
        "row_id": df.index.values,
        "y_true": df["label"].values,
        "y_pred": preds,
    })
    if proba is not None:
        out["p_hate"] = np.round(proba, 5)
    out.to_csv(
        PREDICTIONS_DIR / f"{model_id}_{variant}_s{seed}_{test_name}.csv",
        index=False)


def evaluate_saved_model(model_dir: Path, skip_existing: bool = False) -> None:
    model_dir = Path(model_dir)
    meta = load_json(model_dir / "meta.json")
    model_id = meta["model_id"]
    train_variant = meta.get("train_variant", "clean")
    seed = int(meta.get("seed", 42))
    tag = meta.get("tag", "full")

    if skip_existing:
        summary_path = RESULTS_DIR / "results_summary.csv"
        if summary_path.exists():
            s = pd.read_csv(summary_path)
            if "seed" not in s.columns:
                s["seed"] = 42
            done = s[(s["model"] == model_id)
                     & (s["train_variant"] == train_variant)
                     & (s["seed"] == seed)]["test_set"]
            if set(MAIN_TEST_SETS) <= set(done):
                print(f"  {model_id}/{train_variant}/s{seed}: already "
                      f"recorded, skipping")
                return

    predictor, _ = load_predictor(model_dir)
    print(f"  evaluating {model_id} (train={train_variant}, seed={seed}, "
          f"tag={tag})")
    limit = 500 if tag == "smoke" else None

    for test_name in MAIN_TEST_SETS:
        df = load_test_set(test_name)
        if limit and len(df) > limit:
            df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        preds, proba = predictor.predict_with_proba(df)
        record_results(model_id, train_variant, test_name,
                       df["label"].values, preds, tag=tag, seed=seed)
        dump_predictions(model_id, train_variant, seed, test_name,
                         df, preds, proba)
        if test_name == "bidwesh_heldout":
            for dialect in DIALECTS:
                sub = df[df["dialect"] == dialect]
                if len(sub) == 0:
                    continue
                record_results(model_id, train_variant,
                               f"bidwesh_{dialect}", sub["label"].values,
                               preds[sub.index.values], tag=tag, seed=seed,
                               make_figure=False)


def build_robustness_summary() -> pd.DataFrame:
    """Per (model, variant, seed) macro-F1 drops vs clean."""
    summary = pd.read_csv(RESULTS_DIR / "results_summary.csv")
    if "seed" not in summary.columns:
        summary["seed"] = 42
    rows = []
    for (model, variant, seed), grp in summary.groupby(
            ["model", "train_variant", "seed"]):
        f1 = grp.set_index("test_set")["f1_macro"]
        if "clean" not in f1.index:
            continue
        clean = f1["clean"]
        row = {"model": model, "train_variant": variant, "seed": seed,
               "tag": grp["tag"].iloc[0], "f1_clean": round(clean, 4)}
        for ts in ("augmented", "bidwesh_heldout", "bidwesh_chittagong",
                   "bidwesh_noakhali", "bidwesh_barishal"):
            if ts in f1.index:
                row[f"f1_{ts}"] = round(f1[ts], 4)
                row[f"drop_{ts}"] = round(clean - f1[ts], 4)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["model", "train_variant", "seed"])
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
