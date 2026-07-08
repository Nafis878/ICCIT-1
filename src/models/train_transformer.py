"""Fine-tune a transformer for binary Bangla hate speech detection.

v2 protocol: seed-aware output dirs and result rows, two training methods:
- standard      cross-entropy on whichever --train-file is given
- consistency   NCT (noise-consistency training): clean+noisy paired views
                with symmetric-KL consistency loss (see
                src.models.consistency_trainer); trains from train.csv.

Transformers v5 API (works on >=4.46): ``eval_strategy`` +
``processing_class``. Device auto-detected. ``--smoke`` = tiny CPU check.

Examples:
    python -m src.models.train_transformer --model-name csebuetnlp/banglabert --seed 42
    python -m src.models.train_transformer --model-name csebuetnlp/banglabert \
        --train-file data/processed/train_aug_dia.csv --seed 43
    python -m src.models.train_transformer --model-name csebuetnlp/banglabert \
        --method consistency --consistency-lambda 1.0 --seed 44
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.utils.common import (
    DATA_PROCESSED,
    DEFAULT_SEED,
    MODELS_DIR,
    ensure_dirs,
    save_json,
    set_seed,
    setup_utf8_stdout,
)

INPUT_COLUMN = "text"
LABEL_NAMES = ["not_hate", "hate"]

VARIANT_OF_FILE = {
    "train.csv": "clean",
    "train_augmented.csv": "augmented",
    "train_dialect_aug.csv": "dialect",
    "train_aug_dia.csv": "aug_dia",
}


class TextDataset(torch.utils.data.Dataset):
    """Pre-tokenized dataset over a pandas frame (no HF `datasets` dep)."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.encodings = tokenizer(
            df[INPUT_COLUMN].fillna("").tolist(),
            truncation=True, max_length=max_len)
        self.labels = df["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def slugify(model_name: str) -> str:
    # Also handles local Windows paths passed as --model-name.
    return (model_name.replace("\\", "/").rstrip("/").split("/")[-1]
            .replace("_", "-").lower())


def run_training(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_variant: str,
    model_id: str,
    out_dir: Path,
    *,
    method: str = "standard",
    consistency_lambda: float = 1.0,
    batch_size: int = 16,
    lr: float = 2e-5,
    epochs: float = 3,
    max_len: int = 128,
    seed: int = DEFAULT_SEED,
    tag: str = "full",
    extra_meta: dict | None = None,
    eval_after: bool = True,
) -> Path:
    """Train + save + (optionally) evaluate. Shared by the CLI, adapt.py
    and the Colab runner."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dirs(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
        id2label={i: n for i, n in enumerate(LABEL_NAMES)},
        label2id={n: i for i, n in enumerate(LABEL_NAMES)})

    val_ds = TextDataset(val_df, tokenizer, max_len)
    trainer_cls, trainer_kwargs = Trainer, {}
    if method == "consistency":
        from src.models.consistency_trainer import (
            ConsistencyTrainer,
            PairedCollator,
            PairedTextDataset,
        )

        train_ds = PairedTextDataset(train_df, tokenizer, max_len, seed)
        trainer_cls = ConsistencyTrainer
        trainer_kwargs = {"consistency_lambda": consistency_lambda,
                          "data_collator": PairedCollator(tokenizer)}
    else:
        train_ds = TextDataset(train_df, tokenizer, max_len)

    print(f"device={device} model={model_name} method={method} "
          f"train={len(train_ds)} val={len(val_ds)} "
          f"variant={train_variant} seed={seed} tag={tag}")

    steps_per_epoch = max(1, -(-len(train_ds) // batch_size))
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=int(0.06 * steps_per_epoch * epochs),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10 if tag == "smoke" else 100,
        seed=seed,
        fp16=(device == "cuda"),
        report_to=[],
        # the paired-view keys must survive the RemoveColumnsCollator wrapper
        remove_unused_columns=(method != "consistency"),
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        **trainer_kwargs,
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    val_metrics = trainer.evaluate()
    print(f"best val: f1_macro={val_metrics.get('eval_f1_macro'):.4f} "
          f"acc={val_metrics.get('eval_accuracy'):.4f} "
          f"({train_secs / 60:.1f} min)")

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    import shutil

    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    meta = {
        "model_id": model_id,
        "model_type": "hf",
        "model_name": model_name,
        "method": method,
        "input_column": INPUT_COLUMN,
        "train_variant": train_variant,
        "train_rows": len(train_ds),
        "train_seconds": round(train_secs, 1),
        "tag": tag,
        "seed": seed,
        "max_len": max_len,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "device": device,
        "val_f1_macro": float(val_metrics.get("eval_f1_macro", 0.0)),
    }
    if method == "consistency":
        meta["consistency_lambda"] = consistency_lambda
    if extra_meta:
        meta.update(extra_meta)
    save_json(meta, out_dir / "meta.json")
    print(f"saved to {out_dir}")

    if eval_after:
        from src.evaluation.evaluate import evaluate_saved_model

        evaluate_saved_model(out_dir)
    return out_dir


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="csebuetnlp/banglabert")
    parser.add_argument("--task", default="taskA", choices=["taskA"])
    parser.add_argument("--method", default="standard",
                        choices=["standard", "consistency"])
    parser.add_argument("--consistency-lambda", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-file", type=Path,
                        default=DATA_PROCESSED / "train.csv")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-eval-after", action="store_true")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(DATA_PROCESSED / "val.csv")
    train_variant = VARIANT_OF_FILE.get(args.train_file.name,
                                        args.train_file.stem)
    tag = "smoke" if args.smoke else "full"
    max_len, epochs = args.max_len, args.epochs
    if args.smoke:
        epochs, max_len = 1, 64
        train_df = train_df.sample(n=min(200, len(train_df)),
                                   random_state=args.seed)
        val_df = val_df.sample(n=min(100, len(val_df)),
                               random_state=args.seed)

    model_id = slugify(args.model_name)
    if args.method == "consistency":
        model_id += "-nct"
    if args.smoke:
        model_id += "_smoke"
    out_dir = args.output_dir or MODELS_DIR / (
        f"{model_id}_{train_variant}_s{args.seed}")

    run_training(
        args.model_name, train_df, val_df, train_variant, model_id, out_dir,
        method=args.method, consistency_lambda=args.consistency_lambda,
        batch_size=args.batch_size, lr=args.lr, epochs=epochs,
        max_len=max_len, seed=args.seed, tag=tag,
        eval_after=not args.no_eval_after)


if __name__ == "__main__":
    main()
