"""Fine-tune a transformer for binary Bangla hate speech detection (Task A).

Written against the transformers v5 API (works on >=4.46 too):
``eval_strategy`` + ``processing_class``. Device is auto-detected
(CUDA if available, otherwise CPU). ``--smoke`` runs a tiny end-to-end
check that finishes on CPU in a few minutes.

Robustness training = pass ``--train-file data/processed/train_augmented.csv``
(original train + synthetic noisy copies) and compare against the clean run.

Examples:
    python -m src.models.train_transformer --model-name csebuetnlp/banglabert --smoke
    python -m src.models.train_transformer --model-name xlm-roberta-base --smoke
    # full runs (GPU strongly recommended):
    python -m src.models.train_transformer --model-name csebuetnlp/banglabert \
        --epochs 3 --batch-size 16 --lr 2e-5 --max-len 128
    python -m src.models.train_transformer --model-name csebuetnlp/banglabert \
        --train-file data/processed/train_augmented.csv
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

INPUT_COLUMN = "text"
LABEL_NAMES = ["not_hate", "hate"]


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


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="csebuetnlp/banglabert")
    parser.add_argument("--task", default="taskA", choices=["taskA"],
                        help="binary hate/not-hate (Task B/C are future work)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: outputs/models/<slug>[_aug][_smoke]")
    parser.add_argument("--train-file", type=Path,
                        default=DATA_PROCESSED / "train.csv")
    parser.add_argument("--smoke", action="store_true",
                        help="200 train / 100 val rows, 1 epoch, max_len 64")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-eval-after", action="store_true",
                        help="skip test-set evaluation after training")
    args = parser.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model_name}")

    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(DATA_PROCESSED / "val.csv")
    train_variant = ("augmented" if "augmented" in args.train_file.name
                     else "clean")
    tag = "smoke" if args.smoke else "full"
    if args.smoke:
        args.epochs, args.max_len = 1, 64
        train_df = train_df.sample(n=min(200, len(train_df)),
                                   random_state=args.seed)
        val_df = val_df.sample(n=min(100, len(val_df)),
                               random_state=args.seed)

    model_id = slugify(args.model_name) + ("_smoke" if args.smoke else "")
    out_dir = args.output_dir or MODELS_DIR / (
        model_id + ("_aug" if train_variant == "augmented" else ""))
    ensure_dirs(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2,
        id2label={i: n for i, n in enumerate(LABEL_NAMES)},
        label2id={n: i for i, n in enumerate(LABEL_NAMES)})

    train_ds = TextDataset(train_df, tokenizer, args.max_len)
    val_ds = TextDataset(val_df, tokenizer, args.max_len)
    print(f"train={len(train_ds)} val={len(val_ds)} "
          f"variant={train_variant} tag={tag}")

    steps_per_epoch = max(1, -(-len(train_ds) // args.batch_size))
    warmup_steps = int(0.06 * steps_per_epoch * args.epochs)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10 if args.smoke else 100,
        seed=args.seed,
        fp16=(device == "cuda"),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
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
    save_json(
        {
            "model_id": model_id,
            "model_type": "hf",
            "model_name": args.model_name,
            "input_column": INPUT_COLUMN,
            "train_variant": train_variant,
            "train_rows": len(train_ds),
            "train_seconds": round(train_secs, 1),
            "tag": tag,
            "seed": args.seed,
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "device": device,
            "val_f1_macro": float(val_metrics.get("eval_f1_macro", 0.0)),
        },
        out_dir / "meta.json",
    )
    print(f"saved to {out_dir}")

    if not args.no_eval_after:
        evaluate_saved_model(out_dir)


if __name__ == "__main__":
    main()
