"""Few-shot dialect adaptation: continue fine-tuning a trained model on N
examples from the BIDWESH *adapt* split (grouped split; the held-out
benchmark half is never touched). Validation = BIDWESH dev split.

Produces the data-efficiency curve: how much real dialect data does it
take to close the dialect gap, and what does it cost on the clean test
(catastrophic forgetting check — evaluation covers clean too).

Usage:
    python -m src.models.adapt --from-dir outputs/models/banglabert_augmented_s42 --n 250
    python -m src.models.adapt --from-dir ... --n 0        # 0 = full adapt split
"""
import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.evaluate import load_test_set
from src.models.train_transformer import run_training
from src.utils.common import (
    DATA_PROCESSED,
    DEFAULT_SEED,
    MODELS_DIR,
    load_json,
    setup_utf8_stdout,
)


def load_adapt_rows(n: int, seed: int) -> pd.DataFrame:
    """Full adapt split, or a label-stratified sample of n rows."""
    df = pd.read_csv(DATA_PROCESSED / "bidwesh_test.csv")
    df = df[df["bidwesh_split"] == "adapt"].reset_index(drop=True)
    if n and n < len(df):
        per_label = n // 2
        df = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), per_label),
                                          random_state=seed))
                .sample(frac=1, random_state=seed)  # shuffle
                .reset_index(drop=True))
    return df


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-dir", type=Path, required=True,
                        help="trained model dir to adapt from")
    parser.add_argument("--n", type=int, default=0,
                        help="adapt examples (0 = full adapt split)")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base_meta = load_json(args.from_dir / "meta.json")
    adapt_df = load_adapt_rows(args.n, args.seed)
    val_df = load_test_set("bidwesh_dev")
    tag = "smoke" if args.smoke else base_meta.get("tag", "full")
    epochs, max_len = args.epochs, int(base_meta.get("max_len", 128))
    if args.smoke:
        epochs, max_len = 1, 64
        adapt_df = adapt_df.head(100)
        val_df = val_df.head(50)

    n_label = args.n if args.n else "full"
    model_id = f"{base_meta['model_id']}-adapt{n_label}"
    train_variant = base_meta.get("train_variant", "clean")
    out_dir = MODELS_DIR / f"{model_id}_{train_variant}_s{args.seed}"
    print(f"adapting {base_meta['model_id']} on {len(adapt_df)} BIDWESH "
          f"adapt rows (n={n_label}, seed={args.seed})")

    run_training(
        str(args.from_dir), adapt_df, val_df, train_variant, model_id,
        out_dir, method="standard", batch_size=args.batch_size, lr=args.lr,
        epochs=epochs, max_len=max_len, seed=args.seed, tag=tag,
        extra_meta={"adapted_from": str(args.from_dir),
                    "adapt_n": int(args.n) if args.n else len(adapt_df)})


if __name__ == "__main__":
    main()
