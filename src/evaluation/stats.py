"""Statistical analysis over per-example prediction dumps.

Inputs: outputs/predictions/<model>_<variant>_s<seed>_<testset>.csv
(written by src.evaluation.evaluate).

Outputs:
- outputs/results/stats_summary.csv       per (model, variant, test_set):
    macro-F1 mean +/- std over seeds, 95% bootstrap CI (over test examples,
    seed-pooled), n_seeds, accuracy mean.
- outputs/results/significance_tests.csv  paired comparisons: bootstrap
    p-value on the macro-F1 delta + McNemar exact/chi2 on pooled decisions.

Comparisons default to the paper's key claims and any --compare pairs:
    modelA:variantA vs modelB:variantB on a given test set.

Usage:
    python -m src.evaluation.stats                # all + default comparisons
    python -m src.evaluation.stats --compare "banglabert:aug_dia=banglabert:clean@bidwesh_heldout"
"""
import argparse
import itertools
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluation.evaluate import PREDICTIONS_DIR
from src.utils.common import RESULTS_DIR, setup_utf8_stdout

N_BOOT = 1000
RNG_SEED = 42
_FNAME = re.compile(r"^(?P<model>.+)_(?P<variant>clean|augmented|dialect|"
                    r"aug_dia)_s(?P<seed>\d+)_(?P<testset>clean|augmented|"
                    r"bidwesh_heldout|bidwesh_dev)\.csv$")


def load_dumps() -> pd.DataFrame:
    """Index of all prediction dumps with parsed keys."""
    rows = []
    for p in sorted(PREDICTIONS_DIR.glob("*.csv")):
        m = _FNAME.match(p.name)
        if not m:
            continue
        rows.append({**m.groupdict(), "path": p})
    df = pd.DataFrame(rows)
    if len(df):
        df["seed"] = df["seed"].astype(int)
        df = df[~df["model"].str.contains("smoke")]  # sanity runs excluded
    return df


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def bootstrap_ci(y_true, preds_by_seed: list[np.ndarray],
                 n_boot: int = N_BOOT) -> tuple[float, float]:
    """95% CI for seed-averaged macro-F1, bootstrapping test examples."""
    rng = np.random.RandomState(RNG_SEED)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        stats[b] = np.mean([macro_f1(y_true[idx], p[idx])
                            for p in preds_by_seed])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_bootstrap_p(y_true, preds_a: list[np.ndarray],
                       preds_b: list[np.ndarray],
                       n_boot: int = N_BOOT) -> tuple[float, float]:
    """Two-sided paired bootstrap on the seed-averaged macro-F1 delta.
    Returns (observed_delta, p_value)."""
    rng = np.random.RandomState(RNG_SEED)
    n = len(y_true)
    obs = (np.mean([macro_f1(y_true, p) for p in preds_a])
           - np.mean([macro_f1(y_true, p) for p in preds_b]))
    if obs == 0:  # no direction to test
        return 0.0, 1.0
    count = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        d = (np.mean([macro_f1(y_true[idx], p[idx]) for p in preds_a])
             - np.mean([macro_f1(y_true[idx], p[idx]) for p in preds_b]))
        # sign test relative to zero (two-sided via doubling)
        if (d <= 0 and obs > 0) or (d >= 0 and obs < 0):
            count += 1
    p = min(1.0, 2 * count / n_boot)
    return float(obs), float(p)


def mcnemar_p(y_true, pred_a, pred_b) -> tuple[int, int, float]:
    """Exact binomial McNemar on discordant pairs (chi2 fallback for
    large counts). Returns (b_only_correct, a_only_correct... ) as
    (n01, n10, p)."""
    from scipy.stats import binomtest, chi2

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    n01 = int((~correct_a & correct_b).sum())  # b right, a wrong
    n10 = int((correct_a & ~correct_b).sum())  # a right, b wrong
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    if n < 200:
        p = binomtest(min(n01, n10), n, 0.5).pvalue
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / n
        p = float(chi2.sf(stat, df=1))
    return n01, n10, float(p)


def build_stats_summary(index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, variant, testset), grp in index.groupby(
            ["model", "variant", "testset"]):
        preds, y_true = [], None
        for _, r in grp.sort_values("seed").iterrows():
            d = pd.read_csv(r["path"])
            y_true = d["y_true"].values
            preds.append(d["y_pred"].values)
        f1s = [macro_f1(y_true, p) for p in preds]
        accs = [(y_true == p).mean() for p in preds]
        lo, hi = bootstrap_ci(y_true, preds)
        rows.append({
            "model": model, "train_variant": variant, "test_set": testset,
            "n_seeds": len(preds), "n_examples": len(y_true),
            "f1_macro_mean": round(float(np.mean(f1s)), 4),
            "f1_macro_std": round(float(np.std(f1s)), 4),
            "f1_ci95_lo": round(lo, 4), "f1_ci95_hi": round(hi, 4),
            "accuracy_mean": round(float(np.mean(accs)), 4),
        })
    out = pd.DataFrame(rows).sort_values(["test_set", "f1_macro_mean"],
                                         ascending=[True, False])
    out.to_csv(RESULTS_DIR / "stats_summary.csv", index=False)
    return out


DEFAULT_COMPARISONS = [
    # (model_a, variant_a, model_b, variant_b, testset) — a vs b
    ("banglabert", "augmented", "banglabert", "clean", "bidwesh_heldout"),
    ("banglabert", "augmented", "banglabert", "clean", "augmented"),
    ("banglabert-nct", "clean", "banglabert", "augmented", "bidwesh_heldout"),
    ("banglabert", "aug_dia", "banglabert", "augmented", "bidwesh_heldout"),
    ("banglabert", "dialect", "banglabert", "clean", "bidwesh_heldout"),
    ("banglabert", "clean", "tfidf_lr", "clean", "clean"),
    ("banglabert", "clean", "tfidf_lr", "clean", "bidwesh_heldout"),
    ("tfidf_lr", "augmented", "tfidf_lr", "clean", "bidwesh_heldout"),
    ("tfidf_lr", "dialect", "tfidf_lr", "clean", "bidwesh_heldout"),
    ("tfidf_lr", "aug_dia", "tfidf_lr", "clean", "bidwesh_heldout"),
]


def run_comparisons(index: pd.DataFrame, comparisons) -> pd.DataFrame:
    cache: dict = {}

    def get(model, variant, testset):
        key = (model, variant, testset)
        if key not in cache:
            grp = index[(index["model"] == model)
                        & (index["variant"] == variant)
                        & (index["testset"] == testset)]
            if grp.empty:
                cache[key] = None
            else:
                preds, y_true = [], None
                for _, r in grp.sort_values("seed").iterrows():
                    d = pd.read_csv(r["path"])
                    y_true = d["y_true"].values
                    preds.append(d["y_pred"].values)
                cache[key] = (y_true, preds)
        return cache[key]

    rows = []
    for ma, va, mb, vb, ts in comparisons:
        a, b = get(ma, va, ts), get(mb, vb, ts)
        if a is None or b is None:
            continue
        y_true, preds_a = a
        _, preds_b = b
        delta, p_boot = paired_bootstrap_p(y_true, preds_a, preds_b)
        # McNemar on majority-vote decisions across seeds
        vote_a = (np.mean(preds_a, axis=0) >= 0.5).astype(int)
        vote_b = (np.mean(preds_b, axis=0) >= 0.5).astype(int)
        n01, n10, p_mc = mcnemar_p(y_true, vote_a, vote_b)
        rows.append({
            "a": f"{ma}:{va}", "b": f"{mb}:{vb}", "test_set": ts,
            "delta_f1_macro": round(delta, 4),
            "p_paired_bootstrap": round(p_boot, 4),
            "mcnemar_b_only": n01, "mcnemar_a_only": n10,
            "p_mcnemar": round(p_mc, 6),
            "n_seeds_a": len(preds_a), "n_seeds_b": len(preds_b),
        })
        print(f"  {ma}:{va} vs {mb}:{vb} @ {ts}: dF1={delta:+.4f} "
              f"p_boot={p_boot:.4f} p_mcnemar={p_mc:.2e}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "significance_tests.csv", index=False)
    return out


def parse_compare(spec: str):
    m = re.match(r"^(.+):(.+)=(.+):(.+)@(.+)$", spec)
    if not m:
        raise argparse.ArgumentTypeError(
            "format: modelA:variantA=modelB:variantB@testset")
    return (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5))


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=parse_compare, action="append",
                        default=[])
    args = parser.parse_args()

    index = load_dumps()
    if index.empty:
        print("no prediction dumps found under outputs/predictions/")
        return
    print(f"{len(index)} prediction dumps, "
          f"{index.groupby(['model', 'variant']).ngroups} configurations")

    summary = build_stats_summary(index)
    print(summary.to_string(index=False))
    print()
    run_comparisons(index, DEFAULT_COMPARISONS + args.compare)
    print(f"\nWrote {RESULTS_DIR / 'stats_summary.csv'} and "
          f"significance_tests.csv")


if __name__ == "__main__":
    main()
