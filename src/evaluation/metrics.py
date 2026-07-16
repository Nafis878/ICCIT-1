"""Metric computation, classification reports, confusion-matrix figures,
and the shared results_summary.csv (one row per model x train-variant x
test-set, upserted so re-runs overwrite instead of duplicating)."""
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.common import FIGURES_DIR, RESULTS_DIR, ensure_dirs, save_json

LABEL_NAMES = ["not_hate", "hate"]

# Single-hue sequential ramp (light -> dark) for the heatmap.
_SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_INK = "#1f2429"
_MUTED = "#5f6b76"


@contextmanager
def _summary_lock(timeout_s: float = 30.0, stale_s: float = 120.0):
    """Cross-process lock via O_EXCL lockfile (portable, no deps)."""
    lock = RESULTS_DIR / ".summary.lock"
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:  # clear locks abandoned by killed processes
                if time.time() - lock.stat().st_mtime > stale_s:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.time() > deadline:  # proceed rather than deadlock
                break
            time.sleep(0.2)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def compute_all(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1],
                            zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "f1_not_hate": float(per_class_f1[0]),
        "f1_hate": float(per_class_f1[1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=[0, 1]).tolist(),
    }


def plot_confusion(cm, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    row_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    cmap = LinearSegmentedColormap.from_list("seq_blue", _SEQ_RAMP)

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    ax.imshow(row_norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            dark_cell = row_norm[i, j] > 0.55
            ax.text(j, i, f"{int(cm[i, j]):,}\n{row_norm[i, j]:.1%}",
                    ha="center", va="center", fontsize=11,
                    color="white" if dark_cell else _INK)
    # white gaps between cells
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks([0, 1], labels=[f"pred {l}" for l in LABEL_NAMES],
                  fontsize=9, color=_MUTED)
    ax.set_yticks([0, 1], labels=[f"true {l}" for l in LABEL_NAMES],
                  fontsize=9, color=_MUTED, rotation=90, va="center")
    ax.set_title(title, fontsize=10, color=_INK, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    ensure_dirs(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def record_results(model_id: str, train_variant: str, test_set: str,
                   y_true, y_pred, tag: str = "full", seed: int = 42,
                   make_figure: bool = True) -> dict:
    """Compute metrics, write the JSON report (+ optional figure), and
    upsert the row in outputs/results/results_summary.csv.
    Row key: (model, train_variant, test_set, seed)."""
    ensure_dirs(RESULTS_DIR, FIGURES_DIR)
    metrics = compute_all(y_true, y_pred)
    slug = f"{model_id}_{train_variant}_s{seed}_{test_set}"

    report = {
        "model": model_id,
        "train_variant": train_variant,
        "test_set": test_set,
        "seed": int(seed),
        "tag": tag,
        "labels": LABEL_NAMES,
        **metrics,
    }
    save_json(report, RESULTS_DIR / f"classification_report_{slug}.json")

    # One confusion matrix per configuration (reference seed), not per seed.
    if make_figure and seed == 42:
        plot_confusion(
            metrics["confusion_matrix"],
            FIGURES_DIR / f"confusion_matrix_{model_id}_{train_variant}_"
                          f"{test_set}.png",
            f"{model_id} ({train_variant} train) on {test_set}\n"
            f"macro-F1 {metrics['f1_macro']:.3f}",
        )

    row = {k: v for k, v in report.items()
           if k not in ("confusion_matrix", "labels")}
    row["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    summary_path = RESULTS_DIR / "results_summary.csv"
    # Lockfile keeps the read-modify-write safe when two runner lanes
    # (parallel GPUs/processes) finish jobs at the same moment.
    with _summary_lock():
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            if "seed" not in summary.columns:
                summary["seed"] = 42  # legacy rows predate the seed column
            key = (summary["model"] == model_id) & \
                  (summary["train_variant"] == train_variant) & \
                  (summary["test_set"] == test_set) & \
                  (summary["seed"] == int(seed))
            summary = summary[~key]
            summary = pd.concat([summary, pd.DataFrame([row])],
                                ignore_index=True)
        else:
            summary = pd.DataFrame([row])
        summary.sort_values(["model", "train_variant", "test_set", "seed"],
                            inplace=True)
        summary.to_csv(summary_path, index=False, encoding="utf-8")

    print(f"    [{slug}] acc={metrics['accuracy']:.4f} "
          f"macroF1={metrics['f1_macro']:.4f} "
          f"wF1={metrics['f1_weighted']:.4f} (n={metrics['n']})")
    return metrics
