"""Publication figures from the aggregated result tables.

- robustness_bars_<model>.png   macro-F1 by test set x train variant,
                                95% CI whiskers (stats_summary.csv)
- adaptation_curve.png          few-shot dialect adaptation data-efficiency
                                curve with seed band (results_summary.csv);
                                skipped until adapt runs exist
- faithfulness_bars.png         comprehensiveness / sufficiency /
                                deletion-AUC by method (faithfulness_summary)

Usage: python -m src.evaluation.figures [--models tfidf_lr,banglabert]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.common import FIGURES_DIR, RESULTS_DIR, ensure_dirs, setup_utf8_stdout

# Fixed categorical assignment (train variants), light-mode palette.
VARIANT_COLOR = {
    "clean": "#2a78d6",      # blue
    "augmented": "#1baf7a",  # aqua
    "dialect": "#eda100",    # yellow
    "aug_dia": "#008300",    # green
    "nct": "#4a3aa7",        # violet
}
VARIANT_LABEL = {
    "clean": "clean", "augmented": "synthetic aug",
    "dialect": "DIA", "aug_dia": "synthetic+DIA", "nct": "NCT",
}
TEST_LABEL = {"clean": "clean test", "augmented": "synthetic-noisy",
              "bidwesh_heldout": "BIDWESH held-out"}
_INK, _MUTED = "#1f2429", "#5f6b76"


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.yaxis.grid(True, color="#eceae6", linewidth=0.8)
    ax.set_axisbelow(True)


def robustness_bars(stats: pd.DataFrame, model: str) -> None:
    sub = stats[(stats["model"] == model)
                & stats["test_set"].isin(TEST_LABEL)]
    variants = [v for v in VARIANT_COLOR if v in
                set(sub["train_variant"])]
    if not variants or sub.empty:
        return
    tests = [t for t in TEST_LABEL if t in set(sub["test_set"])]
    x = np.arange(len(tests))
    width = min(0.8 / len(variants), 0.22)

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    for i, v in enumerate(variants):
        rows = sub[sub["train_variant"] == v].set_index("test_set")
        y = [rows.loc[t, "f1_macro_mean"] if t in rows.index else np.nan
             for t in tests]
        lo = [rows.loc[t, "f1_ci95_lo"] if t in rows.index else np.nan
              for t in tests]
        hi = [rows.loc[t, "f1_ci95_hi"] if t in rows.index else np.nan
              for t in tests]
        pos = x + (i - (len(variants) - 1) / 2) * width
        ax.bar(pos, y, width * 0.92, color=VARIANT_COLOR[v],
               label=VARIANT_LABEL[v], zorder=3)
        ax.errorbar(pos, y,
                    yerr=[np.array(y) - np.array(lo),
                          np.array(hi) - np.array(y)],
                    fmt="none", ecolor=_INK, elinewidth=1, capsize=2,
                    zorder=4)
    ax.set_xticks(x, [TEST_LABEL[t] for t in tests], color=_MUTED)
    ax.set_ylim(bottom=max(0.0, sub["f1_ci95_lo"].min() - 0.03))
    ax.set_ylabel("macro-F1", color=_MUTED, fontsize=9)
    ax.set_title(f"{model}: robustness by training variant "
                 f"(95% bootstrap CI)", fontsize=10, color=_INK)
    ax.legend(frameon=False, fontsize=8, ncol=len(variants),
              loc="upper center", bbox_to_anchor=(0.5, -0.10))
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"robustness_bars_{model}.png",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  robustness_bars_{model}.png")


def adaptation_curve() -> None:
    res = pd.read_csv(RESULTS_DIR / "results_summary.csv")
    res = res[res["tag"] == "full"]
    adapt = res[res["model"].str.contains("-adapt", na=False)].copy()
    if adapt.empty:
        print("  adaptation_curve: no adapt runs yet, skipped")
        return
    adapt["n"] = (adapt["model"].str.extract(r"-adapt(\w+)$")[0]
                  .replace({"full": "3605"}).astype(int))
    base_model = adapt["model"].str.replace(r"-adapt\w+$", "", regex=True) \
        .iloc[0]
    base = res[(res["model"] == base_model)
               & (res["train_variant"] == adapt["train_variant"].iloc[0])]

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=150)
    for test_set, color in (("bidwesh_heldout", "#2a78d6"),
                            ("clean", "#1baf7a")):
        grp = (adapt[adapt["test_set"] == test_set]
               .groupby("n")["f1_macro"].agg(["mean", "min", "max"])
               .sort_index())
        if grp.empty:
            continue
        ax.plot(grp.index, grp["mean"], marker="o", markersize=4,
                linewidth=2, color=color, label=TEST_LABEL[test_set],
                zorder=3)
        ax.fill_between(grp.index, grp["min"], grp["max"], color=color,
                        alpha=0.15, linewidth=0, zorder=2)
        b = base[base["test_set"] == test_set]["f1_macro"]
        if len(b):
            ax.axhline(b.mean(), color=color, linewidth=1,
                       linestyle=(0, (4, 3)), alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks([250, 500, 1000, 3605],
                  ["250", "500", "1000", "full\n(3605)"])
    ax.set_xlabel("BIDWESH adaptation examples", color=_MUTED, fontsize=9)
    ax.set_ylabel("macro-F1", color=_MUTED, fontsize=9)
    ax.set_title("Few-shot dialect adaptation (dashed = no adaptation; "
                 "band = seed min–max)", fontsize=10, color=_INK)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "adaptation_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("  adaptation_curve.png")


def faithfulness_bars() -> None:
    path = RESULTS_DIR / "faithfulness_summary.csv"
    if not path.exists():
        print("  faithfulness_bars: no data yet, skipped")
        return
    df = pd.read_csv(path)
    metrics = [("comprehensiveness_mean", "comprehensiveness ↑"),
               ("sufficiency_mean", "sufficiency ↓"),
               ("deletion_auc_mean", "deletion AUC ↓")]
    method_color = {"lime": "#2a78d6", "shap": "#1baf7a",
                    "random": "#9a988f"}
    models = df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.4),
                             dpi=150, squeeze=False)
    for ax, model in zip(axes[0], models):
        sub = df[df["model"] == model]
        x = np.arange(len(metrics))
        methods = [m for m in method_color if m in set(sub["method"])]
        width = min(0.8 / max(len(methods), 1), 0.25)
        for i, meth in enumerate(methods):
            row = sub[sub["method"] == meth].iloc[0]
            y = [row[m] for m, _ in metrics]
            pos = x + (i - (len(methods) - 1) / 2) * width
            ax.bar(pos, y, width * 0.9, color=method_color[meth],
                   label=meth.upper() if meth != "random" else "random",
                   zorder=3)
        ax.set_xticks(x, [lbl for _, lbl in metrics], fontsize=8,
                      color=_MUTED)
        ax.axhline(0, color="#d8d7d2", linewidth=0.8)
        ax.set_title(f"{model} ({sub['n_examples'].iloc[0]} examples)",
                     fontsize=9, color=_INK)
        ax.legend(frameon=False, fontsize=8)
        _style_axes(ax)
    fig.suptitle("Explanation faithfulness (higher comprehensiveness, "
                 "lower sufficiency/deletion-AUC = more faithful)",
                 fontsize=10, color=_INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "faithfulness_bars.png", bbox_inches="tight")
    plt.close(fig)
    print("  faithfulness_bars.png")


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None)
    args = parser.parse_args()
    ensure_dirs(FIGURES_DIR)

    stats = pd.read_csv(RESULTS_DIR / "stats_summary.csv")
    models = (args.models.split(",") if args.models
              else [m for m in stats["model"].unique()
                    if "smoke" not in m and m != "majority"
                    and "-adapt" not in m])
    print("figures:")
    for m in models:
        robustness_bars(stats, m)
    adaptation_curve()
    faithfulness_bars()


if __name__ == "__main__":
    main()
