"""Rebuild outputs/results/results_summary.csv from the per-run
classification_report_*.json files (each report is the source of truth
for one model x variant x test-set x seed cell).

Useful when a summary CSV is lost or goes stale (e.g. the Drive-FUSE
mtime issue that hit the first Colab batch).

Usage: python -m src.evaluation.rebuild_summary
"""
import json

import pandas as pd

from src.utils.common import RESULTS_DIR, setup_utf8_stdout


def main() -> None:
    setup_utf8_stdout()
    rows = []
    for p in sorted(RESULTS_DIR.glob("classification_report_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows.append({k: v for k, v in r.items()
                     if k not in ("confusion_matrix", "labels")})
    df = pd.DataFrame(rows)
    df["seed"] = df.get("seed", 42)
    df = df.drop_duplicates(
        subset=["model", "train_variant", "test_set", "seed"], keep="last")
    df.sort_values(["model", "train_variant", "test_set", "seed"],
                   inplace=True)
    df.to_csv(RESULTS_DIR / "results_summary.csv", index=False,
              encoding="utf-8")
    full = df[df["tag"] == "full"]
    print(f"rebuilt results_summary.csv: {len(df)} rows from "
          f"{len(rows)} reports ({len(full)} full rows, "
          f"{full.groupby(['model', 'train_variant']).ngroups} configs)")


if __name__ == "__main__":
    main()
