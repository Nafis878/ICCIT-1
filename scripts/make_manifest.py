"""Generate scripts/experiments_manifest.json (GPU queue, priority order)
and scripts/smoke_manifest.json (tiny local runner test).

Run: python scripts/make_manifest.py
"""
import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)


def train(model_key, model_name, train_file, seed, *, method="standard",
          batch_size=16, persist=False):
    variant = {"train.csv": "clean", "train_augmented.csv": "augmented",
               "train_dialect_aug.csv": "dialect",
               "train_aug_dia.csv": "aug_dia"}[train_file]
    mid = model_key + ("-nct" if method == "consistency" else "")
    job = {
        "id": f"{mid}_{variant}_s{seed}",
        "kind": "train",
        "model_name": model_name,
        "train_file": train_file,
        "method": method,
        "seed": seed,
        "batch_size": batch_size,
    }
    if persist:
        job["persist_glob"] = f"{mid}_{variant}_s{seed}"
    return job


def main() -> None:
    jobs = []

    bb = "csebuetnlp/banglabert"
    # 1. Core: BanglaBERT matrix (augmented first — adaptation depends on it)
    for s in SEEDS:
        jobs.append(train("banglabert", bb, "train_augmented.csv", s,
                          batch_size=32, persist=True))
    for s in SEEDS:
        jobs.append(train("banglabert", bb, "train.csv", s, batch_size=32))
    for s in SEEDS:
        jobs.append(train("banglabert", bb, "train_aug_dia.csv", s,
                          batch_size=32, persist=(s == 42)))
    for s in SEEDS:
        # NCT does two forwards per step -> half batch
        jobs.append(train("banglabert", bb, "train.csv", s,
                          method="consistency", batch_size=16))
    for s in SEEDS:
        jobs.append(train("banglabert", bb, "train_dialect_aug.csv", s,
                          batch_size=32))

    # 2. Few-shot dialect adaptation curve (short runs; from aug checkpoints)
    for s in SEEDS:
        for n in (250, 500, 1000, 0):  # 0 = full adapt split
            jobs.append({
                "id": f"banglabert-adapt{n or 'full'}_augmented_s{s}",
                "kind": "adapt",
                "from": f"banglabert_augmented_s{s}",
                "n": n,
                "seed": s,
            })

    # 3. Transformer XAI (uses the persisted aug_dia s42 checkpoint)
    best = "banglabert_aug_dia_s42"
    for method in ("lime", "shap", "random"):
        jobs.append({"id": f"faith_banglabert_{method}",
                     "kind": "faithfulness", "model_dir": best,
                     "method": method})
    jobs.append({"id": "explain_banglabert", "kind": "explain",
                 "model_dir": best})

    # 4. Breadth: XLM-R, MuRIL, mBERT
    for s in SEEDS:
        for tf in ("train.csv", "train_augmented.csv", "train_aug_dia.csv"):
            jobs.append(train("xlm-roberta-base", "xlm-roberta-base", tf, s))
    for s in SEEDS:
        for tf in ("train.csv", "train_augmented.csv"):
            jobs.append(train("muril-base-cased", "google/muril-base-cased",
                              tf, s))
    for tf in ("train.csv", "train_augmented.csv"):
        jobs.append(train("bert-base-multilingual-cased",
                          "bert-base-multilingual-cased", tf, 42))

    # 5. LLM baseline
    jobs.append({"id": "llm_qwen25_7b", "kind": "llm"})

    (SCRIPTS / "experiments_manifest.json").write_text(
        json.dumps({"jobs": jobs}, indent=1), encoding="utf-8")
    print(f"experiments_manifest.json: {len(jobs)} jobs")

    # Tiny local smoke manifest (tiny-random model: trivial download/compute
    # and a modern repo layout that works on transformers v5)
    tiny = "hf-internal-testing/tiny-random-ElectraForSequenceClassification"
    tiny_dir = "tiny-random-electraforsequenceclassification_smoke_clean_s42"
    smoke_jobs = [
        train("tiny-random-electraforsequenceclassification", tiny,
              "train.csv", 42),
        {"id": "smoke_adapt", "kind": "adapt",
         "from": tiny_dir, "n": 50, "seed": 42},
        {"id": "smoke_faith", "kind": "faithfulness",
         "model_dir": tiny_dir, "method": "lime"},
    ]
    smoke_jobs[0]["id"] = "smoke_train"
    (SCRIPTS / "smoke_manifest.json").write_text(
        json.dumps({"jobs": smoke_jobs}, indent=1), encoding="utf-8")
    print(f"smoke_manifest.json: {len(smoke_jobs)} jobs")


if __name__ == "__main__":
    main()
