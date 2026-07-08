"""Resumable GPU experiment queue for Colab/Kaggle.

Reads scripts/experiments_manifest.json (ordered by priority), skips jobs
already marked done in the state dir (Google Drive), runs the next ones as
subprocesses, and mirrors all small artifacts (results, predictions,
figures, reports) plus flagged model checkpoints back to the state dir
after every job. Re-run the same cell across sessions until it prints
ALL DONE. Interrupting mid-job is safe: the job simply reruns next time.

Colab cell:
    from google.colab import drive; drive.mount('/content/drive')
    !git clone https://github.com/Nafis878/ICCIT-1.git 2>/dev/null; cd ICCIT-1 && git pull
    %cd ICCIT-1
    !pip install -q lime accelerate sentencepiece bitsandbytes
    !python -m src.data.download && python -m src.data.preprocess && python -m src.data.augment
    !python scripts/colab_runner.py --state-dir /content/drive/MyDrive/iccit_q1_state --time-budget-min 200

Local smoke test:
    python scripts/colab_runner.py --state-dir outputs/runner_smoke_state \
        --manifest scripts/smoke_manifest.json --local-smoke
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS / "models"
MIRROR_DIRS = ["outputs/results", "outputs/predictions", "outputs/figures",
               "reports"]


def job_command(job: dict, smoke: bool) -> list[str]:
    py = [sys.executable, "-m"]
    smoke_flag = ["--smoke"] if smoke else []
    kind = job["kind"]
    if kind == "train":
        cmd = py + ["src.models.train_transformer",
                    "--model-name", job["model_name"],
                    "--train-file", f"data/processed/{job['train_file']}",
                    "--method", job.get("method", "standard"),
                    "--seed", str(job["seed"]),
                    "--batch-size", str(job.get("batch_size", 16)),
                    "--epochs", str(job.get("epochs", 3)),
                    "--lr", str(job.get("lr", 2e-5)),
                    "--max-len", str(job.get("max_len", 128))]
        return cmd + smoke_flag
    if kind == "adapt":
        return py + ["src.models.adapt",
                     "--from-dir", str(MODELS_DIR / job["from"]),
                     "--n", str(job["n"]),
                     "--seed", str(job["seed"])] + smoke_flag
    if kind == "llm":
        cmd = py + ["src.models.llm_baseline"]
        if smoke:
            cmd += ["--smoke", "--no-quant",
                    "--model-name", "Qwen/Qwen2.5-0.5B-Instruct"]
        return cmd
    if kind == "faithfulness":
        return py + ["src.explainability.faithfulness",
                     "--model-dir", str(MODELS_DIR / job["model_dir"]),
                     "--method", job["method"],
                     "--num-examples", "20" if smoke else
                     str(job.get("num_examples", 200)),
                     "--num-samples", "50" if smoke else
                     str(job.get("num_samples", 500))]
    if kind == "explain":
        return py + ["src.explainability.explain",
                     "--model-dir", str(MODELS_DIR / job["model_dir"]),
                     "--num-samples", "50" if smoke else "500"]
    raise ValueError(kind)


def mirror_outputs(state: Path) -> None:
    for rel in MIRROR_DIRS:
        src = PROJECT_ROOT / rel
        if not src.exists():
            continue
        dst = state / "mirror" / rel
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.rglob("*"):
            if p.is_file():
                q = dst / p.relative_to(src)
                q.parent.mkdir(parents=True, exist_ok=True)
                if (not q.exists()
                        or p.stat().st_mtime > q.stat().st_mtime):
                    shutil.copy2(p, q)


def persist_model(job: dict, state: Path) -> None:
    """Copy a checkpoint dir to the state dir (needed by later jobs)."""
    for d in MODELS_DIR.glob(job.get("persist_glob", "")):
        dst = state / "models" / d.name
        if dst.exists():
            continue
        print(f"  persisting {d.name} to state dir")
        shutil.copytree(d, dst)


def restore_models(state: Path) -> None:
    src = state / "models"
    if not src.exists():
        return
    for d in src.iterdir():
        dst = MODELS_DIR / d.name
        if not dst.exists():
            print(f"restoring checkpoint {d.name} from state dir")
            shutil.copytree(d, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "scripts" /
                        "experiments_manifest.json")
    parser.add_argument("--time-budget-min", type=float, default=200,
                        help="stop starting new jobs after this many "
                             "minutes (finish + mirror first)")
    parser.add_argument("--local-smoke", action="store_true")
    args = parser.parse_args()

    state = args.state_dir
    (state / "done").mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    restore_models(state)

    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))["jobs"]
    t0 = time.time()
    n_done_before = sum(
        1 for j in jobs if (state / "done" / f"{j['id']}.json").exists())
    print(f"queue: {len(jobs)} jobs, {n_done_before} already done")

    ran, failed = 0, 0
    for job in jobs:
        marker = state / "done" / f"{job['id']}.json"
        if marker.exists():
            continue
        elapsed_min = (time.time() - t0) / 60
        if elapsed_min > args.time_budget_min:
            print(f"time budget reached ({elapsed_min:.0f} min) — "
                  f"re-run this cell in the next session")
            break
        # dependency check for adapt/faithfulness jobs
        dep = job.get("from") or job.get("model_dir")
        if dep and not (MODELS_DIR / dep).exists():
            print(f"[{job['id']}] SKIP — dependency {dep} not trained yet")
            continue

        cmd = job_command(job, args.local_smoke)
        print(f"\n[{job['id']}] {' '.join(cmd[2:])}")
        t_job = time.time()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        mins = (time.time() - t_job) / 60
        if proc.returncode == 0:
            marker.write_text(json.dumps(
                {"id": job["id"], "minutes": round(mins, 1)}),
                encoding="utf-8")
            ran += 1
            if job.get("persist_glob"):
                persist_model(job, state)
        else:
            failed += 1
            print(f"[{job['id']}] FAILED (exit {proc.returncode}) — "
                  f"will retry next run")
        mirror_outputs(state)

    mirror_outputs(state)
    done_now = sum(
        1 for j in jobs if (state / "done" / f"{j['id']}.json").exists())
    print(f"\nsession: ran {ran}, failed {failed}; "
          f"total done {done_now}/{len(jobs)}")
    if done_now == len(jobs):
        print("=" * 40)
        print("ALL DONE — zip the mirror and hand it back:")
        print(f"  cd {state} && zip -r q1_results.zip mirror")
        print("=" * 40)


if __name__ == "__main__":
    main()
