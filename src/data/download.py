"""Download all directly-accessible public Bangla hate speech datasets.

Raw files are stored unchanged under data/raw/<dataset>/ (archives kept
alongside their extracted contents). Datasets that cannot be fetched
automatically are recorded in the status file with manual instructions
(see DATASETS.md).

Usage:
    python -m src.data.download            # download everything missing
    python -m src.data.download --force    # re-download
    python -m src.data.download --only bdshs
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import requests

from src.utils.common import DATA_RAW, ensure_dirs, save_json, setup_utf8_stdout

USER_AGENT = "bangla-hs-research/0.1 (dataset download script)"
MENDELEY_BIDWESH_API = (
    "https://data.mendeley.com/public-api/datasets/bpkrvf882k/files"
    "?folder_id=root&version=1"
)

DATASETS = {
    "bdshs": {
        "description": "BD-SHS: Bangla hate speech dataset (Romim et al., LREC 2022)",
        "kind": "kaggle",
        "url": "https://www.kaggle.com/api/v1/datasets/download/naurosromim/bdshs",
        "archive": "bdshs.zip",
    },
    "bidwesh": {
        "description": "BIDWESH: Bangla regional dialect hate speech (Fayaz et al., 2025)",
        "kind": "mendeley",
        "url": MENDELEY_BIDWESH_API,
    },
    "bengali_hs_30k": {
        "description": "Bengali Hate Speech Dataset ~30K (Romim et al., 2021)",
        "kind": "kaggle",
        "url": (
            "https://www.kaggle.com/api/v1/datasets/download/"
            "naurosromim/bengali-hate-speech-dataset"
        ),
        "archive": "bengali-hate-speech-dataset.zip",
    },
    "bn_hate_speech": {
        "description": "Bengali Hate Speech (Karim et al., 2020) via Hugging Face",
        "kind": "file",
        "url": (
            "https://huggingface.co/datasets/rezacsedu/bn_hate_speech/"
            "resolve/main/data/train-00000-of-00001.parquet"
        ),
        "filename": "bn_hate_speech_train.parquet",
    },
    # HS-BAN (arXiv:2112.01902) has no official public download link.
    # Recorded here so the status file documents it explicitly.
    "hs_ban": {
        "description": "HS-BAN benchmark (Romim et al., 2021) — no public download",
        "kind": "manual",
        "url": "https://arxiv.org/abs/2112.01902",
        "manual": (
            "No official public link exists (paper states 'will make the "
            "dataset available'). Contact the authors, or see DATASETS.md."
        ),
    },
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_download(url: str, dest: Path, max_retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(
                url, stream=True, timeout=120, headers={"User-Agent": USER_AGENT}
            ) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 - retry any transport error
            if attempt == max_retries:
                raise
            print(f"  retry {attempt}/{max_retries} after error: {e}")
            time.sleep(3 * attempt)


def _curl_exe() -> str | None:
    return shutil.which("curl")


def curl_download(url: str, dest: Path) -> None:
    """Fallback for hosts whose WAF rejects Python's TLS fingerprint
    (e.g. data.mendeley.com behind Cloudflare). System curl uses the
    OS TLS stack and passes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(
        [_curl_exe(), "-sSL", "--fail", "--retry", "3", "-o", str(tmp), url],
        check=True,
    )
    tmp.replace(dest)


def curl_get_json(url: str):
    out = subprocess.run(
        [_curl_exe(), "-sSL", "--fail", "--retry", "3", url],
        check=True,
        capture_output=True,
    )
    return json.loads(out.stdout.decode("utf-8"))


def extract_zip(archive: Path, out_dir: Path) -> list[str]:
    with zipfile.ZipFile(archive) as z:
        z.extractall(out_dir)
        return z.namelist()


def download_kaggle(name: str, spec: dict, force: bool) -> dict:
    out_dir = DATA_RAW / name
    archive = out_dir / spec["archive"]
    if archive.exists() and not force:
        print(f"  {archive.name} already present, skipping download")
    else:
        print(f"  fetching {spec['url']}")
        http_download(spec["url"], archive)
    members = extract_zip(archive, out_dir)
    return {"archive": archive.name, "extracted": members}


def download_mendeley_bidwesh(name: str, spec: dict, force: bool) -> dict:
    out_dir = DATA_RAW / name
    ensure_dirs(out_dir)
    use_curl = False
    try:
        listing = requests.get(
            spec["url"], timeout=60, headers={"User-Agent": USER_AGENT}
        )
        listing.raise_for_status()
        items = listing.json()
    except requests.RequestException:
        if not _curl_exe():
            raise
        print("  requests blocked by WAF; falling back to system curl")
        items = curl_get_json(spec["url"])
        use_curl = True
    files = []
    for item in items:
        fname = item["filename"]
        dest = out_dir / fname
        if dest.exists() and not force:
            print(f"  {fname} already present, skipping")
        else:
            print(f"  fetching {fname} ({item.get('size', '?')} bytes)")
            url = item["content_details"]["download_url"]
            if use_curl:
                curl_download(url, dest)
            else:
                http_download(url, dest)
        files.append(fname)
    return {"extracted": files}


def download_file(name: str, spec: dict, force: bool) -> dict:
    out_dir = DATA_RAW / name
    dest = out_dir / spec["filename"]
    if dest.exists() and not force:
        print(f"  {dest.name} already present, skipping")
    else:
        print(f"  fetching {spec['url']}")
        http_download(spec["url"], dest)
    return {"extracted": [dest.name]}


def preview_tabular_files(dataset_dir: Path) -> list[dict]:
    """Print columns/row-counts for every CSV/XLSX/parquet in a raw dir."""
    import pandas as pd

    previews = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.suffix.lower() not in {".csv", ".parquet", ".xlsx"}:
            continue
        try:
            if path.suffix.lower() == ".csv":
                from src.utils.common import read_csv_any

                df = read_csv_any(path)
            elif path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_excel(path)
        except Exception as e:  # noqa: BLE001
            print(f"    !! could not read {path.name}: {e}")
            continue
        info = {
            "file": str(path.relative_to(DATA_RAW)),
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
        }
        previews.append(info)
        print(f"    {info['file']}: {info['rows']} rows, columns={info['columns']}")
    return previews


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download")
    parser.add_argument("--only", choices=sorted(DATASETS), help="one dataset")
    args = parser.parse_args()

    ensure_dirs(DATA_RAW)
    status_path = DATA_RAW / "download_status.json"
    status = {}
    if args.only and status_path.exists():
        from src.utils.common import load_json

        status = load_json(status_path)  # merge into existing statuses
    for name, spec in DATASETS.items():
        if args.only and name != args.only:
            continue
        print(f"[{name}] {spec['description']}")
        entry = {"source_url": spec["url"], "kind": spec["kind"]}
        if spec["kind"] == "manual":
            entry["status"] = "manual_required"
            entry["instructions"] = spec["manual"]
            print(f"  MANUAL: {spec['manual']}")
        else:
            try:
                if spec["kind"] == "kaggle":
                    result = download_kaggle(name, spec, args.force)
                elif spec["kind"] == "mendeley":
                    result = download_mendeley_bidwesh(name, spec, args.force)
                else:
                    result = download_file(name, spec, args.force)
                entry.update(result)
                entry["status"] = "ok"
                entry["files"] = {}
                for p in sorted((DATA_RAW / name).rglob("*")):
                    if p.is_file() and not p.name.endswith(".part"):
                        entry["files"][str(p.relative_to(DATA_RAW / name))] = {
                            "bytes": p.stat().st_size,
                            "sha256": sha256_of(p),
                        }
                entry["preview"] = preview_tabular_files(DATA_RAW / name)
            except Exception as e:  # noqa: BLE001
                entry["status"] = "failed"
                entry["error"] = str(e)
                print(f"  FAILED: {e}")
        status[name] = entry

    save_json(status, DATA_RAW / "download_status.json")
    print(f"\nStatus written to {DATA_RAW / 'download_status.json'}")
    counts = {}
    for entry in status.values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    print("Summary:", counts)


if __name__ == "__main__":
    main()
