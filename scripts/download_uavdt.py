"""
download_uavdt.py

Downloads the UAVDT (UAV Detection and Tracking) dataset from its official
Zenodo release and extracts it to the expected local directory.

Official source:
https://zenodo.org/records/14575517
"""

import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_API_URL = "https://zenodo.org/api/records/14575517"
UAVDT_DIR = Path("data/raw/uavdt")


def _is_populated(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _download_file(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=dest.name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))


def main() -> int:
    if _is_populated(UAVDT_DIR):
        print(f"UAVDT dataset already present at {UAVDT_DIR.resolve()} — skipping download.")
        return 0

    UAVDT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching UAVDT file list from Zenodo...")
    resp = requests.get(ZENODO_API_URL, timeout=15)
    resp.raise_for_status()
    record = resp.json()
    files = record.get("files", [])

    if not files:
        print("No files found in Zenodo record. Aborting.")
        return 1

    for file_info in files:
        filename = file_info["key"]
        download_url = file_info["links"]["self"]
        dest = UAVDT_DIR / filename

        print(f"\nDownloading {filename} (~{file_info.get('size', 0) / 1e9:.1f} GB)...")
        _download_file(download_url, dest)

        if filename.endswith(".zip"):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(dest, "r") as zf:
                zf.extractall(UAVDT_DIR)
            dest.unlink()
            print("Extraction complete. Archive removed.")

    print(f"\nUAVDT dataset ready at {UAVDT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
