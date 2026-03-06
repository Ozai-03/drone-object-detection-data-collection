"""
download_visdrone.py

Downloads the VisDrone2019-DET dataset splits from Google Drive and extracts
them to the expected local directory.

Official source:
https://github.com/VisDrone/VisDrone-Dataset
"""

import sys
import zipfile
from pathlib import Path

import gdown

VISDRONE_DIR = Path("data/raw/visdrone_raw")

# Google Drive file IDs sourced from the official VisDrone GitHub README
SPLITS = {
    "VisDrone2019-DET-train": "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn",
    "VisDrone2019-DET-val": "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59",
    "VisDrone2019-DET-test-dev": "1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V",
}


def main() -> int:
    VISDRONE_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, file_id in SPLITS.items():
        split_dir = VISDRONE_DIR / split_name
        if split_dir.exists() and any(split_dir.iterdir()):
            print(f"{split_name} already present — skipping.")
            continue

        zip_path = VISDRONE_DIR / f"{split_name}.zip"
        url = f"https://drive.google.com/uc?id={file_id}"

        print(f"\nDownloading {split_name}...")
        gdown.download(url, str(zip_path), quiet=False)

        print(f"Extracting {split_name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            top_level_dirs = {name.split("/")[0] for name in zf.namelist()}
            if split_name in top_level_dirs:
                zf.extractall(VISDRONE_DIR)
            else:
                split_dir.mkdir(exist_ok=True)
                zf.extractall(split_dir)
        zip_path.unlink()
        print(f"{split_name} ready.")

    print(f"\nVisDrone dataset ready at {VISDRONE_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
