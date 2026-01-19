"""
verify_datasets.py

Purpose:
Verify that required datasets (VisDrone and UAVDT) exist locally in the expected
directory structure, and print basic summary statistics (image and annotation counts).

This script does not download data. It only checks what is already present on disk.

Default expected locations:
- VisDrone: data/raw/visdrone
- UAVDT:    data/raw/uavdt

Usage:
  python scripts/verify_datasets.py
  python scripts/verify_datasets.py --visdrone-dir data/raw/visdrone --uavdt-dir data/raw/uavdt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set, Tuple


IMAGE_EXTS_DEFAULT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANN_EXTS_DEFAULT = {".txt", ".json", ".xml", ".csv"}


@dataclass
class DatasetReport:
    name: str
    root: Path
    exists: bool
    image_count: int = 0
    ann_count: int = 0
    notes: Tuple[str, ...] = ()


def count_files(root: Path, extensions: Set[str]) -> int:
    if not root.exists():
        return 0
    exts = {e.lower() for e in extensions}
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def find_any_dirs(root: Path, candidates: Iterable[str]) -> bool:
    for c in candidates:
        if (root / c).exists():
            return True
    return False


def verify_visdrone(visdrone_root: Path, image_exts: Set[str]) -> DatasetReport:
    report = DatasetReport(name="VisDrone", root=visdrone_root, exists=visdrone_root.exists())
    if not report.exists:
        report.notes = ("Root directory missing.",)
        return report

    # Common VisDrone DET folder names (users may nest under visdrone_root)
    expected_subdirs = (
        "VisDrone2019-DET-train",
        "VisDrone2019-DET-val",
        "VisDrone2019-DET-test-dev",
        "VisDrone2019-DET-test-challenge",
    )

    has_any_expected = find_any_dirs(visdrone_root, expected_subdirs)
    if not has_any_expected:
        report.notes = report.notes + (
            "Did not find standard VisDrone2019-DET-* subfolders under the root. "
            "This may be fine if you organized the dataset differently.",
        )

    report.image_count = count_files(visdrone_root, image_exts)

    # VisDrone annotations are typically .txt files; keep it simple and count .txt under the root.
    report.ann_count = count_files(visdrone_root, {".txt"})

    if report.image_count == 0:
        report.notes = report.notes + ("No image files found under VisDrone root.",)
    if report.ann_count == 0:
        report.notes = report.notes + ("No annotation .txt files found under VisDrone root.",)

    return report


def verify_uavdt(uavdt_root: Path, image_exts: Set[str], ann_exts: Set[str]) -> DatasetReport:
    report = DatasetReport(name="UAVDT", root=uavdt_root, exists=uavdt_root.exists())
    if not report.exists:
        report.notes = ("Root directory missing.",)
        return report

    report.image_count = count_files(uavdt_root, image_exts)
    report.ann_count = count_files(uavdt_root, ann_exts)

    if report.image_count == 0:
        report.notes = report.notes + ("No image files found under UAVDT root.",)
    if report.ann_count == 0:
        report.notes = report.notes + (
            "No annotation files found under UAVDT root (checked: .txt, .json, .xml, .csv). "
        )

    return report


def print_report(report: DatasetReport) -> None:
    print(f"{report.name} Dataset")
    print("-" * (len(report.name) + 8))
    print(f"Root: {report.root.resolve()}")
    print(f"Exists: {report.exists}")
    if report.exists:
        print(f"Images found: {report.image_count}")
        print(f"Annotations found: {report.ann_count}")
    if report.notes:
        print("Notes:")
        for n in report.notes:
            print(f"- {n}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local VisDrone and UAVDT datasets.")
    parser.add_argument("--visdrone-dir", type=Path, default=Path("data/raw/visdrone"))
    parser.add_argument("--uavdt-dir", type=Path, default=Path("data/raw/uavdt"))
    parser.add_argument(
        "--min-samples",
        type=int,
        default=15000,
        help="Minimum combined image count to meet bootcamp dataset size requirement.",
    )
    args = parser.parse_args()

    image_exts = set(IMAGE_EXTS_DEFAULT)
    ann_exts = set(ANN_EXTS_DEFAULT)

    visdrone_report = verify_visdrone(args.visdrone_dir, image_exts=image_exts)
    uavdt_report = verify_uavdt(args.uavdt_dir, image_exts=image_exts, ann_exts=ann_exts)

    print_report(visdrone_report)
    print_report(uavdt_report)

    # Fail if either root is missing
    missing = [r.name for r in (visdrone_report, uavdt_report) if not r.exists]
    if missing:
        print("Verification failed.")
        print("Missing dataset roots: " + ", ".join(missing))
        return 1

    combined_images = visdrone_report.image_count + uavdt_report.image_count
    print("Summary")
    print("-------")
    print(f"Combined images found: {combined_images}")
    print(f"Minimum required samples: {args.min_samples}")
    if combined_images >= args.min_samples:
        print("Minimum sample requirement appears to be satisfied based on image counts.")
        return 0

    print("Minimum sample requirement not satisfied based on image counts.")
    print("If your 'samples' are frames or instances rather than images, adjust checks accordingly.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())