"""
combine_datasets.py

Merges the processed VisDrone and UAVDT YOLO datasets into a single combined
dataset.

Both source datasets must already be processed (i.e. their output directories
must exist and be populated) before this module is called.

Merging strategy:
    combined train = visdrone train + uavdt train
    combined val   = visdrone val   + uavdt val
    combined test  = visdrone test  + uavdt test

All filenames are already prefixed ("visdrone_*" / "uavdt_*"), so collisions
are impossible by construction.  The module validates this explicitly and
raises a ValueError if a collision is detected anyway.

Output layout under *combined_out*:
    images/{train,val,test}/
    labels/{train,val,test}/
    combined.yaml
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .class_mapping import FINAL_CLASSES
from .utils import (
    assert_dir_exists,
    check_no_collision,
    copy_image,
    ensure_dir,
    write_json_report,
    write_csv_report,
    write_yaml,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME = "combined_yolo"
_SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SplitStats:
    """Per-split counters for the combined dataset."""
    split: str
    images_copied: int = 0
    labels_copied: int = 0
    from_visdrone: int = 0
    from_uavdt: int = 0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_source(source_root: Path, name: str) -> None:
    """Ensure a processed dataset directory has the expected YOLO layout.

    Args:
        source_root: e.g. ``data/processed/visdrone_yolo/``
        name:        Human-readable name for error messages.

    Raises:
        FileNotFoundError: If any required directory is missing.
        RuntimeError: If any split images directory is empty.
    """
    assert_dir_exists(source_root, f"{name} processed root")
    for split in _SPLITS:
        img_dir = source_root / "images" / split
        lbl_dir = source_root / "labels" / split
        assert_dir_exists(img_dir, f"{name} {split} images")
        assert_dir_exists(lbl_dir, f"{name} {split} labels")

        # Warn (not fail) if a split directory is empty — negative images only
        # splits are technically valid
        img_count = sum(1 for _ in img_dir.iterdir())
        if img_count == 0:
            logger.warning(
                "  %s %s images directory is empty — this split will contribute "
                "nothing to the combined dataset.",
                name, split,
            )


# ---------------------------------------------------------------------------
# Per-split combination
# ---------------------------------------------------------------------------


def combine_split(
    split: str,
    vd_out: Path,
    uavdt_out: Path,
    combined_out: Path,
) -> SplitStats:
    """Copy images and labels from both source datasets into *combined_out* for one split.

    Args:
        split:        Logical split name ("train", "val", "test").
        vd_out:       Processed VisDrone root (``visdrone_yolo/``).
        uavdt_out:    Processed UAVDT root (``uavdt_yolo/``).
        combined_out: Combined output root (``combined_yolo/``).

    Returns:
        SplitStats with copy counts.

    Raises:
        ValueError: If any filename collision is detected between datasets.
    """
    stats = SplitStats(split=split)

    vd_img_dir = vd_out / "images" / split
    vd_lbl_dir = vd_out / "labels" / split
    ua_img_dir = uavdt_out / "images" / split
    ua_lbl_dir = uavdt_out / "labels" / split

    out_img_dir = combined_out / "images" / split
    out_lbl_dir = combined_out / "labels" / split

    # Collect all source image filenames from both datasets
    vd_images = sorted(vd_img_dir.iterdir()) if vd_img_dir.exists() else []
    ua_images = sorted(ua_img_dir.iterdir()) if ua_img_dir.exists() else []

    # Collision check on image filenames before copying anything
    all_image_names = [f.name for f in vd_images] + [f.name for f in ua_images]
    check_no_collision(all_image_names, context=f"combined {split} images")

    # Collision check on label filenames
    vd_labels = sorted(vd_lbl_dir.iterdir()) if vd_lbl_dir.exists() else []
    ua_labels = sorted(ua_lbl_dir.iterdir()) if ua_lbl_dir.exists() else []
    all_label_names = [f.name for f in vd_labels] + [f.name for f in ua_labels]
    check_no_collision(all_label_names, context=f"combined {split} labels")

    # Copy VisDrone images
    for src in vd_images:
        copy_image(src, out_img_dir / src.name)
        stats.images_copied += 1
        stats.from_visdrone += 1

    # Copy VisDrone labels (including empty label files for negative images)
    for src in vd_labels:
        import shutil
        shutil.copy2(src, out_lbl_dir / src.name)
        stats.labels_copied += 1

    # Copy UAVDT images
    for src in ua_images:
        copy_image(src, out_img_dir / src.name)
        stats.images_copied += 1
        stats.from_uavdt += 1

    # Copy UAVDT labels
    for src in ua_labels:
        import shutil
        shutil.copy2(src, out_lbl_dir / src.name)
        stats.labels_copied += 1

    logger.info(
        "  [combined %s] %d images (%d visdrone + %d uavdt)",
        split, stats.images_copied, stats.from_visdrone, stats.from_uavdt,
    )
    return stats


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run(
    vd_out_root: Path,
    uavdt_out_root: Path,
    combined_out_root: Path,
    report_dir: Path,
) -> None:
    """Build the combined YOLO dataset from processed VisDrone and UAVDT outputs.

    Args:
        vd_out_root:      ``data/processed/visdrone_yolo/``
        uavdt_out_root:   ``data/processed/uavdt_yolo/``
        combined_out_root: ``data/processed/combined_yolo/``
        report_dir:       ``data/processed/reports/``

    Raises:
        FileNotFoundError: If either source dataset is missing.
        ValueError: If filename collisions are detected.
    """
    print("\nBuilding combined dataset...")

    # --- Validate sources ---
    _validate_source(vd_out_root, "visdrone_yolo")
    _validate_source(uavdt_out_root, "uavdt_yolo")

    # --- Prepare output directories ---
    for split in _SPLITS:
        ensure_dir(combined_out_root / "images" / split)
        ensure_dir(combined_out_root / "labels" / split)

    # --- Combine each split ---
    all_stats: dict[str, SplitStats] = {}
    for split in _SPLITS:
        print(f"  Combining split: {split}")
        stats = combine_split(
            split=split,
            vd_out=vd_out_root,
            uavdt_out=uavdt_out_root,
            combined_out=combined_out_root,
        )
        all_stats[split] = stats

    # --- Print summary ---
    total_images = sum(s.images_copied for s in all_stats.values())
    total_vd = sum(s.from_visdrone for s in all_stats.values())
    total_ua = sum(s.from_uavdt for s in all_stats.values())
    print(f"  Total images combined : {total_images}")
    print(f"    from VisDrone        : {total_vd}")
    print(f"    from UAVDT           : {total_ua}")
    print(f"  Output written to      : {combined_out_root}")

    # --- Generate YAML ---
    yaml_path = combined_out_root / "combined.yaml"
    write_yaml(yaml_path, {
        "path": str(combined_out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(FINAL_CLASSES),
        "names": FINAL_CLASSES,
    })
    logger.info("  YAML written: %s", yaml_path)

    # --- Generate reports ---
    ensure_dir(report_dir)
    report_data = _build_report(all_stats)
    write_json_report(report_dir / "combined_report.json", report_data)
    write_csv_report(report_dir / "combined_report.csv", report_data)
    print(f"  Reports written to: {report_dir}")


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def _build_report(all_stats: dict[str, SplitStats]) -> dict:
    """Build a combined report dict from per-split stats."""
    splits_data: dict[str, dict] = {}
    for split_name, s in all_stats.items():
        splits_data[split_name] = {
            "images_copied": s.images_copied,
            "labels_copied": s.labels_copied,
            "from_visdrone": s.from_visdrone,
            "from_uavdt": s.from_uavdt,
        }

    return {
        "dataset": "combined",
        "sources": ["visdrone_yolo", "uavdt_yolo"],
        "total_images": sum(s.images_copied for s in all_stats.values()),
        "from_visdrone": sum(s.from_visdrone for s in all_stats.values()),
        "from_uavdt": sum(s.from_uavdt for s in all_stats.values()),
        "splits": splits_data,
    }
