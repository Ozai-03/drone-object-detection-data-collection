"""
uavdt_processor.py

Processes the UAVDT raw dataset (Zenodo record 14575517) into a YOLO-ready
output folder.

Raw dataset layout expected under *raw_root*:
    UAVDT/
        train/
            images/   *.jpg
            labels/   *.txt   (already YOLO normalised)
        val/
            images/
            labels/
        test/
            images/
            labels/

Annotation format (already YOLO normalised, one box per line):
    class_id  cx  cy  w  h      (all floats, space-separated)

The UAVDT dataset is vehicle-focused.  Class IDs are mapped to the final
taxonomy at runtime after scanning all label files.  If unexpected class IDs
are found the pipeline fails loudly — see class_mapping.build_uavdt_map().

Output layout under *out_root*/uavdt_yolo/:
    images/{train,val,test}/   prefixed filenames: uavdt_<stem>.jpg
    labels/{train,val,test}/   prefixed filenames: uavdt_<stem>.txt
    uavdt.yaml

TODO: If the Zenodo release ships a `classes.txt` or `data.yaml`, read it
      to confirm class names instead of relying on build_uavdt_map() alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .class_mapping import FINAL_CLASSES, apply_mapping, build_uavdt_map
from .utils import (
    assert_dir_exists,
    copy_image,
    ensure_dir,
    find_image_for_label,
    validate_yolo_box,
    write_json_report,
    write_csv_report,
    write_label_file,
    write_yaml,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREFIX = "uavdt"
DATASET_NAME = "uavdt_yolo"

# The UAVDT Zenodo release nests data under an UAVDT/ subdirectory
_UAVDT_SUBDIR = "UAVDT"
_SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SplitStats:
    """Per-split processing counters."""
    split: str
    images_processed: int = 0
    images_with_labels: int = 0
    negative_images: int = 0
    missing_images: int = 0
    annotations_read: int = 0
    annotations_kept: int = 0
    annotations_dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=lambda: {
        "unmapped_class": 0,
        "invalid_geometry": 0,
        "malformed_line": 0,
    })
    per_class_counts: dict[str, int] = field(default_factory=lambda: {
        cls: 0 for cls in FINAL_CLASSES
    })


# ---------------------------------------------------------------------------
# Structure discovery and validation
# ---------------------------------------------------------------------------


def _uavdt_data_root(raw_root: Path) -> Path:
    """Return the path to the UAVDT/ subdirectory inside raw_root."""
    return raw_root / _UAVDT_SUBDIR


def discover_splits(raw_root: Path) -> dict[str, tuple[Path, Path]]:
    """Discover UAVDT split directories.

    Args:
        raw_root: ``data/raw/uavdt_raw/`` — must contain UAVDT/.

    Returns:
        Dict mapping split name -> (images_dir, labels_dir).

    Raises:
        FileNotFoundError: If any expected directory is missing.
    """
    data_root = _uavdt_data_root(raw_root)
    assert_dir_exists(data_root, "UAVDT data root (uavdt_raw/UAVDT/)")

    splits: dict[str, tuple[Path, Path]] = {}
    for split_name in _SPLITS:
        split_dir = data_root / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        assert_dir_exists(split_dir, f"UAVDT {split_name} split")
        assert_dir_exists(images_dir, f"UAVDT {split_name} images")
        assert_dir_exists(labels_dir, f"UAVDT {split_name} labels")
        splits[split_name] = (images_dir, labels_dir)
    return splits


def validate_uavdt_structure(raw_root: Path) -> None:
    """Validate that the UAVDT raw directory has the expected layout.

    Raises:
        FileNotFoundError: On any missing directory.
        RuntimeError: If a split contains no label files.
    """
    assert_dir_exists(raw_root, "UAVDT raw root")
    splits = discover_splits(raw_root)
    for split_name, (images_dir, labels_dir) in splits.items():
        label_files = list(labels_dir.glob("*.txt"))
        if not label_files:
            raise RuntimeError(
                f"UAVDT {split_name} labels directory is empty: {labels_dir}\n"
                "Expected *.txt label files. "
                "Has the dataset been fully extracted?"
            )
        img_files = list(images_dir.glob("*"))
        if not img_files:
            raise RuntimeError(
                f"UAVDT {split_name} images directory is empty: {images_dir}"
            )
        logger.info(
            "  UAVDT %s: %d label files, %d images",
            split_name,
            len(label_files),
            len(img_files),
        )


# ---------------------------------------------------------------------------
# Class ID detection
# ---------------------------------------------------------------------------


def detect_uavdt_classes(raw_root: Path) -> set[int]:
    """Scan all UAVDT label files to find the set of unique class IDs present.

    Args:
        raw_root: ``data/raw/uavdt_raw/``.

    Returns:
        Set of integer class IDs found across all splits.

    Raises:
        RuntimeError: If no label files are found (dataset not extracted).
    """
    data_root = _uavdt_data_root(raw_root)
    detected: set[int] = set()
    files_scanned = 0

    for split_name in _SPLITS:
        labels_dir = data_root / split_name / "labels"
        if not labels_dir.exists():
            continue
        for label_file in labels_dir.glob("*.txt"):
            files_scanned += 1
            for line in label_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts:
                    try:
                        detected.add(int(parts[0]))
                    except ValueError:
                        pass  # will be caught during per-file processing

    if files_scanned == 0:
        raise RuntimeError(
            "No UAVDT label files found while scanning for class IDs. "
            "Is the dataset extracted under data/raw/uavdt_raw/UAVDT/?"
        )

    logger.info("  UAVDT class IDs detected: %s (scanned %d files)", sorted(detected), files_scanned)
    return detected


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------


def parse_uavdt_annotation(line: str) -> Optional[dict]:
    """Parse a single UAVDT YOLO annotation line.

    Expected format (5 space-separated values):
        class_id  cx  cy  w  h

    Args:
        line: A single stripped text line from a UAVDT label file.

    Returns:
        Dict with keys {class_id, cx, cy, w, h}, or None if malformed.
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None
    return {"class_id": class_id, "cx": cx, "cy": cy, "w": w, "h": h}


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------


def process_split(
    split: str,
    images_dir: Path,
    labels_dir: Path,
    out_images: Path,
    out_labels: Path,
    class_map: dict[int, int | None],
) -> SplitStats:
    """Process all images and labels for one UAVDT split.

    Since UAVDT labels are already YOLO-normalised, only class remapping and
    geometry validation are needed.  No coordinate conversion is performed.

    Args:
        split:      Logical split name ("train", "val", "test").
        images_dir: Raw images directory.
        labels_dir: Raw labels directory.
        out_images: Output images directory (must already exist).
        out_labels: Output labels directory (must already exist).
        class_map:  Mapping from raw UAVDT class IDs to final class IDs.

    Returns:
        SplitStats with per-split processing counters.
    """
    stats = SplitStats(split=split)
    label_files = sorted(labels_dir.glob("*.txt"))

    for label_path in label_files:
        stem = label_path.stem
        out_stem = f"{PREFIX}_{stem}"

        # --- Locate image ---
        image_path = find_image_for_label(stem, images_dir)
        if image_path is None:
            logger.warning("  [%s] No image found for label: %s", split, stem)
            stats.missing_images += 1
            continue

        # --- Parse and remap annotations ---
        raw_lines = label_path.read_text(encoding="utf-8").splitlines()
        yolo_annotations: list[str] = []

        for line in raw_lines:
            stats.annotations_read += 1
            parsed = parse_uavdt_annotation(line)

            if parsed is None:
                stats.annotations_dropped += 1
                stats.drop_reasons["malformed_line"] += 1
                continue

            # Class remapping
            final_cid = apply_mapping(parsed["class_id"], class_map)
            if final_cid is None:
                stats.annotations_dropped += 1
                stats.drop_reasons["unmapped_class"] += 1
                continue

            # Geometry validation (already normalised — just range-check)
            if not validate_yolo_box(parsed["cx"], parsed["cy"], parsed["w"], parsed["h"]):
                stats.annotations_dropped += 1
                stats.drop_reasons["invalid_geometry"] += 1
                continue

            yolo_annotations.append(
                f"{final_cid} {parsed['cx']:.6f} {parsed['cy']:.6f} "
                f"{parsed['w']:.6f} {parsed['h']:.6f}"
            )
            stats.annotations_kept += 1
            stats.per_class_counts[FINAL_CLASSES[final_cid]] += 1

        # --- Write outputs ---
        out_label_path = out_labels / f"{out_stem}.txt"
        write_label_file(out_label_path, yolo_annotations)

        out_image_path = out_images / f"{out_stem}{image_path.suffix}"
        copy_image(image_path, out_image_path)

        # --- Update counters ---
        stats.images_processed += 1
        if yolo_annotations:
            stats.images_with_labels += 1
        else:
            stats.negative_images += 1

    logger.info(
        "  [%s] images=%d  kept=%d  dropped=%d  negatives=%d",
        split,
        stats.images_processed,
        stats.annotations_kept,
        stats.annotations_dropped,
        stats.negative_images,
    )
    return stats


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run(
    raw_root: Path,
    out_root: Path,
    report_dir: Path,
) -> None:
    """Process the entire UAVDT dataset and write YOLO outputs + reports.

    Args:
        raw_root:   ``data/raw/uavdt_raw/`` — must contain UAVDT/{train,val,test}/.
        out_root:   ``data/processed/`` — uavdt_yolo/ will be created here.
        report_dir: ``data/processed/reports/`` — reports written here.

    Raises:
        FileNotFoundError: If raw directories are missing.
        ValueError: If unexpected UAVDT class IDs are found (unmappable).
        RuntimeError: If any split is empty.
    """
    print("\nProcessing UAVDT...")

    # --- Validate structure ---
    validate_uavdt_structure(raw_root)
    splits = discover_splits(raw_root)

    # --- Detect and validate class IDs ---
    print("  Scanning UAVDT label files for class IDs...")
    detected_classes = detect_uavdt_classes(raw_root)
    print(f"  Detected class IDs: {sorted(detected_classes)}")

    class_map = build_uavdt_map(detected_classes)
    print(f"  Class mapping: { {k: FINAL_CLASSES[v] for k, v in class_map.items() if v is not None} }")

    # --- Prepare output directories ---
    uavdt_out = out_root / DATASET_NAME
    for split in splits:
        ensure_dir(uavdt_out / "images" / split)
        ensure_dir(uavdt_out / "labels" / split)

    # --- Process each split ---
    all_stats: dict[str, SplitStats] = {}
    for split_name, (images_dir, labels_dir) in splits.items():
        print(f"  Processing split: {split_name}")
        stats = process_split(
            split=split_name,
            images_dir=images_dir,
            labels_dir=labels_dir,
            out_images=uavdt_out / "images" / split_name,
            out_labels=uavdt_out / "labels" / split_name,
            class_map=class_map,
        )
        all_stats[split_name] = stats

    # --- Print summary ---
    total_imgs = sum(s.images_processed for s in all_stats.values())
    total_kept = sum(s.annotations_kept for s in all_stats.values())
    total_dropped = sum(s.annotations_dropped for s in all_stats.values())
    print(f"  Images processed   : {total_imgs}")
    print(f"  Annotations kept   : {total_kept}")
    print(f"  Annotations dropped: {total_dropped}")
    print(f"  Output written to  : {uavdt_out}")

    # --- Generate YAML ---
    yaml_path = uavdt_out / "uavdt.yaml"
    write_yaml(yaml_path, {
        "path": str(uavdt_out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(FINAL_CLASSES),
        "names": FINAL_CLASSES,
    })
    logger.info("  YAML written: %s", yaml_path)

    # --- Generate reports ---
    ensure_dir(report_dir)
    report_data = _build_report(all_stats, class_map)
    write_json_report(report_dir / "uavdt_report.json", report_data)
    write_csv_report(report_dir / "uavdt_report.csv", report_data)
    print(f"  Reports written to: {report_dir}")


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def _build_report(
    all_stats: dict[str, SplitStats],
    class_map: dict[int, int | None],
) -> dict:
    """Aggregate per-split stats into a single report dict."""
    total_images = sum(s.images_processed for s in all_stats.values())
    total_read = sum(s.annotations_read for s in all_stats.values())
    total_kept = sum(s.annotations_kept for s in all_stats.values())
    total_dropped = sum(s.annotations_dropped for s in all_stats.values())
    total_missing = sum(s.missing_images for s in all_stats.values())

    drop_reasons: dict[str, int] = {}
    for s in all_stats.values():
        for k, v in s.drop_reasons.items():
            drop_reasons[k] = drop_reasons.get(k, 0) + v

    per_class: dict[str, int] = {}
    for s in all_stats.values():
        for cls, cnt in s.per_class_counts.items():
            per_class[cls] = per_class.get(cls, 0) + cnt

    splits_data: dict[str, dict] = {}
    for split_name, s in all_stats.items():
        splits_data[split_name] = {
            "images_processed": s.images_processed,
            "images_with_labels": s.images_with_labels,
            "negative_images": s.negative_images,
            "missing_images": s.missing_images,
            "annotations_read": s.annotations_read,
            "annotations_kept": s.annotations_kept,
            "annotations_dropped": s.annotations_dropped,
            "drop_reasons": s.drop_reasons,
            "per_class_counts": s.per_class_counts,
        }

    # Serialisable version of class_map (int keys -> string values)
    readable_map = {
        str(k): (FINAL_CLASSES[v] if v is not None else "DROP")
        for k, v in class_map.items()
    }

    return {
        "dataset": "uavdt",
        "uavdt_class_mapping": readable_map,
        "total_images": total_images,
        "total_annotations_read": total_read,
        "annotations_kept": total_kept,
        "annotations_dropped": total_dropped,
        "drop_reasons": drop_reasons,
        "per_class_counts": per_class,
        "images_with_labels": sum(s.images_with_labels for s in all_stats.values()),
        "negative_images": sum(s.negative_images for s in all_stats.values()),
        "missing_images": total_missing,
        "splits": splits_data,
    }
