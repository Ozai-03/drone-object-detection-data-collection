"""
visdrone_processor.py

Processes the VisDrone2019-DET raw dataset into a YOLO-ready output folder.

Raw dataset layout expected under *raw_root*:
    VisDrone2019-DET-train/
        images/      *.jpg
        annotations/ *.txt   (8-field CSV per line)
    VisDrone2019-DET-val/
        images/
        annotations/
    VisDrone2019-DET-test-dev/
        images/
        annotations/

Annotation format (one bounding box per line):
    x_left, y_top, width, height, score, class_id, truncation, occlusion

Output layout under *out_root*/visdrone_yolo/:
    images/{train,val,test}/   prefixed filenames: visdrone_<stem>.jpg
    labels/{train,val,test}/   prefixed filenames: visdrone_<stem>.txt
    visdrone.yaml
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .class_mapping import FINAL_CLASSES, VISDRONE_MAP, apply_mapping
from .utils import (
    abs_to_yolo,
    assert_dir_exists,
    copy_image,
    ensure_dir,
    find_image_for_label,
    get_image_dims,
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

PREFIX = "visdrone"
DATASET_NAME = "visdrone_yolo"

# Maps logical split name -> raw folder suffix
_SPLIT_DIR_MAP: dict[str, str] = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",  # NOTE: test-dev is the public test split
}

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
        "ignored_region": 0,
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


def discover_splits(raw_root: Path) -> dict[str, tuple[Path, Path]]:
    """Discover VisDrone split directories under *raw_root*.

    Returns:
        Dict mapping split name ("train"/"val"/"test") to
        (images_dir, annotations_dir) Path pairs.

    Raises:
        FileNotFoundError: If any expected split directory or sub-directory is
                           missing.
    """
    splits: dict[str, tuple[Path, Path]] = {}
    for split_name, folder_name in _SPLIT_DIR_MAP.items():
        split_dir = raw_root / folder_name
        images_dir = split_dir / "images"
        ann_dir = split_dir / "annotations"
        assert_dir_exists(split_dir, f"VisDrone {split_name} split")
        assert_dir_exists(images_dir, f"VisDrone {split_name} images")
        assert_dir_exists(ann_dir, f"VisDrone {split_name} annotations")
        splits[split_name] = (images_dir, ann_dir)
    return splits


def validate_visdrone_structure(raw_root: Path) -> None:
    """Validate that the VisDrone raw directory has the expected layout.

    Raises:
        FileNotFoundError: On any missing directory.
        RuntimeError: If a split contains no annotation files.
    """
    assert_dir_exists(raw_root, "VisDrone raw root")
    splits = discover_splits(raw_root)
    for split_name, (images_dir, ann_dir) in splits.items():
        ann_files = list(ann_dir.glob("*.txt"))
        if not ann_files:
            raise RuntimeError(
                f"VisDrone {split_name} annotations directory is empty: {ann_dir}\n"
                "Expected *.txt annotation files. "
                "Has the dataset been fully extracted?"
            )
        img_files = list(images_dir.glob("*"))
        if not img_files:
            raise RuntimeError(
                f"VisDrone {split_name} images directory is empty: {images_dir}"
            )
        logger.info(
            "  VisDrone %s: %d annotation files, %d images",
            split_name,
            len(ann_files),
            len(img_files),
        )


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------


def parse_visdrone_annotation(line: str) -> Optional[dict]:
    """Parse a single VisDrone annotation line into its component fields.

    Expected format (8 comma-separated integers/values):
        x_left, y_top, width, height, score, class_id, truncation, occlusion

    Args:
        line: A single stripped text line from a VisDrone annotation file.

    Returns:
        Dict with keys {x, y, w, h, score, class_id, truncation, occlusion},
        or None if the line is malformed (wrong field count, non-numeric, etc.)
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split(",")
    if len(parts) != 8:
        return None
    try:
        x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        score = int(parts[4])
        class_id = int(parts[5])
        truncation = int(parts[6])
        occlusion = int(parts[7])
    except ValueError:
        return None
    return {
        "x": x, "y": y, "w": w, "h": h,
        "score": score, "class_id": class_id,
        "truncation": truncation, "occlusion": occlusion,
    }


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------


def process_split(
    split: str,
    images_dir: Path,
    ann_dir: Path,
    out_images: Path,
    out_labels: Path,
) -> SplitStats:
    """Process all images and annotations for one dataset split.

    For each annotation file found:
      1. Locate the corresponding image file.
      2. Read image dimensions.
      3. Parse all annotation lines, apply class mapping, convert to YOLO.
      4. Write the YOLO label file (empty if no valid annotations).
      5. Copy the image with a ``visdrone_`` prefix.

    Args:
        split:      Logical split name ("train", "val", "test").
        images_dir: Raw images directory.
        ann_dir:    Raw annotations directory.
        out_images: Output images directory (must already exist).
        out_labels: Output labels directory (must already exist).

    Returns:
        SplitStats with per-split processing counters.
    """
    stats = SplitStats(split=split)
    ann_files = sorted(ann_dir.glob("*.txt"))

    for ann_path in ann_files:
        stem = ann_path.stem
        out_stem = f"{PREFIX}_{stem}"

        # --- Locate image ---
        image_path = find_image_for_label(stem, images_dir)
        if image_path is None:
            logger.warning("  [%s] No image found for annotation: %s", split, stem)
            stats.missing_images += 1
            continue

        # --- Read image dimensions ---
        try:
            img_w, img_h = get_image_dims(image_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("  [%s] Cannot read image dims for %s: %s", split, stem, exc)
            stats.missing_images += 1
            continue

        if img_w <= 0 or img_h <= 0:
            logger.warning(
                "  [%s] Invalid image dimensions (%dx%d) for %s — skipping",
                split, img_w, img_h, stem,
            )
            stats.missing_images += 1
            continue

        # --- Parse annotations ---
        raw_lines = ann_path.read_text(encoding="utf-8").splitlines()
        yolo_annotations: list[str] = []

        for line in raw_lines:
            stats.annotations_read += 1
            parsed = parse_visdrone_annotation(line)

            if parsed is None:
                stats.annotations_dropped += 1
                stats.drop_reasons["malformed_line"] += 1
                continue

            # Class mapping
            raw_cid = parsed["class_id"]
            final_cid = apply_mapping(raw_cid, VISDRONE_MAP)

            if final_cid is None:
                stats.annotations_dropped += 1
                if raw_cid == 0:
                    stats.drop_reasons["ignored_region"] += 1
                elif raw_cid == 11:
                    stats.drop_reasons["unmapped_class"] += 1
                else:
                    stats.drop_reasons["unmapped_class"] += 1
                continue

            # Bounding-box conversion
            cx, cy, nw, nh = abs_to_yolo(
                parsed["x"], parsed["y"], parsed["w"], parsed["h"], img_w, img_h
            )

            if not validate_yolo_box(cx, cy, nw, nh):
                stats.annotations_dropped += 1
                stats.drop_reasons["invalid_geometry"] += 1
                continue

            yolo_annotations.append(
                f"{final_cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
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
    """Process the entire VisDrone dataset and write YOLO outputs + reports.

    Args:
        raw_root:   ``data/raw/visdrone_raw/`` — must contain the three split
                    directories.
        out_root:   ``data/processed/`` — visdrone_yolo/ will be created here.
        report_dir: ``data/processed/reports/`` — reports written here.

    Raises:
        FileNotFoundError: If raw directories are missing.
        RuntimeError: If any split is empty.
    """
    print("\nProcessing VisDrone...")

    # --- Validate raw structure ---
    validate_visdrone_structure(raw_root)
    splits = discover_splits(raw_root)

    # --- Prepare output directories ---
    vd_out = out_root / DATASET_NAME
    for split in splits:
        ensure_dir(vd_out / "images" / split)
        ensure_dir(vd_out / "labels" / split)

    # --- Process each split ---
    all_stats: dict[str, SplitStats] = {}
    for split_name, (images_dir, ann_dir) in splits.items():
        print(f"  Processing split: {split_name}")
        stats = process_split(
            split=split_name,
            images_dir=images_dir,
            ann_dir=ann_dir,
            out_images=vd_out / "images" / split_name,
            out_labels=vd_out / "labels" / split_name,
        )
        all_stats[split_name] = stats

    # --- Print summary ---
    total_imgs = sum(s.images_processed for s in all_stats.values())
    total_kept = sum(s.annotations_kept for s in all_stats.values())
    total_dropped = sum(s.annotations_dropped for s in all_stats.values())
    print(f"  Images processed : {total_imgs}")
    print(f"  Annotations kept : {total_kept}")
    print(f"  Annotations dropped: {total_dropped}")
    print(f"  Output written to: {vd_out}")

    # --- Generate YAML ---
    yaml_path = vd_out / "visdrone.yaml"
    write_yaml(yaml_path, {
        "path": str(vd_out.resolve()),
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
    write_json_report(report_dir / "visdrone_report.json", report_data)
    write_csv_report(report_dir / "visdrone_report.csv", report_data)
    print(f"  Reports written to: {report_dir}")


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def _build_report(all_stats: dict[str, SplitStats]) -> dict:
    """Aggregate per-split stats into a single report dict."""
    total_images = sum(s.images_processed for s in all_stats.values())
    total_read = sum(s.annotations_read for s in all_stats.values())
    total_kept = sum(s.annotations_kept for s in all_stats.values())
    total_dropped = sum(s.annotations_dropped for s in all_stats.values())
    total_missing = sum(s.missing_images for s in all_stats.values())

    # Aggregate drop reasons
    drop_reasons: dict[str, int] = {}
    for s in all_stats.values():
        for k, v in s.drop_reasons.items():
            drop_reasons[k] = drop_reasons.get(k, 0) + v

    # Aggregate per-class counts
    per_class: dict[str, int] = {}
    for s in all_stats.values():
        for cls, cnt in s.per_class_counts.items():
            per_class[cls] = per_class.get(cls, 0) + cnt

    # Per-split breakdown
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

    return {
        "dataset": "visdrone",
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
