"""
utils.py

Shared utility functions used across all dataset processors.

Covers:
    - directory management (create, safe overwrite)
    - image dimension reading via Pillow
    - bounding-box conversion (absolute pixels -> YOLO normalised)
    - YOLO geometry validation
    - label file writing
    - image copying
    - JSON / CSV report writing
    - image-to-label file discovery helpers
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

# Image extensions considered valid across both datasets
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    """Create *path* (and all parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def safe_clear_dir(path: Path) -> None:
    """Delete *path* and re-create it as an empty directory.

    Safe in the sense that it only operates on the specific path given — it
    will not traverse upward or delete sibling directories.

    Args:
        path: Directory to wipe and recreate.
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def get_image_dims(image_path: Path) -> tuple[int, int]:
    """Return (width, height) in pixels for the image at *image_path*.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        (width, height) as integers.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If Pillow cannot identify / open the file as an image.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot read image dimensions for {image_path}: {exc}") from exc


def copy_image(src: Path, dst: Path) -> None:
    """Copy image file *src* to *dst*, preserving metadata.

    Args:
        src: Source image path.
        dst: Destination image path (parent directory must exist).
    """
    shutil.copy2(src, dst)


def find_image_for_label(label_stem: str, images_dir: Path) -> Optional[Path]:
    """Search *images_dir* for an image whose stem matches *label_stem*.

    Tries all recognised image extensions in order.

    Args:
        label_stem: Filename without extension (e.g. "0000001_00001_d_0000001").
        images_dir: Directory to search.

    Returns:
        Path to the matching image, or None if not found.
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{label_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Bounding-box conversion
# ---------------------------------------------------------------------------

def abs_to_yolo(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Convert absolute pixel bounding box to YOLO normalised format.

    Args:
        x:     Left edge in pixels (top-left corner).
        y:     Top edge in pixels (top-left corner).
        w:     Box width in pixels.
        h:     Box height in pixels.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        (cx, cy, nw, nh) all normalised to [0, 1].
    """
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def validate_yolo_box(cx: float, cy: float, w: float, h: float) -> bool:
    """Return True if the YOLO box values are geometrically valid.

    A valid box satisfies:
    - w > 0 and h > 0
    - all four values are in the range (0, 1] for centres / dims
      (centres must be > 0 and <= 1; widths / heights must be > 0 and <= 1)

    In practice we check that cx and cy are in (0, 1] and that w and h are
    in (0, 1].  Boxes that touch the very edge (value == 1.0) are accepted.

    Args:
        cx: Normalised x-centre.
        cy: Normalised y-centre.
        w:  Normalised width.
        h:  Normalised height.

    Returns:
        True if valid, False otherwise.
    """
    if w <= 0 or h <= 0:
        return False
    if not (0.0 < cx <= 1.0 and 0.0 < cy <= 1.0):
        return False
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return False
    return True


# ---------------------------------------------------------------------------
# Label file writing
# ---------------------------------------------------------------------------

def write_label_file(path: Path, annotations: list[str]) -> None:
    """Write YOLO annotation lines to *path*.

    If *annotations* is empty an empty file is written — this is the correct
    representation of a negative (no-object) image for YOLO training.

    Args:
        path:        Destination .txt file path.
        annotations: List of YOLO annotation strings
                     (e.g. "0 0.500 0.500 0.100 0.200").
    """
    with open(path, "w", encoding="utf-8") as f:
        if annotations:
            f.write("\n".join(annotations) + "\n")
        # Empty file for negative images — write nothing


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_json_report(path: Path, data: dict) -> None:
    """Write *data* as indented JSON to *path*.

    Args:
        path: Destination .json file path.
        data: Serialisable dict.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv_report(path: Path, data: dict) -> None:
    """Write a flattened key-value CSV from *data* to *path*.

    Nested dicts are flattened using dot-notation keys
    (e.g. ``{"splits": {"train": {"images": 5}}}`` → ``splits.train.images, 5``).

    Args:
        path: Destination .csv file path.
        data: Dict to serialise (may contain nested dicts and plain values).
    """
    rows = _flatten_dict(data)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in rows.items():
            writer.writerow([key, value])


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, object]:
    """Recursively flatten a nested dict into dot-separated key-value pairs."""
    result: dict[str, object] = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            result.update(_flatten_dict(v, full_key))
        else:
            result[full_key] = v
    return result


# ---------------------------------------------------------------------------
# YAML generation helper
# ---------------------------------------------------------------------------

def write_yaml(path: Path, content: dict) -> None:
    """Write a YOLO dataset YAML file to *path*.

    Uses PyYAML with default_flow_style=False for human-readable output.

    Args:
        path:    Destination .yaml file path.
        content: Dict to serialise as YAML.
    """
    import yaml  # imported here to isolate the dependency

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def assert_dir_exists(path: Path, label: str = "") -> None:
    """Raise FileNotFoundError if *path* does not exist or is not a directory.

    Args:
        path:  Directory path to check.
        label: Human-readable description for the error message.

    Raises:
        FileNotFoundError: With a descriptive message.
    """
    if not path.exists():
        desc = f" ({label})" if label else ""
        raise FileNotFoundError(
            f"Required directory not found{desc}: {path}\n"
            "Make sure the raw dataset has been downloaded before running the pipeline."
        )
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory but found a file: {path}")


def check_no_collision(names: list[str], context: str = "") -> None:
    """Raise ValueError if *names* contains any duplicates.

    Args:
        names:   List of filenames or strings to check for uniqueness.
        context: Description of where the collision was detected.

    Raises:
        ValueError: Listing the duplicate names.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for n in names:
        if n in seen:
            duplicates.add(n)
        seen.add(n)
    if duplicates:
        ctx = f" [{context}]" if context else ""
        raise ValueError(
            f"Filename collision detected{ctx}: {sorted(duplicates)}\n"
            "All output filenames must be unique. "
            "Ensure dataset-source prefixes are applied correctly."
        )
