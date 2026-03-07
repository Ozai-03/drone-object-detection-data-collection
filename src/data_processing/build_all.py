"""
build_all.py

Top-level CLI entry point for the YOLO preprocessing pipeline.

Runs the following steps in order:
    1. Process VisDrone raw data -> data/processed/visdrone_yolo/
    2. Process UAVDT raw data   -> data/processed/uavdt_yolo/
    3. Combine both datasets    -> data/processed/combined_yolo/
    4. Write reports            -> data/processed/reports/
    5. Write class mapping summary

Usage:
    # Full pipeline (first run)
    python -m src.data_processing.build_all

    # Force clean rebuild (wipes existing processed dirs first)
    python -m src.data_processing.build_all --force

    # Custom paths
    python -m src.data_processing.build_all --raw-root data/raw --out-root data/processed --force

    # Skip individual steps (e.g., only redo the combine step)
    python -m src.data_processing.build_all --skip-visdrone --skip-uavdt

    # Only process VisDrone
    python -m src.data_processing.build_all --skip-uavdt --skip-combine
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import visdrone_processor, uavdt_processor, combine_datasets
from .class_mapping import FINAL_CLASSES, VISDRONE_MAP, VEHICLE
from .utils import ensure_dir, safe_clear_dir, write_json_report

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (relative to the project root)
# ---------------------------------------------------------------------------

DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_OUT_ROOT = Path("data/processed")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument list (uses sys.argv if None).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="build_all",
        description="Build YOLO-ready datasets from VisDrone and UAVDT raw data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        metavar="PATH",
        help=(
            "Root directory containing raw datasets "
            "(default: %(default)s). "
            "Expected sub-dirs: visdrone_raw/ and uavdt_raw/."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        metavar="PATH",
        help="Root output directory for processed datasets (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Safely delete and rebuild all processed output directories. "
            "Without this flag, existing outputs are skipped with a warning."
        ),
    )
    parser.add_argument(
        "--skip-visdrone",
        action="store_true",
        help="Skip the VisDrone processing step.",
    )
    parser.add_argument(
        "--skip-uavdt",
        action="store_true",
        help="Skip the UAVDT processing step.",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Skip the dataset combination step.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------


def _prepare_output_dir(path: Path, force: bool, label: str) -> bool:
    """Prepare an output directory for writing.

    Args:
        path:  Target directory.
        force: If True, wipe and recreate. If False, skip if exists.
        label: Human-readable name for console messages.

    Returns:
        True if processing should proceed, False if it should be skipped
        (only possible when force=False and the directory already exists).
    """
    if path.exists() and not force:
        print(
            f"  WARNING: {label} output already exists at {path}. "
            "Pass --force to rebuild from scratch."
        )
        return False
    if force and path.exists():
        print(f"  --force: removing existing {label} output at {path}")
        safe_clear_dir(path)
    else:
        ensure_dir(path)
    return True


# ---------------------------------------------------------------------------
# Class mapping summary
# ---------------------------------------------------------------------------


def _write_class_mapping_summary(report_dir: Path) -> None:
    """Write a human-readable class mapping summary JSON file.

    Covers both VisDrone and UAVDT (assumed) mappings.
    """
    # VisDrone mapping with human-readable names
    vd_class_names = {
        0: "ignored_region",
        1: "pedestrian",
        2: "people",
        3: "bicycle",
        4: "car",
        5: "van",
        6: "truck",
        7: "tricycle",
        8: "awning-tricycle",
        9: "bus",
        10: "motor",
        11: "others",
    }
    visdrone_mapping = {
        vd_class_names[raw_id]: (FINAL_CLASSES[final_id] if final_id is not None else "DROP")
        for raw_id, final_id in VISDRONE_MAP.items()
    }

    uavdt_mapping = {
        "0 (car/vehicle-type)": FINAL_CLASSES[VEHICLE],
        "1 (bus/truck-type)": FINAL_CLASSES[VEHICLE],
        "2 (van/vehicle-type)": FINAL_CLASSES[VEHICLE],
        "_note": (
            "UAVDT class IDs are resolved at runtime by scanning label files. "
            "All detected IDs in {0,1,2} are mapped to 'vehicle'. "
            "Unexpected IDs cause the pipeline to fail loudly."
        ),
    }

    summary = {
        "final_classes": {str(i): name for i, name in enumerate(FINAL_CLASSES)},
        "visdrone_class_mapping": visdrone_mapping,
        "uavdt_class_mapping": uavdt_mapping,
    }

    path = report_dir / "class_mapping_summary.json"
    write_json_report(path, summary)
    print(f"  Class mapping summary written: {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the full preprocessing pipeline.

    Args:
        argv: Optional argument list (uses sys.argv if None).

    Returns:
        0 on success, 1 on error.
    """
    args = parse_args(argv)

    raw_root: Path = args.raw_root.resolve()
    out_root: Path = args.out_root.resolve()
    report_dir: Path = out_root / "reports"

    vd_raw = raw_root / "visdrone_raw"
    uavdt_raw = raw_root / "uavdt_raw"
    vd_out = out_root / "visdrone_yolo"
    uavdt_out = out_root / "uavdt_yolo"
    combined_out = out_root / "combined_yolo"

    print("=" * 60)
    print("  YOLO Preprocessing Pipeline")
    print("=" * 60)
    print(f"  Raw root     : {raw_root}")
    print(f"  Output root  : {out_root}")
    print(f"  Force rebuild: {args.force}")
    print()

    ensure_dir(out_root)
    ensure_dir(report_dir)

    # ------------------------------------------------------------------
    # Step 1: VisDrone
    # ------------------------------------------------------------------
    if not args.skip_visdrone:
        should_process = _prepare_output_dir(vd_out, args.force, "visdrone_yolo")
        if should_process:
            try:
                visdrone_processor.run(
                    raw_root=vd_raw,
                    out_root=out_root,
                    report_dir=report_dir,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(f"\nERROR processing VisDrone: {exc}", file=sys.stderr)
                return 1
        else:
            print("  Skipping VisDrone (output exists; use --force to rebuild).")
    else:
        print("  Skipping VisDrone (--skip-visdrone).")

    # ------------------------------------------------------------------
    # Step 2: UAVDT
    # ------------------------------------------------------------------
    if not args.skip_uavdt:
        should_process = _prepare_output_dir(uavdt_out, args.force, "uavdt_yolo")
        if should_process:
            try:
                uavdt_processor.run(
                    raw_root=uavdt_raw,
                    out_root=out_root,
                    report_dir=report_dir,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(f"\nERROR processing UAVDT: {exc}", file=sys.stderr)
                return 1
        else:
            print("  Skipping UAVDT (output exists; use --force to rebuild).")
    else:
        print("  Skipping UAVDT (--skip-uavdt).")

    # ------------------------------------------------------------------
    # Step 3: Combine
    # ------------------------------------------------------------------
    if not args.skip_combine:
        # The combine step depends on both source datasets being present
        if not vd_out.exists():
            print(
                "\nERROR: Cannot combine — visdrone_yolo/ not found at "
                f"{vd_out}.\n"
                "Run without --skip-visdrone first.",
                file=sys.stderr,
            )
            return 1
        if not uavdt_out.exists():
            print(
                "\nERROR: Cannot combine — uavdt_yolo/ not found at "
                f"{uavdt_out}.\n"
                "Run without --skip-uavdt first.",
                file=sys.stderr,
            )
            return 1

        should_combine = _prepare_output_dir(combined_out, args.force, "combined_yolo")
        if should_combine:
            try:
                combine_datasets.run(
                    vd_out_root=vd_out,
                    uavdt_out_root=uavdt_out,
                    combined_out_root=combined_out,
                    report_dir=report_dir,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(f"\nERROR combining datasets: {exc}", file=sys.stderr)
                return 1
        else:
            print("  Skipping combine (output exists; use --force to rebuild).")
    else:
        print("  Skipping combine (--skip-combine).")

    # ------------------------------------------------------------------
    # Step 4: Class mapping summary
    # ------------------------------------------------------------------
    _write_class_mapping_summary(report_dir)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Pipeline complete.")
    print(f"  Processed datasets : {out_root}")
    print(f"  Reports            : {report_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
