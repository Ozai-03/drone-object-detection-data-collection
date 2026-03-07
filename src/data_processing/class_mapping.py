"""
class_mapping.py

Defines the final class taxonomy for the unified YOLO dataset and provides
mapping helpers from raw dataset class IDs to final class IDs.

Final taxonomy (shared across all three output datasets):
    0 = person
    1 = vehicle
    2 = two_wheeler
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Final class taxonomy
# ---------------------------------------------------------------------------

PERSON: int = 0
VEHICLE: int = 1
TWO_WHEELER: int = 2

FINAL_CLASSES: list[str] = ["person", "vehicle", "two_wheeler"]

# ---------------------------------------------------------------------------
# VisDrone class mapping
# ---------------------------------------------------------------------------
# VisDrone2019-DET annotation class IDs (from official documentation):
#   0  = ignored regions  -> DROP (None)
#   1  = pedestrian       -> person
#   2  = people           -> person
#   3  = bicycle          -> two_wheeler
#   4  = car              -> vehicle
#   5  = van              -> vehicle
#   6  = truck            -> vehicle
#   7  = tricycle         -> two_wheeler
#   8  = awning-tricycle  -> two_wheeler
#   9  = bus              -> vehicle
#   10 = motor            -> two_wheeler
#   11 = others           -> DROP (None)

VISDRONE_MAP: dict[int, int | None] = {
    0: None,          # ignored regions — drop
    1: PERSON,        # pedestrian
    2: PERSON,        # people (crowd)
    3: TWO_WHEELER,   # bicycle
    4: VEHICLE,       # car
    5: VEHICLE,       # van
    6: VEHICLE,       # truck
    7: TWO_WHEELER,   # tricycle
    8: TWO_WHEELER,   # awning-tricycle
    9: VEHICLE,       # bus
    10: TWO_WHEELER,  # motor / motorcycle
    11: None,         # others — drop
}

# Human-readable names for drop-reason reporting
VISDRONE_CLASS_NAMES: dict[int, str] = {
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

# ---------------------------------------------------------------------------
# UAVDT class mapping
# ---------------------------------------------------------------------------
# The Zenodo UAVDT release (record 14575517) ships labels already in YOLO
# normalized format.  The dataset is vehicle-focused; native class IDs
# observed are 0, 1, 2 representing vehicle sub-types (car, bus/truck, van).
# All map to `vehicle` in the final taxonomy.
#
# This mapping is built at runtime via `build_uavdt_map()` after scanning the
# raw label files.  If unexpected class IDs are found the pipeline fails
# loudly so the mapping can be extended explicitly.
#
# TODO: If the Zenodo release ships a `classes.txt` or `data.yaml`, the
#       processor should read it to confirm class names instead of relying
#       on this hardcoded assumption.

# Known vehicle-only class IDs (all map to VEHICLE).
_UAVDT_KNOWN_VEHICLE_IDS: frozenset[int] = frozenset({0, 1, 2})


def build_uavdt_map(detected_class_ids: set[int]) -> dict[int, int | None]:
    """Build a class-ID mapping for the UAVDT dataset based on observed IDs.

    The UAVDT Zenodo release is vehicle-focused; IDs {0, 1, 2} are treated as
    vehicle sub-types and mapped to VEHICLE.  If any class ID outside this
    known set is detected, the function raises a ValueError with instructions
    so that the developer can extend the mapping explicitly.

    Args:
        detected_class_ids: Set of unique integer class IDs found across all
                            UAVDT label files.

    Returns:
        A dict mapping each detected class ID to a final class ID (or None to
        drop).

    Raises:
        ValueError: If any detected class ID is not in the known set and
                    therefore cannot be mapped without human review.
    """
    unknown = detected_class_ids - _UAVDT_KNOWN_VEHICLE_IDS
    if unknown:
        raise ValueError(
            f"UAVDT label files contain unexpected class IDs: {sorted(unknown)}.\n"
            "The pipeline cannot silently guess their meaning.\n"
            "Please inspect the raw label files and extend `build_uavdt_map()` "
            "in src/data_processing/class_mapping.py with the correct mapping."
        )

    mapping: dict[int, int | None] = {}
    for cid in detected_class_ids:
        # All known UAVDT IDs are vehicle sub-types
        mapping[cid] = VEHICLE

    return mapping


# ---------------------------------------------------------------------------
# Generic helper
# ---------------------------------------------------------------------------

def apply_mapping(class_id: int, mapping: dict[int, int | None]) -> int | None:
    """Map a raw dataset class ID to a final class ID.

    Args:
        class_id: The raw class integer from the annotation file.
        mapping:  A dict mapping raw IDs to final IDs; None value means drop.

    Returns:
        The mapped final class ID, or None if the annotation should be dropped.
        Also returns None for any class_id not present in the mapping dict.
    """
    return mapping.get(class_id, None)
