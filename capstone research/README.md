## Research Survey

**File:** `research/capstone_research_paper.pdf`

This document presents a structured survey of:
- Academic research on drone-based object detection using YOLOv8
- Model-level optimization approaches for UAV object detection
- Open-source and industry systems that convert detections into structured metadata
- Emerging agentic and natural language querying systems for video analytics

The survey identifies key gaps in existing work—particularly the lack of semantic querying and reasoning over detection outputs—and motivates the design choices made in the proposed capstone project.

---

## Reproduced Experiments

### Notebook 01 — YOLOv8 Drone Detection Baseline

**File:** `notebooks/yolov8_baseline.ipynb`

This notebook reproduces a **baseline YOLOv8 object detection workflow** on drone imagery.

Key elements:
- Uses the official Ultralytics YOLOv8 implementation
- Evaluates detection performance on a drone dataset split (train/val/test)
- Reports standard metrics:
  - Precision
  - Recall
  - mAP@50
  - mAP@50–95
  - Inference speed
- Saves results to CSV/JSON for reproducibility

This notebook establishes a **quantitative performance baseline** against which future model and system improvements can be compared.

---

### Notebook 02 — Detection to Structured Event Pipeline

**File:** `notebooks/Detection_to_event_pipeline.ipynb`

This notebook reproduces the **system-level pattern** used by modern video analytics platforms.

Key elements:
- Runs YOLOv8 inference on drone images or video
- Converts frame-level detections into structured event records
- Persists events in machine-readable formats (JSONL and CSV)
- Demonstrates query-style analytics:
  - Object counts by class
  - Confidence-based filtering
  - Peak activity windows

This notebook directly supports the capstone’s core idea:  
**object detection outputs become structured, queryable data that enables higher-level reasoning and natural language interaction.**

---

## How This Relates to the Capstone

Together, the research survey and reproduced notebooks demonstrate:
- Understanding of prior work and existing solutions
- Hands-on reproduction of public implementations
- Clear identification of limitions in existing approaches
- Justification for the proposed capstone system, which integrates:
  - YOLOv8-based drone detection
  - Structured event storage
  - An agentic natural language querying layer

These artifacts form the foundation for the final capstone implementation.

---

## Environment & Reproducibility

- Notebooks are designed for **Google Colab**
- Minimal dependencies are required (`ultralytics`)
- Outputs are saved to Google Drive for easy export and version control

Each notebook includes inline documentation describing:
- What was reproduced
- What was learned
- How results compare to surveyed research

---

## Notes

- Large datasets are not included in this repository.  
  Dataset sources and structures are documented within the notebooks.
- Notebooks are intended for google colab

---

## Capstone Checklist

- [x] Research survey of related work  
- [x] Identification of public implementations  
- [x] Reproduced model-level baseline  
- [x] Reproduced system-level event pipeline  
- [x] Analysis of strengths, weaknesses, and improvements  

---

**Author:** Mathew Peguero  
**Project:** Agentic Object Detection and Natural Language Querying for Drone Footage
