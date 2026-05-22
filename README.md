# Drone Object Detection

**Agentic Object Detection and Natural Language Querying for Drone Footage**

Mathew Peguero — Machine Learning Engineering Capstone

---

## Overview

This project trains a high-performance object detection model on UAV (drone) imagery with the
goal of turning raw aerial footage into structured, queryable data. The system detects three
object classes — persons, vehicles, and two-wheelers — from drone viewpoints and converts
frame-level detections into structured event records that support downstream analytics and
natural language querying.

Two large-scale public UAV datasets (VisDrone and UAVDT) are unified into a single
production-ready corpus via a rigorous 5-step data cleaning pipeline. Four YOLOv8 variants
are benchmarked on a held-out subset, and the best model (YOLOv8m) is trained to completion
on the full dataset, achieving **mAP50 = 0.515** on the held-out test set with particularly
strong vehicle detection at **mAP50 = 0.830**.

---

## Background

Standard object detection benchmarks (COCO, ImageNet) are dominated by ground-level
photography: objects are large relative to frame size, viewed from eye level, and rarely
occluded by overlapping neighbors. Drone imagery breaks all three assumptions.

From a UAV at operational altitude, pedestrians occupy as few as 10–20 pixels in a 640×640
frame, vehicles appear as uniform rectangles without distinguishing frontal or side-profile
features, and crowd scenes produce hundreds of densely overlapping bounding boxes per image.
Models trained exclusively on ground-level data transfer poorly to these conditions.

VisDrone and UAVDT are the two most widely used benchmarks for this domain. Each captures a
different slice of the problem: VisDrone emphasizes pedestrian-heavy urban scenes with mixed
object classes, while UAVDT focuses on traffic monitoring with near-exclusive vehicle coverage.
Training on both, after carefully unifying their annotation schemas, yields a more
generalizable detector than either dataset alone.

---

## Datasets

### VisDrone2019-DET

| Property | Value |
|---|---|
| Source | https://github.com/VisDrone/VisDrone-Dataset |
| Images | 10,000+ |
| Raw classes | 12 (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor, ignored_region, others) |
| Annotation format | 8-field CSV: x, y, w, h, score, class_id, truncation, occlusion |
| Splits | train / val / test-dev |
| Raw bounding boxes | ~540,000 |

VisDrone is a widely used benchmark for UAV-based detection. Scenes span urban and suburban
environments in China, captured at varying altitudes and viewpoints with dense annotation of
all visible objects.

### UAVDT

| Property | Value |
|---|---|
| Source | https://zenodo.org/records/14575517 |
| Frames | 77,000+ |
| Raw classes | 3 (car, truck/bus, van — all vehicle sub-types) |
| Annotation format | YOLO normalized: class_id cx cy w h |
| Splits | train / val / test |
| Raw bounding boxes | ~840,000 |

UAVDT is a vehicle-focused traffic surveillance dataset recorded from drone platforms under
diverse conditions: varying altitude, weather, illumination, and camera motion. It contributes
density and scale variation to the combined corpus.

### Unified Class Taxonomy

Both datasets are remapped to a shared 3-class taxonomy. Classes outside this taxonomy are
dropped during processing.

| Final Class | ID | VisDrone source classes | UAVDT source classes |
|---|---|---|---|
| person | 0 | pedestrian (1), people (2) | — |
| vehicle | 1 | car (4), van (5), truck (6), bus (9) | car (0), truck/bus (1), van (2) |
| two_wheeler | 2 | bicycle (3), tricycle (7), awning-tricycle (8), motor (10) | — |
| dropped | — | ignored_region (0), others (11) | — |

### Combined Dataset (Final)

| Split | Images | Source breakdown |
|---|---|---|
| Train | 7,737 | VisDrone + UAVDT |
| Val | 819 | VisDrone + UAVDT |
| Test | 1,882 | VisDrone + UAVDT |
| **Total** | **10,438** | VisDrone: 8,629 (82.7%) / UAVDT: 1,809 (17.3%) |

**Total annotations (post-cleaning): 21,968**

| Class | Count | Share |
|---|---|---|
| vehicle | 11,942 | 54.4% |
| person | 7,724 | 35.2% |
| two_wheeler | 2,302 | 10.5% |

---

## Data Processing Pipeline

The pipeline is implemented in [src/data_processing/](src/data_processing/) and orchestrated
by [src/data_processing/build_all.py](src/data_processing/build_all.py). Processing runs in
three stages: per-dataset conversion, per-dataset cleaning, and final merge.

### Cleaning Steps

A 5-step cleaning pipeline is applied independently to each dataset before merging:

| Step | Filter | VisDrone dropped | UAVDT dropped |
|---|---|---|---|
| 1 | Invalid boxes (negative coordinates, outside image bounds) | 0 | 12 |
| 2 | Degenerate boxes (width < 5 px or height < 5 px) | 445 | 6 |
| 3 | Outlier area (IQR upper fence: Q3 + 1.5 × IQR) | 2,112 | 1,189 |
| 4 | Non-target classes (ignored_region, others) | 377 | — |
| 5 | Schema unification: remap raw class IDs to final taxonomy | — | — |

Steps 1–3 remove annotation noise before class mapping. Step 4 is VisDrone-specific. Step 5
produces the unified 3-class labels written to YOLO format.

### Pipeline Outputs

```
data/processed/
├── visdrone_yolo/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── visdrone.yaml
├── uavdt_yolo/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── uavdt.yaml
├── combined_yolo/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── combined.yaml
└── reports/
    ├── visdrone_report.json
    ├── uavdt_report.json
    ├── combined_report.json
    └── class_mapping_summary.json
```

---

## Model Architecture and Training

**Framework:** YOLOv8 (Ultralytics)  
**Input resolution:** 640 × 640  
**GPU:** NVIDIA A100-SXM4-40GB (Google Colab)

### Phase 1 — Subset Model Comparison

Four YOLOv8 variants are trained on 35% of the training data for 20 epochs to identify the
best architecture before committing to a full run.

| Model | mAP50 | mAP50-95 | Precision | Recall | Weights | Train time |
|---|---|---|---|---|---|---|
| yolov8n | 0.4365 | 0.2068 | 0.5498 | 0.4389 | 6.25 MB | 467 s |
| yolov8s | 0.5214 | 0.2594 | 0.6411 | 0.5002 | 22.52 MB | 560 s |
| **yolov8m** | **0.5663** | **0.2899** | **0.6770** | **0.5404** | **52.03 MB** | **833 s** |
| yolov8n (tuned) | 0.4311 | 0.2051 | 0.5359 | 0.4363 | 6.25 MB | 463 s |

The tuned nano variant uses lr0 = 0.001, mosaic = 1.0, HSV augmentation (h: 0.015, s: 0.7,
v: 0.4), and vertical flip probability 0.1. Despite tuning, it does not outperform the
baseline nano — the capacity gap between nano and medium is the dominant factor.

**Selected model: YOLOv8m**

### Phase 2 — Full Training

YOLOv8m is trained on 100% of the training data for 100 epochs.

| Setting | Value |
|---|---|
| Epochs | 100 |
| Batch size | 8 |
| Optimizer | Auto (AdamW) |
| Learning rate schedule | Cosine annealing (lr0 = 0.01, lrf = 0.01) |
| Augmentation | mosaic = 1.0, fliplr = 0.5, erasing = 0.4, RandAugment |
| Early stopping patience | 100 |
| Best epoch | 77 |
| Total training time | 11,668 s (~3.24 hours) |

---

## Results

### Validation Performance (Best Epoch: 77 / 100)

| Metric | Value |
|---|---|
| mAP50 | 0.6438 |
| mAP50-95 | 0.3467 |
| Precision | 0.7428 |
| Recall | 0.6094 |

### Test Set Performance (Held-Out)

| Metric | Value |
|---|---|
| mAP50 | 0.5154 |
| mAP50-95 | 0.2704 |
| Precision | 0.6783 |
| Recall | 0.5041 |

### Per-Class Test Performance

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| person | 0.611 | 0.302 | 0.318 | 0.123 |
| vehicle | 0.859 | 0.797 | 0.830 | 0.522 |
| two_wheeler | 0.566 | 0.413 | 0.399 | 0.166 |

---

## Key Findings

**Vehicle detection is reliable.** With a mAP50 of 0.830 on the test set, vehicle detection
is strong enough for production use cases. Vehicles are the largest objects in the frame,
the most consistently annotated across both datasets, and represent the majority class
(54.4% of annotations).

**Person detection is the hardest problem.** Pedestrians often span fewer than 20 pixels
in a 640 × 640 frame at typical UAV altitude. The mAP50 of 0.318 reflects this fundamental
challenge — recall drops to 0.302, meaning most persons are missed rather than misclassified.
More data, higher input resolution (e.g., 1280 × 1280), and tiled inference are the most
promising paths forward.

**Two-wheelers are underrepresented.** At 10.5% of the dataset, two-wheelers are the
minority class. The mAP50 of 0.399 is consistent with class imbalance: the model sees
insufficient examples to generalize reliably, particularly for tricycles and
awning-tricycles that appear exclusively in VisDrone.

**Model scaling matters more than hyperparameter tuning at this data scale.** The tuned
nano model failed to close the gap to yolov8s, let alone yolov8m. The ~17% mAP50
improvement from nano to medium (0.4365 → 0.5663) at roughly 8× the parameter count
indicates the task is capacity-bound at this dataset size.

**Validation-to-test gap (~12 mAP50 points).** The gap between best validation performance
(0.6438) and held-out test performance (0.5154) reflects mild distribution shift between
the VisDrone splits, which were originally composed for different benchmark tracks. This
is expected and not a sign of severe overfitting — training and validation curves converge
cleanly by epoch 77.

---

## Project Structure

```
drone-object-detection-capstone/
├── data/
│   ├── raw/                          # Raw dataset downloads (not committed)
│   │   ├── visdrone_raw/
│   │   └── uavdt_raw/
│   ├── samples/                      # Sample images for quick testing
│   └── processed/                    # YOLO-format datasets (generated)
│       ├── visdrone_yolo/
│       ├── uavdt_yolo/
│       ├── combined_yolo/
│       └── reports/
├── scripts/
│   ├── download_visdrone.py          # Download VisDrone from Google Drive
│   ├── download_uavdt.py             # Download UAVDT from Zenodo
│   ├── verify_datasets.py            # Validate dataset integrity
│   └── zip_for_colab.py             # Bundle dataset for Colab upload
├── src/
│   └── data_processing/
│       ├── build_all.py              # CLI: runs full ETL pipeline
│       ├── visdrone_processor.py     # VisDrone CSV -> YOLO converter
│       ├── uavdt_processor.py        # UAVDT -> unified YOLO converter
│       ├── combine_datasets.py       # Merge both processed datasets
│       ├── class_mapping.py          # Class taxonomy and mapping dicts
│       └── utils.py                  # Shared I/O utilities
├── notebooks/
│   └── step5_data_wrangling.ipynb    # EDA and data quality analysis
├── colab/
│   ├── train.ipynb                   # Subset model comparison (template)
│   ├── train_full.ipynb              # Full training (template)
│   ├── train_ran.ipynb               # Subset comparison (executed)
│   └── train_full_ran.ipynb          # Full training (executed)
├── Capstone Research/
│   ├── README.md                     # Research survey documentation
│   ├── research/                     # Literature survey PDF
│   └── notebooks/                    # Reproduced baseline experiments
├── requirements.txt
└── DATA_SOURCES.md
```

---

## Setup

**Requirements:** Python 3.12, pip, Google account (for Colab training)

```bash
pip install -r requirements.txt
```

Key dependencies: `ultralytics >= 8.0.0`, `torch >= 2.0.0`, `torchvision >= 0.15.0`,
`numpy`, `pandas`, `Pillow`, `PyYAML`, `gdown`, `requests`, `tqdm`, `matplotlib`, `seaborn`

---

## Reproducing the Full Pipeline

### 1. Download raw datasets

```bash
python scripts/download_visdrone.py    # ~4 GB -> data/raw/visdrone_raw/
python scripts/download_uavdt.py       # ~4 GB -> data/raw/uavdt_raw/
```

### 2. Verify integrity

```bash
python scripts/verify_datasets.py
```

### 3. Build the combined YOLO dataset

```bash
python -m src.data_processing.build_all --force
```

This runs all three processing stages (VisDrone conversion, UAVDT conversion, merge) and
writes the final dataset to `data/processed/combined_yolo/` along with JSON reports in
`data/processed/reports/`.

### 4. Package for Google Colab

```bash
python scripts/zip_for_colab.py --dataset combined
```

Produces `colab_uploads/combined_yolo.zip` (~4.8 GB).

### 5. Subset model comparison (Colab)

Upload `combined_yolo.zip` to Google Drive, then open
[colab/train.ipynb](colab/train.ipynb) in Google Colab. The notebook trains all four
YOLOv8 variants on 35% of the training data and saves results to Drive.

### 6. Full training (Colab)

Open [colab/train_full.ipynb](colab/train_full.ipynb), set `BEST_MODEL = "yolov8m"`,
and run. The notebook trains for 100 epochs and automatically evaluates on the held-out
test set. Weights and metrics are saved to Drive under `drone_detection_training/final_model/`.

### Viewing executed results

Pre-executed versions of both notebooks are included:
- [colab/train_ran.ipynb](colab/train_ran.ipynb) — subset comparison with full output
- [colab/train_full_ran.ipynb](colab/train_full_ran.ipynb) — final training run with all metrics

---

## Research Foundation

The [Capstone Research](Capstone%20Research/) folder contains the academic foundation for
this project:

- **Literature survey** (`research/capstone_research_paper.pdf`) — structured review of
  drone-based object detection research, model-level optimizations for UAV imagery, and
  existing detection-to-event and agentic video analytics systems. The survey identifies the
  key gap motivating this project: the lack of semantic querying over detection outputs.

- **YOLOv8 baseline notebook** — reproduces a standard YOLOv8 detection workflow on drone
  imagery and establishes quantitative baselines (precision, recall, mAP@50, mAP@50-95,
  inference speed).

- **Detection-to-event pipeline notebook** — converts frame-level bounding box outputs into
  structured JSONL and CSV event records, then demonstrates query-style analytics (object
  counts by class, confidence-based filtering, peak activity windows). This is the foundation
  for the capstone's structured data layer and the natural language querying system built
  on top of it.

---

## Data Sources and Licensing

Raw datasets are not stored in this repository. They must be downloaded from their official
sources. See [DATA_SOURCES.md](DATA_SOURCES.md) for full attribution and licensing details.

| Dataset | License | Source |
|---|---|---|
| VisDrone2019-DET | Academic / research use | https://github.com/VisDrone/VisDrone-Dataset |
| UAVDT | Academic / research use | https://zenodo.org/records/14575517 |

Both datasets are used strictly for non-commercial educational purposes in accordance with
their respective license terms.
