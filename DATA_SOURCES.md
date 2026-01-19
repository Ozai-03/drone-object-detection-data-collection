# Data Sources

This project uses large-scale, publicly available UAV (drone) datasets for
non-commercial, educational research purposes as part of a Machine Learning
Engineering bootcamp capstone project.

Due to dataset size and licensing restrictions, raw data is **not stored**
in this GitHub repository. Instead, scripts and documentation are provided
to reproduce the data setup locally.

---

## VisDrone Detection Dataset

**Official Source:**  
https://github.com/VisDrone/VisDrone-Dataset

**Description:**  
The VisDrone Detection Dataset consists of images and video frames captured
from UAV-mounted cameras across various urban and suburban environments.
The dataset includes bounding box annotations for multiple object classes
commonly observed from aerial viewpoints.

**Common Object Classes:**
- Pedestrian
- Car
- Van
- Bus
- Truck
- Bicycle
- Motorcycle

**Why This Dataset Was Chosen:**  
VisDrone is a widely used benchmark for drone-based object detection and
closely aligns with the project objective of detecting and counting objects
from aerial imagery. The dataset provides diverse scenes, camera altitudes,
and object densities suitable for training deep learning models.

---

## UAVDT (UAV Detection and Tracking) Dataset

**Official Source:**  
https://zenodo.org/records/14575517

**Description:**  
The UAVDT dataset contains UAV-captured video sequences focused primarily
on vehicle detection and tracking. Frames are annotated with bounding boxes
and collected under varying conditions, including changes in altitude,
weather, illumination, and camera motion.

**Why This Dataset Was Chosen:**  
UAVDT complements VisDrone by providing a large number of vehicle-focused
samples from drone footage. Using both datasets increases data diversity
and improves model robustness and generalization for real-world UAV
object detection tasks.

---

## Dataset Size Summary

- **VisDrone:** 10,000+ labeled images / frames  
- **UAVDT:** 77,000+ annotated frames  
- **Combined:** 85,000+ labeled samples  

The combined dataset significantly exceeds the minimum dataset size
requirement of 15,000 samples.

---

## Licensing and Usage Notes

- The **VisDrone** dataset is provided for academic and research use.
- The **UAVDT** dataset is licensed for **research and academic purposes only**.

This project uses both datasets strictly for **non-commercial educational
purposes**. Raw datasets are **not redistributed** in this repository.
Users must obtain the data directly from the official sources and comply
with all associated license terms.

---

## Data Access and Reproducibility

To reproduce the dataset locally, refer to the scripts provided in the
`scripts/` directory:

- `download_visdrone.py`
- `download_uavdt.py`
- `verify_datasets.py`

These scripts document expected directory structures and verify that the
datasets are correctly installed before further processing.