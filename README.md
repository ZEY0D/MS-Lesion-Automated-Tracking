# Automated Longitudinal Tracking of Multiple Sclerosis Lesions 🧠

> **Research Assistant Project** | Neuroimaging & AI Group
> **Role:** Technical Lead / Research Assistant

## 📌 Project Overview
This repository houses the computational pipeline developed during my tenure as a Research Assistant to automate the quantification of Multiple Sclerosis (MS) disease progression.

In clinical settings, manually tracking lesion changes across timepoints is prone to human error and inter-observer variability. This project solves that challenge by engineering a **fully automated, end-to-end system** that ingests raw longitudinal MRI scans and outputs a precise volumetric report of disease activity.

## 🎯 Key Objectives achieved
1.  **Automated Segmentation:** Deployed **nnU-Net** to extract White Matter Lesions from FLAIR MRI sequences with high sensitivity.
2.  **Robust Motion Correction:** Engineered a **Multi-Stage Rigid Registration** algorithm (using SimpleITK) capable of autonomously correcting patient head motion with sub-millimeter precision.
3.  **Disease Quantification:** Developed logic to classify lesion evolution into clinical categories: **New** (Active), **Disappeared** (Healing), **Enlarged**, and **Shrunk**.
4.  **Pipeline Validation:** Implemented rigorous "Stress Tests" by artificially inducing geometric distortions (sabotaging data) to verify the algorithm's self-correcting capabilities.

## 🔬 Clinical Impact
The pipeline was validated on real-world patient data (Patient 11 Case Study), successfully identifying complex progression patterns:
* **Active Disease:** Detected **7 new lesions** and 4 enlarged lesions, flagging a breakthrough in disease activity.
* **Treatment Response:** Simultaneously tracked **12 disappeared lesions**, quantifying the patient's partial response to therapy.
* **Precision:** Achieved a registration correlation score of **0.91** (Clinical Grade), validated via Fusion Overlay techniques.

## 🛠️ Tech Stack
* **Core:** Python, SimpleITK, Nibabel
* **AI/ML:** PyTorch, nnU-Net
* **Data Analysis:** NumPy, Pandas, Matplotlib
