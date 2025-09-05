# MIP-Based Tumor Segmentation: A Radiologist-Inspired Approach

Official implementation and dataset of the paper:  
**MIP-Based Tumor Segmentation: A Radiologist-Inspired Approach**  
Accepted to the 1st Workshop on Efficient Medical AI (EMA4MICCAI) at MICCAI 2025, Daejeon, South Korea.

[![EMA4MICCAI 2025](https://img.shields.io/badge/EMA4MICCAI-2025-blue.svg)](https://miccai.org)

**Authors:**  
Romario Zarik, Nahum Kiryati, Michael Green, Liran Domachevsky, Arnaldo Mayer  
Contact: Romario Zarik – romariozarik@mail.tau.ac.il  
Computational Imaging Lab (CILAB) – [https://www.cilab.org.il/](https://www.cilab.org.il/)

---

## 1. Introduction

This repository accompanies our paper accepted at **EMA4MICCAI 2025**, proposing a method for **tumor segmentation using Multi-Angle Maximum Intensity Projections (MIPs)** derived from PET/CT scans.

Traditional clinical workflows often start with rotational MIPs for tumor detection before volumetric slice analysis, but automated segmentation models typically operate on 3D volumes, making them computationally intensive.

Our approach:

- Trains segmentation models **directly on MIPs**, matching clinical practice.  
- Introduces a **novel occlusion correction** method for improved MIP annotation quality.  
- Achieves **comparable segmentation accuracy** to 3D models, with significant reductions in training time and computational resources.

---

## 2. Repository Structure

```
EMA4MICCAI-2025-MIP-Based-Tumor-Segmentation/  
│  
├── Checkpoints/ # Will hold all the checkpoints created when training the models  
├── Codes/ # Will hold all the scripts and codes  
├── Configs/ # Training and Testing configuration files  
├── Datasets/ # Dataset placeholders (not included)  
├── TestPredictions/ # Will contain the predicted segmentation masks (Test) for the different experiments (notincluded)  
├── TestResults/ # Will contain json files with metrics for Tests  
├── ValPredictions/  # Will contain the predicted segmentation masks (Validation during different epochs) for the different experiments (not included)  
├── requirements.txt # Python package dependencies  
└── README.md # This README file  
```

## 2. Dataset

We use the **autoPET 2022 Grand Challenge** dataset.  
The dataset is held in the TCIA website.

- Grand Challenge link and details: [https://autopet.grand-challenge.org/](https://autopet.grand-challenge.org/)
- Dataset link and pre-processing codes: [https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/](https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/)

---

## 3. Data Preparation and MIP Generation

### 3.1 Pre-processing

Use the official **autoPET pre-processing tools** (available on the dataset site) to prepare raw 3D PET/CT volumes in niftii format. This ensures standardized, resampled high-quality inputs.

### 3.2 Directories organization 

Under the Datasets folder, make sure to keep the newly created niftis in the following directory tree:

```
EMA4MICCAI-2025-MIP-Based-Tumor-Segmentation/
│
├── Datasets/                               # Dataset placeholders (not included)
│   └── FDG-PET-CT-Lesions/
│       └── manifest-1654187277763/
│           └── niftis/                  # This folder contains all our raw 3D data, contains one folder per patient
│               ├── PETCT_0af7ffe12a/        # Example patient
│               │   └── 08-12-2005-NA-PET-CT Ganzkoerper  primaer mit KM-96698/  # Example case for this patient
│               │       ├── CT.nii.gz        # High-resolution CT image
│               │       ├── CTres.nii.gz     # CT image resampled to PET resolution
│               │       ├── PET.nii.gz       # PET image
│               │       ├── SUV.nii.gz       # Standardized Uptake Value (SUV) image
│               │       └── SEG.nii.gz       # Ground-truth segmentation mask
│               ├── PETCT_0b57b247b6/        # Another patient (may have multiple cases)
│               └── PETCT_0b98dbe00d/        # Another patient
│
└── ...                                      # Additional patients/cases follow this structure

```

### 3.3 MIP Generation

Our repository provides scripts to generate rotational Multi-Angle MIPs: OR-MIPs and OC-MIPs

- Located in `Codes/create_dataset`  
- Supports customizable rotation angles, start and end angles, binary and multi-label segmentations, and different occlusion correction configurations.

**Commands to create the dataset used in the paper:**  

OR-MIPs:  
```bash
python codes/create_dataset/creating_all_mips_new.py --num_of_mips 48 --starting_angle 0 --ending_angle 180 --input_path Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis --output_path Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs
```


Soon to be uploaded...
