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

## 2. Installation

We recommend using **[pyenv](https://github.com/pyenv/pyenv)** to manage Python versions and keep the environment isolated.
We used Pyenv with: 
- Python version: 3.12.3
- Torch version: 2.6.0
- MONAI version: 1.5.0

### Step 1: Create a Python Environment
```
# Install Python (example: 3.12.3)
pyenv install 3.12.3

# Create a virtual environment with pyenv-virtualenv
pyenv virtualenv 3.12.3 mip-tumor-seg

# Activate the environment
pyenv activate mip-tumor-seg
```

### Step 2: Install Dependencies

Use the requirements.txt file in the project to install the dependencies. 
```
pip install -r requirements.txt
```

Note that if at any point in the installation, any of the packages is not being installed properly, it is recommended to manually delete the specific package version by deleting the number following the package name in the requirements.txt file. 
If you wish to install the latest versions of all dependencies, you can remove version pins from requirements.txt with the following command:
```
sed -i 's/[=<>!].*//g' requirements.txt
```

### Step 3: Install the Local pet_ct Package (written by ZROM) 
This package contains functions and classes that are mandatory for the scripts to run. 
Run the following command in the terminal: 
```
pip install -e Codes/pet_ct
```

## 2. Repository Structure

```
EMA4MICCAI-2025-MIP-Based-Tumor-Segmentation/  
│  
├── Checkpoints/ # Will hold all the checkpoints created when training the models  
├── Codes/ # Will hold all the scripts and codes  
├── Configs/ # Training and Testing configuration files  
├── Datasets/ # Dataset placeholders (not included)  
├── TestPredictions/ # Will contain the predicted segmentation masks (Test) for the different experiments (not included)  
├── TestResults/ # Will contain JSON files with metrics for Tests  
├── ValPredictions/  # Will contain the predicted segmentation masks (Validation during different epochs) for the different experiments (not included)  
├── requirements.txt # Python package dependencies  
└── README.md # This README file  
```

## 3. Dataset

We use the **autoPET 2022 Grand Challenge** dataset.  
The dataset is held on the TCIA website.

- Grand Challenge link and details: [https://autopet.grand-challenge.org/](https://autopet.grand-challenge.org/)
- Dataset link and pre-processing codes: [https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/](https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/)

---

## 4. Data Preparation and MIP Generation

### 4.1 Pre-processing

Use the official **autoPET pre-processing tools** (available on the dataset site) to prepare raw 3D PET/CT volumes in niftii format. This ensures standardized, resampled high-quality inputs.

### 4.2 Directories organization 

Under the Datasets folder, make sure to keep the newly created Niftis in the following directory tree:

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

### 4.3 MIP Generation

Our repository provides scripts to generate rotational Multi-Angle MIPs: OR-MIPs and OC-MIPs

- Located in `Codes/create_dataset`  
- Supports customizable rotation angles, start and end angles, binary and multi-label segmentations, and different occlusion correction configurations.

**Commands to create the dataset used in the paper:**  

OR-MIPs:  
```bash
python Codes/create_dataset/creating_all_mips_new.py --num_of_mips 48 --starting_angle 0 --ending_angle 180 --input_path Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis --output_path Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs
```
OC-MIPs: 
```
python Codes/create_dataset/create_screened_dataset.py --num_of_mips 48 --volume_threshold 25 --threshold 75 --split_tumors --start_angle 0 --end_angle 0 --input_path Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis --output_path Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new --filter_by_gradient
```

You should run these commands with num_of_mips = 16/32/48/64/80 for all datasets.  

## 5. Experiments

We conducted three main experiments to evaluate our MIP-based tumor segmentation approach.  
Each experiment can be **reproduced** using the provided configuration files in the `Configs/` directory, or replicated with **pre-trained weights** (see Section 6).

---

### **5.1 Segmentation: MIPs vs. 3D Volumes**

**Goal:**  
Compare MIP-based segmentation models to full 3D volumetric segmentation pipelines.

**Setup:**  
- **3D model:** Swin-UNETR trained on volumetric PET (SUV) data  
- **2D model:** Attention U-Net trained on Multi-Angle MIPs  
- **Loss:** Dice or Dice + Cross-Entropy  
- **Evaluation metrics:** Dice, IoU, Hausdorff Distance, Convergence Time, Energy per Epoch, TFLOPs  
- **Configs:**  
  - `Configs/config_segmentation_3D.yaml`  
  - `Configs/config_segmentation_MIPs_48.yaml`

**Training from scratch:**
```bash
# Train 3D (fold 1):
python Codes/main.py --config Configs/experiment_configurations/train/SwinUNETR_3D/PET_non_healthy_swin_pet_only_fold1.yaml

# Train OR-MIPs (fold 1): 
python Codes/main.py --config Configs/experiment_configurations/train/AttentionUnet_48mips_OR/AttUnet_48MIPs_fold1.yaml
```
To train the OC-MIPs, see experiment 5.3 (48 MIPs only). 

**Testing using pre-trained weights:**  

The Weights can be found [here](https://drive.google.com/drive/u/2/folders/1xmzh8d2Uxs-AYG2TkCYP3Cq8UHPBNO1W).
Copy the "AttentionUnet_48mips_original", "AttentionUnet_48mips_TrainAS_TestOriginal", and "" directories to the "Weights" directory in the project and run the following: 

```bash
# Test projected 3D predictions on OR-MIPs (fold 1):
python Codes/main_test.py --config Configs/experiment_configurations/test/SwinUNETR_3D/PET_non_healthy_swin_pet_only_naive_mips_test_fold1.yaml

# Test OC-MIPs on OR-MIPs (fold 1): 
python Codes/main_test.py --config Configs/experiment_configurations/test/AttentionUnet_48mips_original/AttUnet_48MIPs_TrainAS_TestOriginal_fold1.yaml

# Test OR-MIPs on OR-MIPs (fold 1):
python Codes/main_test.py --config Configs/experiment_configurations/test/AttentionUnet_48mips_original/AttUnet_48MIPs_fold1.yaml
```
The results will appear under 'TestResults'.

---

### **5.2 Classification of Healthy vs. Non-Healthy Cases**

**Goal:**  
Evaluate whether a 2D CNN trained on Multi-Angle MIPs can classify patients as *healthy* or *non-healthy* (pathological) using PET scans, compared to a 3D CNN trained on full volumetric data.

**Setup:**  
- **Input:** 16 MIPs per case (PET MIPs only) / Volumetric 3D PET 
- **Model:** CNN with attention pooling and fully connected classification head  
- **Loss:** Cross-Entropy  
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score  
- **Train Configs:** `Configs/experiment_configurations/train/classifier/mip_16_binary_classifier_fold{X}.yaml` (for 2D) and `Configs/experiment_configurations/train/classifier/volumetric_binary_classifier_fold{X}` (for 3D)

**Training from scratch:**
```bash
# Train 3D (fold 1):
python Codes/main_classifier.py --config Configs/experiment_configurations/train/classifier/volumetric_binary_classifier_fold1.yaml

# Train 16 MIPs (fold 1): 
python Codes/main_classifier.py --config Configs/experiment_configurations/train/classifier/mip_16_binary_classifier_fold1.yaml
```

**Testing using pre-trained weights:**  

The Weights can be found [here](https://drive.google.com/drive/u/2/folders/1xmzh8d2Uxs-AYG2TkCYP3Cq8UHPBNO1W).
Copy the "classifier" directory to the "Weights" directory in the project and run the following: 

```bash
# Test 3D:
python Codes/main_classifier.py --config Configs/experiment_configurations/test/classifier/volumetric_binary_classifier_fold1.yaml

# Test 16 MIPs: 
python Codes/main_classifier.py --config Configs/experiment_configurations/test/classifier/mip_16_binary_classifier_fold1.yaml
```
The results will appear under 'TestResults/classifier'.

---

### **5.3 Segmentation Performance vs. Number of MIPs**

**Goal:**  
Analyze how the number of projection angles (MIPs) impacts segmentation performance and efficiency.

**Setup:**  
- **Input:** Multi-Angle PET MIPs generated at different angular resolutions (16, 32, 48, 64, 80)  
- **Model:** Attention U-Net  
- **Loss:** Dice Loss
- **Evaluation metrics:** Dice, IoU, Hausdorff Distance, TFLOPs, Inference Time  
- **Configs:** `Configs/config_segmentation_MIPs_{16,32,48,64,80}.yaml`

**Train command example (for 48 MIPs):**
```bash
python Codes/train.py --config Configs/config_segmentation_MIPs_48.yaml
```

**Note:**  
48 MIPs were found to provide the best trade-off between accuracy and computational cost.

---

After training, validation and test predictions will be automatically stored in:  
```
ValPredictions/  
TestPredictions/  
TestResults/
```

Pre-trained model weights and configuration files are available for download (see below).

---

## 6. Results

### **6.1 Segmentation: MIPs vs. 3D**

| Metric | 3D (projected) | OR-MIPs | OC-MIPs |
|:--|:--:|:--:|:--:|
| Dice ↑ | 0.597 ± 0.05 | 0.578 ± 0.01 | **0.591 ± 0.01** |
| IoU ↑ | 0.471 ± 0.04 | 0.452 ± 0.01 | **0.466 ± 0.01** |
| HD ↓ | 139.61 ± 8.42 | 102.81 ± 9.61 | **102.26 ± 9.53** |
| CT (h) ↓ | 54.64 ± 19.22 | 24.14 ± 17.8 | **13.18 ± 4.1** |
| EPE (Wh/epoch) ↓ | 142.2 ± 79.1 | 40.22 ± 12.48 | **34.19 ± 4.7** |
| TFLOPs ↓ | 317.4 ± 144.0 | **0.97 ± 0.29** | **0.97 ± 0.29** |

---

### **6.2 Classification: Healthy vs. Non-Healthy**

| Metric | 3D | 16 MIPs |
|:--|:--:|:--:|
| Accuracy (%) ↑ | 72.8 ± 3.2 | **80.5 ± 1.7** |
| Precision (%) ↑ | 75.4 ± 6.0 | **83.6 ± 3.3** |
| Recall (%) ↑ | **91.9 ± 8.8** | 89.5 ± 2.9 |
| F1-score (%) ↑ | 82.3 ± 1.2 | **86.4 ± 0.8** |
| CT (h) ↓ | 44.7 ± 1.5 | **4.2 ± 0.2** |
| EPE (Wh/epoch) ↓ | 70.7 ± 3.23 | **4.7 ± 0.1** |

---

### **6.3 Effect of the Number of MIPs**

| MIPs | Dice (mean ± std) | TFLOPs (↓) | Inference Time (s) ↓ |
|:--|:--:|:--:|:--:|
| 16 | 0.562 ± 0.012 | 0.32 | 3.1 |
| 32 | 0.579 ± 0.010 | 0.64 | 5.2 |
| **48** | **0.591 ± 0.009** | 0.97 | 7.0 |
| 64 | 0.590 ± 0.011 | 1.31 | 9.6 |
| 80 | 0.589 ± 0.010 | 1.64 | 11.3 |

---

**Summary:**  
- **MIP-based segmentation** achieves similar Dice scores to 3D while being 4× faster and 300× more efficient.  
- **Classification on 16 MIPs** surpasses 3D with 10× faster training and 93% lower energy use.  
- **48 MIPs** provide the optimal trade-off between segmentation performance and efficiency.

---






