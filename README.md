# MIP-Based Tumor Segmentation: A Radiologist-Inspirde Approach

Official implementation and dataset of the paper:
**MIP-Based Tumor Segmentation: A Radiologist-Inspired Approach**
accepted to the 1st workshop on Efficient Medical AI in MICCAI 2025 (Daejeon, South Korea).

[![EMA4MICCAI 2025](https://img.shields.io/badge/EMA4MICCAI-2025-blue.svg)](https://miccai.org)

## Table of Contents
1. [Introduction](#1-introduction)  
2. [Dataset](#2-dataset)  
3. [Data Preparation and MIP Generation](#3-data-preparation-and-mip-generation)  
4. [Environment Setup](#4-environment-setup)  
5. [Training](#5-training)  
6. [Evaluation](#6-evaluation)  
7. [Results Summary](#7-results-summary)  
8. [Repository Structure](#8-repository-structure)  
9. [Citation](#9-citation)  
10. [Acknowledgments](#10-acknowledgments)  
11. [License](#11-license)  

---

## 1. Introduction

This repository accompanies our paper accepted at **EMA4MICCAI 2025**, proposing a method for **tumor segmentation using Multi-Angle Maximum Intensity Projections (MIPs)** derived from PET/CT scans.

Traditional clinical workflows often start with rotational MIPs for tumor detection before volumetric slice analysis, but automated segmentation models typically operate on 3D volumes, making them computationally intensive.

Our approach:

- Trains segmentation models **directly on MIPs**, matching clinical practice.  
- Introduces a **novel occlusion correction** method for improved MIP annotation quality.  
- Achieves **comparable segmentation accuracy** to 3D models, with significant reductions in training time and computational resources.

---

## 2. Dataset

We use the **autoPET 2022 Grand Challenge** dataset.

- Dataset link and details: [https://autopet.grand-challenge.org/](https://autopet.grand-challenge.org/)  
- Please follow their instructions to download and pre-process the data.

---

## 3. Data Preparation and MIP Generation

### 3.1 Pre-processing

Use the official **autoPET pre-processing tools** (available on the dataset site) to prepare raw 3D PET/CT volumes. This ensures standardized, high-quality inputs.

### 3.2 MIP Generation

Our repository provides scripts to generate rotational Multi-Angle MIPs:

- Located in `scripts/mip_generation/`  
- Supports customizable rotation angles and occlusion correction.

**Example command:**

```bash
python scripts/mip_generation/create_mips.py --input_dir path/to/preprocessed_data --output_dir path/to/mip_output --angles 48



Soon to be uploaded...
