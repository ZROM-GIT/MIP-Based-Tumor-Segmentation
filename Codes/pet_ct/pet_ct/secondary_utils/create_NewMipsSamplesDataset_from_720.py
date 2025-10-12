"""
This file is meant to create a new dataset by sampling from the 720 MIPS
"""

import monai
import nibabel as nib
import numpy as np
import os
from pathlib2 import Path
import torch
from tqdm import tqdm

from pet_ct.secondary_utils.create_screened_mips import create_screened_mips
from pet_ct.secondary_utils.AffineMatrices import affine_matrices

num_of_mips = 16

# Input and Output paths:
Input_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs720')  # Path to all subjects with 720 MIPs
Output_path = Path(f'/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs{num_of_mips}')

trials_paths = sorted(list(Input_path.glob('*/*')))
print('Patients paths loaded...')

# Load transform

# Indices to take
all_indices = np.linspace(start=0, stop=(180 - 180 / num_of_mips), num=num_of_mips) * 2

for j, trial_path in enumerate(tqdm(trials_paths, desc='Case Studies')):
    # All files paths:
    PET_path = trial_path / 'PET.nii.gz'
    CT_path = trial_path / 'CT.nii.gz'
    SEG_path = trial_path / 'SEG.nii.gz'
    SUV_path = trial_path / 'SUV.nii.gz'
    CTres_path = trial_path / 'CTres.nii.gz'
    HGUO_path = trial_path / 'HGUO.nii.gz'




