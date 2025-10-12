"""
This file is meant for changing the resolution of a dataset.
"""

import monai
import torch
import numpy as np
from pathlib2 import Path
from tqdm import tqdm
import os

input_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis')
output_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis_forPACS')

current_spacing = np.array([2.03642, 2.03642, 3])
target_spacing = np.array([4, 4, 4])

load_bilinear = monai.transforms.Compose([monai.transforms.LoadImage(image_only=True, ensure_channel_first=True),
                                             monai.transforms.Spacing(pixdim=(target_spacing), mode="bilinear")])
load_nearest = monai.transforms.Compose([monai.transforms.LoadImage(image_only=True, ensure_channel_first=True),
                                             monai.transforms.Spacing(pixdim=(target_spacing), mode="nearest")])

trials_paths = sorted(list(input_path.glob('*/*')))
print('Patients paths loaded...')

for j, trial_path in enumerate(tqdm(trials_paths, desc='Case Studies')):

    # All files paths:
    PET_path = trial_path / 'PET.nii.gz'
    SEG_path = trial_path / 'SEG.nii.gz'
    SUV_path = trial_path / 'SUV.nii.gz'
    CTres_path = trial_path / 'CTres.nii.gz'
    HGUO_path = trial_path / 'HGUO.nii.gz'

    PET = load_bilinear(PET_path)[0]
    SUV = load_bilinear(SUV_path)[0]
    CTres = load_bilinear(CTres_path)[0]
    SEG = load_nearest(SEG_path)[0]
    if os.path.exists(HGUO_path):
        HGUO = load_nearest(HGUO_path)[0]

    output_trial_path = output_path / trial_path.parent.name / trial_path.name
    output_trial_path.mkdir(parents=True, exist_ok=True)

    save = monai.transforms.SaveImage(output_dir=str(output_trial_path), output_postfix='', output_ext='.nii.gz',
                                          separate_folder=False, channel_dim=None)

    save(PET)
    save(SUV)
    save(SEG)
    save(CTres)
    if os.path.exists(HGUO_path):
        save(HGUO)