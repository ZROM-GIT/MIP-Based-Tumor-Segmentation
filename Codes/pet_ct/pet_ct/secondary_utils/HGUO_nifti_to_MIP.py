"""
This file is meant to turn 3D nifti segmentations into a dataset of 2D segmentation MIPs in different angles
"""

# IMPORTS
from pathlib import Path
import os
import numpy as np
import monai
import createMIP
import shutil

# Hyperparameters
number_of_MIP_directions = 720
number_of_trials = 100  # Depends on how many I segmented
multi_label = False

# Input and Output paths:
Input_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis')  # Path to all nifti files
Output_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs')  # Path to all MIPs

load = monai.transforms.Compose([monai.transforms.LoadImage(image_only=True, ensure_channel_first=True),
                                 monai.transforms.Orientation(axcodes='IPL')])

trials_paths = sorted(list(Input_path.glob('*/*')))[9:10] #TODO change range to 0:number_of_trials
print('Paths loaded...')

for trial_path in trials_paths:
    HGUO_path = trial_path / 'HGUO.nii.gz'

    # Load patient 3D HGUO segmentation as numpy array using Monai
    np_HGUO = load(HGUO_path).squeeze().astype('int32')
    print('Numpy array created...')

    # Turn all HGUO segmentation into one class
    # TODO: Notice that this is temporary, maybe we change to multi-class
    if multi_label:
        continue
    else:
        np_HGUO[np_HGUO != 0] = 1

    # Pre-allocate 3D numpy array for all 2D MIP stacks
    np_HGUO_3D = np.zeros([number_of_MIP_directions, np_HGUO.shape[1], np_HGUO.shape[0]])

    all_angles = np.linspace(0, 360 - 360 / number_of_MIP_directions, num=number_of_MIP_directions)

    for i in range(0, number_of_MIP_directions):
        angle = all_angles[i]
        # np_HGUO_rotated = createMIP(np_HGUO, horizontal_angle=angle, modality='hguo')

        # np_HGUO_3D[i, :, :] = np_HGUO_rotated

    print(f'MIPs created...')

    output_trial_path = Output_path / trial_path.parent.name / trial_path.name
    save_HGUO = monai.transforms.SaveImage(output_dir=str(output_trial_path), output_postfix='HGUO', output_ext='.nii.gz',
                                           separate_folder=False, channel_dim=None)
    save_HGUO(np_HGUO_3D)
    print('Trial finished')




