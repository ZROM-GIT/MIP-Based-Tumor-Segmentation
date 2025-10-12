"""
This file is meant to turn 3D axial image data into a dataset of 2D MIPs in different angles
"""

# IMPORTS
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import monai
from pet_ct.secondary_utils.createMIP import createMIP

# Hyperparameters
number_of_MIP_directions = 720
number_of_trials = 100  # Depends on how many I segmented
multi_label = False

# Input and Output paths:
Input_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis')  # Path to all nifti files
Output_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs/MIPs720')  # Path to all MIPs

load = monai.transforms.Compose([monai.transforms.LoadImage(image_only=True, ensure_channel_first=True),
                                 monai.transforms.Orientation(axcodes='ILP')])

trials_paths = sorted(list(Input_path.glob('*/*')))
print('Patients paths loaded...')

for j, trial_path in tqdm(enumerate(trials_paths)):

    # All files paths:
    PET_path = trial_path / 'PET.nii.gz'
    CT_path = trial_path / 'CT.nii.gz'
    SEG_path = trial_path / 'SEG.nii.gz'
    SUV_path = trial_path / 'SUV.nii.gz'
    CTres_path = trial_path / 'CTres.nii.gz'
    HGUO_path = trial_path / 'HGUO.nii.gz'

    # Load patient 3D PET/SUV & SEG as numpy array using Monai
    # np_PET = load(PET_path).squeeze()
    np_SUV = load(SUV_path).squeeze()
    np_SEG = load(SEG_path).squeeze().astype('int8')
    if os.path.exists(HGUO_path):
        np_HGUO = load(HGUO_path).squeeze().astype('int8')
    print('Numpy arrays created...')

    # Deal with HGUO
    if os.path.exists(HGUO_path):
        if not multi_label:
            np_HGUO[np_HGUO != 0] = 1

    # Pre-allocate 3D numpy array for all 2D MIP stacks
    # np_PET_3D = np.zeros([number_of_MIP_directions, np_PET.shape[1], np_PET.shape[0]])
    np_SUV_3D = np.zeros([number_of_MIP_directions, np_SUV.shape[1], np_SUV.shape[0]])
    np_SUV_inds = np.zeros([number_of_MIP_directions, np_SUV.shape[1], np_SUV.shape[0]])
    np_SEG_3D = np.zeros([number_of_MIP_directions, np_SEG.shape[1], np_SEG.shape[0]])
    if os.path.exists(HGUO_path):
        np_HGUO_3D = np.zeros([number_of_MIP_directions, np_HGUO.shape[1], np_HGUO.shape[0]])

    all_angles = np.linspace(0, 360 - 360 / number_of_MIP_directions, num=number_of_MIP_directions)

    for i in range(0, number_of_MIP_directions):
        angle = all_angles[i]
        np_SUV_rotated, np_SUV_inds_rotated = createMIP(np_SUV, return_inds=True, horizontal_angle=angle, modality='suv')
        # np_PET_rotated = createMIP(np_PET, horizontal_angle=angle, modality='pet')
        np_SEG_rotated = createMIP(np_SEG, horizontal_angle=angle, modality='seg')
        if os.path.exists(HGUO_path):
            np_HGUO_rotated = createMIP(np_HGUO, horizontal_angle=angle, modality='hguo')

        # np_PET_3D[i, :, :] = np_PET_rotated
        np_SUV_3D[i, :, :] = np_SUV_rotated
        np_SUV_inds[i, :, :] = np_SUV_inds_rotated
        np_SEG_3D[i, :, :] = np_SEG_rotated
        if os.path.exists(HGUO_path):
            np_HGUO_3D[i, :, :] = np_HGUO_rotated

    print(f'MIPs created...')

    output_trial_path = Output_path / trial_path.parent.name / trial_path.name
    output_trial_path.mkdir(parents=True, exist_ok=True)

    # save_PET = monai.transforms.SaveImage(output_dir=str(output_trial_path), output_postfix='PET', output_ext='.nii.gz',
    #                                       separate_folder=False, channel_dim=None)
    save_SUV = monai.transforms.SaveImage(output_dir=str(output_trial_path), output_postfix='SUV', output_ext='.nii.gz',
                                          separate_folder=False, channel_dim=None)
    save_SUV_inds = monai.transforms.SaveImage(output_dir=str(output_trial_path), output_postfix='SUV_inds', output_ext='.nii.gz',
                                               separate_folder=False, channel_dim=None)
    save_SEG = monai.transforms.SaveImage(output_dir=str(output_trial_path), output_postfix='SEG', output_ext='.nii.gz',
                                          separate_folder=False, channel_dim=None)
    if os.path.exists(HGUO_path):
         save_HGUO = monai.transforms.SaveImage(output_dir=str(output_trial_path), output_postfix='HGUO',
                                                output_ext='.nii.gz', separate_folder=False, channel_dim=None)

    # save_PET(np_PET_3D)
    # save_SUV(np_SUV_3D)
    # save_SUV_inds(np_SUV_inds)
    # save_SEG(np_SEG_3D)
    # if os.path.exists(HGUO_path):
    #     save_HGUO(np_HGUO_3D)

    print(f'Patient number {j + 1} done!')
print('Patient finished')
