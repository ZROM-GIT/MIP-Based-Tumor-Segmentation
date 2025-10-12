# """
# This file is meant to create new MIPs with screening of tumors fixed
# """
#
# import monai
# import nibabel as nib
# import numpy as np
# import os
# from pathlib2 import Path
# import torch
# from tqdm import tqdm
#
# from create_screened_mips import create_screened_mips
# from pet_ct.secondary_utils.AffineMatrices import affine_matrices
#
# # Hyperparameters
# volume_threshold = 0 # Any tumor with smaller volume than this will be removed [mm^3]
# threshold = 0  # Percentage of tumor to be "seen" to stay in dataset
# split_tumors = False
# filter_split_tumors_by_contrast = False
# filter_split_tumors_by_gradient = False
# num_of_mips = 32
# starting_angle = 0
# ending_angle = 180
#
# # Input and Output paths:
# Input_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D')  # Path to all 3D nifti files
# # Path to all MIPs
# # if not split_tumors:
# #     Output_path = Path(f'/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs{num_of_mips}_{threshold}th_{volume_threshold}vth_{starting_angle}_{ending_angle}')
# # elif split_tumors & ((not filter_split_tumors_by_contrast) or (not filter_split_tumors_by_gradient)):
# #     Output_path = Path(f'/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs{num_of_mips}_{threshold}th_{volume_threshold}vth_IncSplit_{starting_angle}_{ending_angle}')
# # elif split_tumors & filter_split_tumors_by_contrast:
# #     Output_path = Path(f'/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs{num_of_mips}_{threshold}th_{volume_threshold}vth_IncSplit_FiltContrSplit_{starting_angle}_{ending_angle}')
# # elif split_tumors & filter_split_tumors_by_gradient:
# #     Output_path = Path(f'/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs{num_of_mips}_{threshold}th_{volume_threshold}vth_IncSplit_FiltGradSplit_{starting_angle}_{ending_angle}')
#
# Output_path = Path(f'/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_MIPs/MIPs{num_of_mips}')
#
#
#
# load = monai.transforms.Compose([monai.transforms.LoadImage(image_only=True, ensure_channel_first=True),
#                                  monai.transforms.Orientation(axcodes='ILP')])
#
# trials_paths = sorted(list(Input_path.glob('*/*')))
# print('Patients paths loaded...')
#
# for j, trial_path in enumerate(tqdm(trials_paths, desc='Case Studies')):
#
#     # All files paths:
#     PET_path = trial_path / 'PET.nii.gz'
#     CT_path = trial_path / 'CT.nii.gz'
#     SEG_path = trial_path / 'SEG.nii.gz'
#     SUV_path = trial_path / 'SUV.nii.gz'
#     CTres_path = trial_path / 'CTres.nii.gz'
#     HGUO_path = trial_path / 'HGUO.nii.gz'
#
#     # Load patient 3D SUV & SEG using Monai
#     SUV = load(SUV_path)[0]
#     SEG = load(SEG_path)[0]
#     if os.path.exists(HGUO_path):
#         HGUO = load(HGUO_path)[0]
#
#     all_angles = np.linspace(start=starting_angle, stop=(ending_angle - ending_angle / num_of_mips), num=num_of_mips)
#
#     if os.path.exists(HGUO_path):
#        suv_mips, suv_inds_mips, seg_mips, hguo_mips = create_screened_mips(suv=SUV,
#                                                             seg=SEG,
#                                                             hguo=HGUO,
#                                                             horizontal_rot_angles=all_angles,
#                                                             threshold=threshold,
#                                                             volume_threshold=volume_threshold,
#                                                             split_tumors=split_tumors,
#                                                             filter_split_tumors_by_contrast=filter_split_tumors_by_contrast,
#                                                             filter_split_tumors_by_gradient=filter_split_tumors_by_gradient)
#     else:
#        suv_mips, suv_inds_mips, seg_mips = create_screened_mips(suv=SUV,
#                                                                 seg=SEG,
#                                                                 horizontal_rot_angles=all_angles,
#                                                                 threshold=threshold,
#                                                                 volume_threshold=volume_threshold,
#                                                                 split_tumors=split_tumors,
#                                                                 filter_split_tumors_by_contrast=filter_split_tumors_by_contrast,
#                                                                 filter_split_tumors_by_gradient=filter_split_tumors_by_gradient)
#
#     output_trial_path = Output_path / trial_path.parent.name / trial_path.name
#     output_trial_path.mkdir(parents=True, exist_ok=True)
#     affine = affine_matrices['SRA']
#
#     suv_mips_nif = nib.Nifti1Image(suv_mips.numpy(), affine=affine)
#     suv_inds_mips_nif = nib.Nifti1Image(suv_inds_mips.numpy(), affine=affine)
#     seg_mips_nif = nib.Nifti1Image(seg_mips.numpy(), affine=affine)
#     if os.path.exists(HGUO_path):
#         hguo_mips_nif = nib.Nifti1Image(hguo_mips.numpy(), affine=affine)
#
#     nib.save(suv_mips_nif, output_trial_path / 'SUV.nii.gz')
#     nib.save(suv_inds_mips_nif, output_trial_path / 'SUV_inds.nii.gz')
#     nib.save(seg_mips_nif, output_trial_path / 'SEG.nii.gz')
#     if os.path.exists(HGUO_path):
#         nib.save(hguo_mips_nif, output_trial_path / 'HGUO.nii.gz')
#

import numpy as np
import nibabel as nib
import monai
from tqdm import tqdm
from pathlib2 import Path
from pet_ct.secondary_utils.AffineMatrices import affine_matrices
from create_screened_mips import create_screened_mips

# ------------------------
# Configuration
# ------------------------
volume_threshold = 0
threshold = 0
split_tumors = True
filter_by_contrast = False
filter_by_gradient = True
num_of_mips = 16
start_angle = 0
end_angle = 180

input_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D')
output_path = Path(f'/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_MIPs/MIPs{num_of_mips}_AS')
affine = affine_matrices['SRA']
angles = np.linspace(start=start_angle, stop=end_angle - end_angle / num_of_mips, num=num_of_mips)

load_nifti = monai.transforms.Compose([
    monai.transforms.LoadImage(image_only=True, ensure_channel_first=True),
    monai.transforms.Orientation(axcodes='ILP')
])

# ------------------------
# Main Loop
# ------------------------
trials = sorted(input_path.glob('*/*'))
print(f"Loaded {len(trials)} patient paths...")

for trial_path in tqdm(trials, desc='Processing cases'):
    paths = {
        'SUV': trial_path / 'SUV.nii.gz',
        'SEG': trial_path / 'SEG.nii.gz',
        'HGUO': trial_path / 'HGUO.nii.gz'
    }

    # Load required volumes
    SUV = load_nifti(paths['SUV'])[0]
    SEG = load_nifti(paths['SEG'])[0]
    HGUO = load_nifti(paths['HGUO'])[0] if paths['HGUO'].exists() else None

    # Run MIP creation
    mips = create_screened_mips(
        suv=SUV,
        seg=SEG,
        hguo=HGUO,
        horizontal_rot_angles=angles,
        threshold=threshold,
        volume_threshold=volume_threshold,
        split_tumors=split_tumors,
        filter_split_tumors_by_contrast=filter_by_contrast,
        filter_split_tumors_by_gradient=filter_by_gradient
    )

    # Output folder
    out_dir = output_path / trial_path.parent.name / trial_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save MIPs
    suv_mips_nif = nib.Nifti1Image(mips[0].cpu().numpy(), affine)
    suv_inds_nif = nib.Nifti1Image(mips[1].cpu().numpy(), affine)
    seg_mips_nif = nib.Nifti1Image(mips[2].cpu().numpy(), affine)

    nib.save(suv_mips_nif, out_dir / 'SUV.nii.gz')
    nib.save(suv_inds_nif, out_dir / 'SUV_inds.nii.gz')
    nib.save(seg_mips_nif, out_dir / 'SEG.nii.gz')

    if HGUO is not None and len(mips) == 4:
        hguo_mips_nif = nib.Nifti1Image(mips[3].cpu().numpy(), affine)
        nib.save(hguo_mips_nif, out_dir / 'HGUO.nii.gz')
