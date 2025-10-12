# IMPORTS
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import cupy as cp
import monai
import matplotlib.pyplot as plt
from tqdm import tqdm
from pet_ct.secondary_utils.createMIP_new import create_mip_new
from pet_ct.secondary_utils.AffineMatrices import affine_matrices

# Hyperparameters
num_of_mips = 80
multi_label = False
starting_angle = 0
ending_angle = 180
generate_ct_only = True  # <-- Set this to True for CT-only MIP generation

# Input and Output paths
Input_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D')
Output_path = Path(f'/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_MIPs/MIPs{num_of_mips}')

load = monai.transforms.Compose([
    monai.transforms.LoadImage(image_only=True, ensure_channel_first=True),
    monai.transforms.Orientation(axcodes='ILP')
])

trials_paths = sorted(list(Input_path.glob('*/*')))
print('Patients paths loaded...')

for j, trial_path in tqdm(enumerate(trials_paths), desc='Case Studies', total=len(trials_paths)):
    SUV_inds_path = Output_path / trial_path.parent.name / trial_path.name / 'SUV_inds.nii.gz'
    CT_path = trial_path / 'CTres.nii.gz'

    # Load CT volume
    CT_vol = cp.array(load(CT_path).squeeze())

    output_trial_path = Output_path / trial_path.parent.name / trial_path.name
    output_trial_path.mkdir(parents=True, exist_ok=True)

    affine = affine_matrices['SRA']

    if generate_ct_only:
        # CT-only MIP generation using saved SUV_inds
        SUV_inds = cp.array(load(SUV_inds_path).squeeze())
        CT_MIPs = cp.zeros_like(SUV_inds)
        all_angles = np.linspace(start=starting_angle, stop=(ending_angle - ending_angle / num_of_mips),
                                 num=num_of_mips)

        for i in range(num_of_mips):
            angle = all_angles[i]
            inds = SUV_inds[:, :, i].astype(cp.int32)
            inds_expanded = inds[:, :, cp.newaxis]

            # Rotate CT volume for current angle
            rad_angle = (2 * np.pi) * (angle / 360)
            rotation = monai.transforms.Affine(
                rotate_params=(rad_angle, 0, 0),
                image_only=True,
                padding_mode='zeros',
                mode='bilinear',  # or 'nearest' if desired
                device='cuda'
            )
            CT_rot = cp.array(rotation(CT_vol))

            # Gather rotated voxel values at depth indices
            CT_mip = cp.take_along_axis(CT_rot, inds_expanded, axis=2).squeeze(axis=2)
            CT_MIPs[:, :, i] = CT_mip

        ct_nifti = nib.Nifti1Image(CT_MIPs.get(), affine)
        nib.save(ct_nifti, f'{output_trial_path}/CT.nii.gz')
        print(f'[CT-ONLY] Patient {j + 1} done!')
        continue  # Skip rest of full-mode processing

    # Full generation
    SUV_path = trial_path / 'SUV.nii.gz'
    SEG_path = trial_path / 'SEG.nii.gz'
    HGUO_path = trial_path / 'HGUO.nii.gz'

    SUV = cp.array(load(SUV_path).squeeze())
    SEG = cp.array(load(SEG_path).squeeze())
    if os.path.exists(HGUO_path):
        HGUO = cp.array(load(HGUO_path).squeeze())

    if os.path.exists(HGUO_path) and not multi_label:
        HGUO[HGUO != 0] = 1

    SUV_3D = cp.zeros((SUV.shape[0], SUV.shape[1], num_of_mips))
    SUV_inds_3D = cp.zeros_like(SUV_3D)
    SEG_3D = cp.zeros_like(SUV_3D)
    if os.path.exists(HGUO_path):
        HGUO_3D = cp.zeros_like(SUV_3D)

    CT_3D = cp.zeros_like(SUV_3D)  # <-- To hold CT MIPs

    all_angles = np.linspace(start=starting_angle, stop=(ending_angle - ending_angle / num_of_mips), num=num_of_mips)

    for i in range(num_of_mips):
        angle = all_angles[i]
        SUV_mip, SUV_inds = create_mip_new(SUV, return_inds=True, horizontal_angle=angle, modality='suv', device='cuda')
        SEG_mip = create_mip_new(SEG, return_inds=False, horizontal_angle=angle, modality='seg', device='cuda')
        if os.path.exists(HGUO_path):
            HGUO_mip = create_mip_new(HGUO, return_inds=False, horizontal_angle=angle, modality='hguo', device='cuda')

        # Generate CT MIP using SUV_inds
        inds = SUV_inds.astype(cp.int32)
        CT_mip = cp.zeros_like(SUV_mip)
        for x in range(CT_vol.shape[0]):
            for y in range(CT_vol.shape[1]):
                CT_mip[x, y] = CT_vol[x, y, inds[x, y]]

        SUV_3D[:, :, i] = SUV_mip
        SUV_inds_3D[:, :, i] = SUV_inds
        SEG_3D[:, :, i] = SEG_mip
        CT_3D[:, :, i] = CT_mip
        if os.path.exists(HGUO_path):
            HGUO_3D[:, :, i] = HGUO_mip

    # Save all MIPs
    nib.save(nib.Nifti1Image(SUV_3D.get(), affine), f'{output_trial_path}/SUV.nii.gz')
    nib.save(nib.Nifti1Image(SUV_inds_3D.get(), affine), f'{output_trial_path}/SUV_inds.nii.gz')
    nib.save(nib.Nifti1Image(SEG_3D.get(), affine), f'{output_trial_path}/SEG.nii.gz')
    nib.save(nib.Nifti1Image(CT_3D.get(), affine), f'{output_trial_path}/CT.nii.gz')
    if os.path.exists(HGUO_path):
        nib.save(nib.Nifti1Image(HGUO_3D.get(), affine), f'{output_trial_path}/HGUO.nii.gz')

    print(f'Patient {j + 1} done!')

print('All patients processed.')
