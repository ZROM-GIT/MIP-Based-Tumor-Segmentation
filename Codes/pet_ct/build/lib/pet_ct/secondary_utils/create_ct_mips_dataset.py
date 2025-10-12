import numpy as np
import nibabel as nib
import monai
import matplotlib.pyplot as plt
from pathlib2 import Path
from tqdm import tqdm
from create_CT_mip import *
from pet_ct.secondary_utils.AffineMatrices import affine_matrices

ct_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis')
mips_dataset_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs80_75th_25vth_IncSplit_0_180')

ct_trials_paths = sorted(list(ct_path.glob('*/*')))
mips_trials_paths = sorted(list(mips_dataset_path.glob('*/*')))
print('Patients paths loaded...')

for j, trial_path in enumerate(tqdm(ct_trials_paths, desc='Case Studies')):
    mips_trial_path = mips_trials_paths[j]

    ct_trial_path = trial_path / 'CTres.nii.gz'
    suv_mips_path = mips_trial_path / 'SUV.nii.gz'
    suv_inds_path = mips_trial_path / 'SUV_inds.nii.gz'

    CT = load_nifti_image(ct_trial_path)[0]
    SUV_mips = load_nifti_image(suv_mips_path)[0]
    SUV_inds = load_nifti_image(suv_inds_path)[0].astype(int)

    CT_mips = create_ct_mips(ct_data=CT, suv_inds=SUV_inds, starting_angle=0, ending_angle=180, device='cpu')

    output_trial_path = mips_dataset_path / trial_path.parent.name / trial_path.name
    output_trial_path.mkdir(parents=True, exist_ok=True)
    affine = affine_matrices['SLA']

    ct_mips_nif = nib.Nifti1Image(CT_mips, affine=affine)
    nib.save(ct_mips_nif, output_trial_path / 'CT.nii.gz')

print('CT MIPs dataset created.')


