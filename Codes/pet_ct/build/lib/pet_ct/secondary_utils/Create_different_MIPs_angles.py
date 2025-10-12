'''
This file is meant for the creation of individual files that contain different MIPs angles
'''

from pathlib import Path
import numpy as np
import monai
import nibabel as nib
import time

start = time.time()

# Path to take patients from
input_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs/MIPs720')

# Number of angles to take from MIPs
angles = [4, 6, 8 ,10, 12, 16, 24, 32, 48, 72]

# All trial paths
trials_paths = sorted(list(input_path.glob('*/*')))

for trial_path in trials_paths:
    SUV_EXT, SUV_ind_EXT, SEG_EXT, HGUO_EXT = '0_SUV.nii.gz', '0_SUV_inds.nii.gz', '0_SEG.nii.gz', '0_HGUO.nii.gz'
    SUV_PATH, SUV_ind_PATH, SEG_PATH, HGUO_PATH = trial_path / SUV_EXT, trial_path / SUV_ind_EXT, trial_path / SEG_EXT, trial_path / HGUO_EXT
    SUV_data, SUV_ind_data, SEG_data = nib.load(SUV_PATH).get_fdata(), nib.load(SUV_ind_PATH).get_fdata(), nib.load(SEG_PATH).get_fdata()

    if Path.exists(HGUO_PATH):
        HGUO_data = nib.load(HGUO_PATH).get_fdata()

    # Create different SUV and SEG data angle reductions
    for i, angs in enumerate(angles):
        inds = np.linspace(start=0, stop=180, num=angs, dtype=int) * 2
        SUV_data_new, SUV_ind_data_new , SEG_data_new = SUV_data[inds, :, :], SUV_ind_data[inds, :, :], SEG_data[inds, :, :]
        if Path.exists(HGUO_PATH):
            HGUO_data_new = HGUO_data[inds, :, :]
        trial_path_new = str(trial_path).replace('MIPs720', 'MIPs'+str(angs))
        save_SUV = monai.transforms.SaveImage(output_dir=trial_path_new, output_postfix='SUV', output_ext='.nii.gz',
                                           separate_folder=False, channel_dim=None)
        save_SEG = monai.transforms.SaveImage(output_dir=trial_path_new, output_postfix='SEG', output_ext='.nii.gz',
                                              separate_folder=False, channel_dim=None)
        save_SUV_ind = monai.transforms.SaveImage(output_dir=trial_path_new, output_postfix='SUV_inds', output_ext='.nii.gz',
                                                  separate_folder=False, channel_dim=None)
        save_HGUO = monai.transforms.SaveImage(output_dir=trial_path_new, output_postfix='HGUO', output_ext='.nii.gz',
                                                  separate_folder=False, channel_dim=None)
        save_SUV(SUV_data_new)
        save_SEG(SEG_data_new)
        save_SUV_ind(SUV_ind_data_new)
        if Path.exists(HGUO_PATH):
            save_HGUO(HGUO_data_new)

end = time.time()
time_elapsed = end - start
print(f'Total run time: {time_elapsed} seconds')
print(f'Total run time: {time_elapsed / 60} minutes')
