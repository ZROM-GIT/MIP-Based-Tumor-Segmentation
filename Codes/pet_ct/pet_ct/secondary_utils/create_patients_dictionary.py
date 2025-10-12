"""Patient dictionary creator

    Iterates all patients from a given path and saves patient information in a dictionary for future use.
"""

from pathlib import Path
import json
import nibabel as nib
from tqdm import tqdm
import pandas as pd
import datetime

# filename
filename = Path('patients_PET_data.json')

# Create dictionary for patients
patients = dict()

# Paths
Input_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis')
metadata_path = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/Clinical Metadata FDG PET_CT Lesions.csv')

# Load Metadata
METADATA = pd.read_csv(metadata_path)
METADATA_PT = METADATA[METADATA.Modality == 'PT']

# Iterate all patients and their trials
trials_paths = sorted(list(Input_path.glob('*/*')))
loader = tqdm(trials_paths)
for trial_path in loader:
    trial_name = str(trial_path.parent.name) + '/' + str(trial_path.name)
    subject_id = str(trial_path.parent.name)
    suv_nifti = nib.load(trial_path/'SUV.nii.gz')
    dim = suv_nifti.header['dim'][1:4].tolist()
    date = trial_path.name.split('-')[0:3]
    date = datetime.date(int(date[2]), int(date[0]), int(date[1]))
    date = date.strftime('%-m/%-d/%Y')

    diagnosis = str(METADATA_PT[(METADATA_PT['Subject ID'] == subject_id) & (METADATA_PT['Study Date'] == date)].diagnosis).split()[1]

    patients[trial_name] = {'full_path': str(trial_path),
                            'Subject_ID': subject_id,
                            'trial_name': str(trial_path.name),
                            'dim': dim,
                            'date': date,
                            'num_of_slices': dim[-1],
                            'diagnosis': diagnosis
                            }

json.dump(patients, filename.open('w'), indent=4)

