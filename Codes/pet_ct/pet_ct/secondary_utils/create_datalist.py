"""
Data list creater

This file uses a datalist configuration file inserted under the parameter args.
It creates a list of string names of patient+trial that are valid under the configuration file's restrictions.
Data list file is saved in written in configuration file.
"""

# IMPORTS
import os
from pet_ct.main_utils.load_args import load_args
import json
import pandas as pd

# Import arguments for patients
args = load_args('/mnt/sda1/PET/datalists/datalists_confiurations/non_negative.yaml')

# Import patients data
patients_data = json.load(open(args.patients_data_path, 'r'))
patients_excel_data = pd.read_csv(args.patients_excel_path)

patients = sorted(list(patients_data.keys()))

# Filtering
for filt, val in args.filter.items():
    if not val:
        continue
    if filt == 'alphabetically':
        [min_ind, max_ind] = args.alphabetically
        patients = sorted(patients)[min_ind: max_ind]
    if filt == 'num_of_slices':
        num_of_slices = args.num_of_slices
        for patient in patients:
            if patients_data[patient]['num_of_slices'] != num_of_slices:
                patients.remove(patient)
    patients2 = patients.copy()
    if filt == 'num_of_slices_range':
        min_val, max_val = args.num_of_slices_range
        for patient in patients:
            if min_val <= patients_data[patient]['num_of_slices'] <= max_val:
                print(patients_data[patient]['num_of_slices'])
                continue
            else:
                patients.remove(patient)
    if filt == 'organs':
        for patient in patients2:
            patient_name = patients_data[patient]['Subject_ID']
            date = patients_data[patient]['date']
            patient_ind = patients_excel_data.index[(patients_excel_data['Study Date'] == date) &
                                                    (patients_excel_data['Subject ID'] == patient_name) &
                                                    (patients_excel_data['Modality'] == 'PT')].tolist()[-1]
            for organ, thresh in args.organs.items():
                if (thresh == 'any') or (thresh == patients_excel_data[organ][patient_ind]):
                    continue
                elif patient in patients:
                    patients.remove(patient)
    if filt == 'modality':
        for patient in patients2:
            diagnosis = patients_data[patient]['diagnosis']
            if args['modality'][diagnosis]:
                continue
            else:
                patients.remove(patient)


path_to_file = '/mnt/sda1/PET/datalists'
path_to_file = os.path.join(path_to_file, args.filename)
with open(path_to_file, 'w') as f:
    for patient_path in patients:
        f.write(patient_path + '\n')