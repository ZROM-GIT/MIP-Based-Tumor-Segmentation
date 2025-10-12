import copy
import os.path
from pathlib2 import Path
import json
import yaml
import munch
import random


def fix_json(args, json_dict):
    with open(args.patients_dict_path) as f:
        patients_dict = json.load(f)

    json_dict2 = copy.deepcopy(json_dict)
    for split, list_of_patients_per_split in json_dict2.items():
        for i, patient in enumerate(list_of_patients_per_split):
            patient_key = '/'.join(patient['SUV_3D'].split('/')[-3:-1])
            diagnosis = patients_dict[patient_key]['diagnosis']
            if diagnosis in args['modalities_to_remove']:
                json_dict[split].remove(patient)

    return json_dict

def main(args):
    outPath = Path(getattr(args, 'new_json_path', '/mnt/sda1/PET/json_datasets'))
    outPath.mkdir(parents=True, exist_ok=True)

    jsonName = args.new_json_name
    outPath = outPath.joinpath(jsonName)

    with open(args.current_json_path) as f:
        data = json.load(f)

    # Fix json dictionary
    new_data = fix_json(args, data)


    json.dump(new_data,
              outPath.open('w'),
              indent=4)
    print(f'output file written to {outPath}')


if __name__ == '__main__':
    args = {'current_json_path': '/mnt/sda1/PET/json_datasets/num_of_mips_comparison/MIPs48/MIPs48_75th_25vth_IncSplit_0_180_fold2.json',
            'new_json_path': '/mnt/sda1/PET/json_datasets/non_healthy/MIPs48',
            'new_json_name': 'MIPs48_75th_25vth_IncSplit_0_180_non_healthy_fold2.json',
            'patients_dict_path': '/mnt/sda1/PET/json_datasets/patients_PET_data.json',
            'modalities_to_remove': ['NEGATIVE']
            }
    args = munch.Munch(args)
    main(args)