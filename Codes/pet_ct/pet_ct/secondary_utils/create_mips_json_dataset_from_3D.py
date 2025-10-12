import json
import re
import os


def extract_patient_name(path):
    match = re.search(r'PETCT\w+.*', path)
    return match.group(0) if match else None


def add_patient_name_to_dataset(data, new_path, new_keys_and_values):
    for key, value in data.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    updates = {}  # Collect updates here
                    for inner_key, inner_value in item.items():
                        if isinstance(inner_value, str):
                            patient_end = extract_patient_name(inner_value)
                            patient_name = patient_end.split('/')[:-1]
                            for new_key, new_value in new_keys_and_values.items():
                                updates[new_key] = os.path.join(new_path, patient_name[0], patient_name[1] , new_value)
                    # Apply updates after iteration
                    item.update(updates)
    return data


def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


def save_json(data, filepath):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    input_filepath = '/mnt/sda1/PET/json_datasets/AllData3D.json'
    output_filepath = '/mnt/sda1/PET/json_datasets/MIPs16_75th_25vth_IncSplit_0_180_With3D.json'

    new_path = '/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs16_75th_25vth_IncSplit_0_180'
    new_keys_and_values = {
        'SUV_mips': 'SUV.nii.gz',
        'SEG_mips': 'SEG.nii.gz',
        'max_inds_mips': 'SUV_inds.nii.gz',
        'HGUO_mips': 'HGUO.nii.gz',
    }

    dataset = load_json(input_filepath)
    updated_dataset = add_patient_name_to_dataset(dataset, new_path=new_path, new_keys_and_values=new_keys_and_values)
    save_json(updated_dataset, output_filepath)

    print(f"Updated dataset saved to {output_filepath}")
