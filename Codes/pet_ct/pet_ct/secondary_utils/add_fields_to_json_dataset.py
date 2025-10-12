import json
import os
from munch import Munch


def replace_last_part(file_path, new_value):
    """
    Replace the last part of the file name (before the extension) with the specified new value,
    correctly handling multi-part extensions (e.g., '.nii.gz').

    Args:
    file_path (str): The original file path.
    new_value (str): The new value to replace the last part of the file name.

    Returns:
    str: The updated file path with the last part replaced.
    """
    # Split the path into directory and filename
    directory, filename = os.path.split(file_path)

    # Handle multi-part extensions like '.nii.gz'
    if filename.endswith('.nii.gz'):
        file_base = filename[:-7]  # Remove both '.nii.gz'
        file_ext = '.nii.gz'
    else:
        file_base, file_ext = os.path.splitext(filename)  # Normal single extension

    # Replace the base file name (before extension) with new_value
    new_filename = new_value + file_ext

    # Reconstruct the full path with the new filename
    new_path = os.path.join(directory, new_filename)

    return new_path

def add_ct_key(data_dict, args):
    key_to_add = args.key_to_add
    value_to_add = args.value_to_add
    key_to_take_from = args.key_to_take_from

    original_path = data_dict[key_to_take_from]

    data_dict[key_to_add] = replace_last_part(original_path, value_to_add)

def main(json_path, args):
    with open(json_path, 'r') as f:
        data = json.load(f)

    for key, value in data.items():
        for inner_dict in value:
            add_ct_key(inner_dict, args)

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    json_path = '/mnt/sda1/PET/json_datasets/num_of_mips_comparison/MIPs48/MIPs48_75th_25vth_IncSplit_0_180_fold5.json'
    args = Munch({'key_to_add': 'CT_3D',
                 'value_to_add': 'CTres',
                 'key_to_take_from': 'SUV_3D'})

    main(json_path, args)