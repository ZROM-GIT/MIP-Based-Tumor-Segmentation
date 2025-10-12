import json
import os
import re


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def change_parent_path(data, new_parent_path, keys_to_modify):
    # Pattern to match 'PETCT_' followed by exactly 10 alphanumeric characters (letters or digits)
    pattern = re.compile(r'.*(PETCT_[A-Za-z0-9]{10})')
    for key in keys_to_modify:
        if key in data:
            old_path = data[key]
            # Use re.sub to replace the part of the path before 'PETCT_xxxxxxxxxx' pattern with the new parent path
            new_path = re.sub(pattern, os.path.join(new_parent_path, r'\1'), old_path)
            data[key] = new_path


def get_keys_to_modify(data):
    print("Available keys to modify:")
    for key in data:
        print(key)
    keys_to_modify = input("Enter the keys you want to modify, separated by commas: ").split(',')
    keys_to_modify = [key.strip() for key in keys_to_modify]
    return keys_to_modify


def main():
    json_file_path = '/mnt/sda1/PET/json_datasets/num_of_mips_comparison/MIPs48/MIPs48_75th_25vth_IncSplit_0_180_fold5.json'  # Replace with the path to your JSON file
    new_json_file_path = '/mnt/sda1/PET/json_datasets/num_of_mips_comparison/MIPs48/MIPs48_fold5.json'  # Replace with the desired new file path

    data = load_json(json_file_path)

    # Keys you want to modify
    keys_to_modify = ['SUV_mips', 'SEG_mips', 'max_inds_mips', 'HGUO_mips', 'CT_mips']
    new_parent_path = '/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs48'  # Replace with your new parent path

    # Iterate over all keys in the outer dictionary
    for outer_dict_key, inner_list in data.items():
        # Iterate over all dictionaries in each list
        for inner_dict in inner_list:
            change_parent_path(inner_dict, new_parent_path, keys_to_modify)

    save_json(data, new_json_file_path)
    print(f"Paths updated successfully! New JSON file saved as '{new_json_file_path}'.")


if __name__ == "__main__":
    main()


# def main():
#     json_file_path = '/mnt/sda1/PET/json_datasets/num_of_mips_comparison/MIPs16/MIPs16_75th_25vth_IncSplit_0_180_fold5.json'  # Replace with the path to your JSON file
#     new_json_file_path = '/mnt/sda1/PET/json_datasets/num_of_mips_comparison/MIPs64/MIPs64_75th_25vth_IncSplit_0_180_fold5.json'  # Replace with the desired new file path
#
#     data = load_json(json_file_path)
#
#     # Assuming the structure is { "outer_dict_key": [ { "inner_dict_key": "value" } ] }
#     outer_dict_key = list(data.keys())[0]
#     inner_list = data[outer_dict_key]
#     inner_dict = inner_list[0]
#
# #    keys_to_modify = get_keys_to_modify(inner_dict)
#     keys_to_modify = ['SUV_mips', 'SEG_mips', 'max_inds_mips', 'HGUO_mips']
#     new_parent_path = '/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs64_75th_25vth_IncSplit_0_180'  # Replace with your new parent path
#
#     change_parent_path(inner_dict, new_parent_path, keys_to_modify)
#
#     save_json(data, new_json_file_path)
#     print(f"Paths updated successfully! New JSON file saved as '{new_json_file_path}'.")
#
#
# if __name__ == "__main__":
#     main()
