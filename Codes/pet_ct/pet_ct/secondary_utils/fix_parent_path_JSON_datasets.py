import json
import re

original_json_path = '/mnt/sda1/Research/PET_CT_TS/Configs/json_datasets/num_of_mips_comparison/MIPs48/MIPs48_75th_25vth_IncSplit_0_180_fold5.json'
new_json_path = '/mnt/sda1/Research/PET_CT_TS/Configs/json_datasets/num_of_mips_comparison/MIPs48/MIPs48_fold5.json'
new_path_prefix = 'Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs48'  # Adjust this if needed


def update_paths(data, new_path_prefix):
    pattern = re.compile(r'.*?(PETCT_\w+/.+)')  # Ensure it captures PETCT_xxx and everything after it

    for outer_key, inner_list in data.items():
        if isinstance(inner_list, list):
            for inner_dict in inner_list:
                if isinstance(inner_dict, dict):
                    for key, value in inner_dict.items():
                        if isinstance(value, str):
                            match = pattern.search(value)
                            if match:
                                new_path = new_path_prefix.rstrip('/') + '/' + match.group(1).lstrip(
                                    '/')  # Ensure proper merging of paths
                                inner_dict[key] = new_path
    return data


with open(original_json_path, 'r') as file:
    data = json.load(file)

new_data = update_paths(data, new_path_prefix)

with open(new_json_path, 'w') as file:
    json.dump(new_data, file, indent=4)