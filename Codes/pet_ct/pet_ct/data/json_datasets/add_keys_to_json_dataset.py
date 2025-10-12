import json
import os


def add_keys_to_case(case_dict, keys_config):
    """
    Add new keys to a single case dictionary based on the keys configuration.

    Args:
    case_dict (dict): A single case dictionary from the dataset
    keys_config (dict): Dictionary where keys are the new key names to add,
                       and values are lists of [main_path, filename]
    """
    full_case_name = case_dict['full_case_name']

    for new_key, path_config in keys_config.items():
        main_path, filename = path_config

        # Ensure main_path ends with "/"
        if not main_path.endswith('/'):
            main_path += '/'

        # Construct the full path: main_path + full_case_name + filename
        full_path = main_path + full_case_name + '/' + filename
        case_dict[new_key] = full_path


def add_keys_to_dataset(data_dict, keys_config):
    """
    Add new keys to all cases in the dataset.

    Args:
    data_dict (dict): The main dataset dictionary
    keys_config (dict): Dictionary where keys are the new key names to add,
                       and values are lists of [main_path, filename]
    """
    for key, value in data_dict.items():
        if isinstance(value, list):
            for case_dict in value:
                add_keys_to_case(case_dict, keys_config)


def main(json_path, keys_config):
    """
    Main function to process the JSON dataset and add new keys.

    Args:
    json_path (str): Path to the JSON file
    keys_config (dict): Dictionary where keys are the new key names to add,
                       and values are lists of [main_path, filename]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    add_keys_to_dataset(data, keys_config)

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Successfully added keys {list(keys_config.keys())} to {json_path}")


if __name__ == "__main__":
    keys_to_add = {
        'UNETR_encoder_features': ['Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs16_features',
                                   'UNETR_features_fold1.nii.gz'],
    }

    json_path = '/mnt/sda1/Research/PET_CT_TS/Configs/json_datasets/num_of_mips_comparison/MIPs16/MIPs16_75th_25vth_IncSplit_0_180_fold1.json'

    try:
        main(json_path, keys_to_add)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")

    print("Processing complete!")