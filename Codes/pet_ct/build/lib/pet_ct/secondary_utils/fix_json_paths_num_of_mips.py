
import json
from pathlib2 import Path
import os
from copy import deepcopy

path_to_save = Path('/mnt/sda1/PET/json_datasets/')

num_of_mips_to_take_from = 16
list_of_num_of_mips = [32, 48, 64]

for i in range(1,6):
    path = f'/mnt/sda1/PET/json_datasets/MIPs{num_of_mips_to_take_from}/MIPs{num_of_mips_to_take_from}_75th_25vth_IncSplit_0_180_fold{i}.json'

    # Open JSON file
    f = open(path)

    # Returns JSON object as dict
    data = json.load(f)

    for num in list_of_num_of_mips:
        new_path = path.replace('MIPs' + str(num_of_mips_to_take_from), 'MIPs' + str(num))
        new_data = deepcopy(data)
        for set, set_list in new_data.items():
            for j, patient_dict in enumerate(set_list):
                for key, key_path in patient_dict.items():
                    new_data[set][j][key] = key_path.replace('MIPs' + str(num_of_mips_to_take_from), 'MIPs' + str(num))

        json.dump(new_data,
                  Path(new_path).open('w'),
                  indent=4)
        print(f'output file written to {new_path}')

