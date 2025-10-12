"""
JSON dataset creater

    Recieves a configuration file as input including:
    - baseDirs: Dirs for the different modalities
    - datalist: Dir for the datalist used to create the JSON
    - datasetname
    - length: Length of the *whole* dataset (Including training, validation and test)
    - valPerc: From the length integer inserted above, what percentage will act as validation.
    - testPerc: From the length integer inserted above, what percentage will act as test.

"""
import copy
# Imports
import os.path
from pathlib2 import Path
import json
import yaml
import munch
import random


def divide_data(datalist, args):
    cross_validation_args = args.cross_validation_arguments

    if hasattr(args, 'patients_dict'):
        test_perc = cross_validation_args['test_perc']

        patients_dict = yaml.load(Path(args.patients_dict).open('r'), Loader=yaml.CLoader)
        patients_dict = munch.Munch(patients_dict)

        # Extract unique diagnosis from patients dictionary
        diags = set([patients_dict[patient]['diagnosis'] for patient in
                     patients_dict])  # diags = ['NEGATIVE', 'MELANOMA', 'LUNG_CANCER', 'LYMPHOMA']

        # Create dictionary {'diagnosis' : [list of patients with the diagnosis]}
        datadict = {diag: [] for diag in diags}

        # Randomize datalist (Order of patients to choose from)
        random.shuffle(datalist)

        for patient in datalist:
            datadict[patients_dict[patient]['diagnosis']].append(patient)

        if cross_validation_args['act'] == True:
            test_perc = cross_validation_args['test_perc']

            test_set = []
            for diag in datadict:
                num_of_patients = len(datadict[diag])
                to_take = round((num_of_patients * test_perc) / 100)
                test_set.extend(datadict[diag][:to_take])
                for i in range(to_take):
                    datadict[diag].pop(0)

            k_folds = cross_validation_args['k_folds']
            fold_perc_delta = 1/k_folds
            folds = [{'training': [], 'validation': [], 'test': test_set} for fold in range(k_folds)]

            for fold_num in range(1, cross_validation_args['k_folds'] + 1):
                data = copy.deepcopy(datadict)
                fold_perc_start = fold_perc_delta * (fold_num - 1)
                fold_perc_end = fold_perc_delta * (fold_num)
                for diag in datadict:
                    to_take_start = round(len(datadict[diag]) * fold_perc_start)
                    to_take_end = round(len(datadict[diag]) * fold_perc_end)
                    for i in range(to_take_end - to_take_start):
                        folds[fold_num - 1]['validation'].append(data[diag].pop(to_take_start))
                    folds[fold_num - 1]['training'].extend(data[diag])

            return folds

        else:

            trainPerc = args.trainPerc
            valPerc = args.valPerc
            testPerc = args.testPerc

            if ((trainPerc + valPerc + testPerc) != 100) or (not 0 <= trainPerc <= 100) or (not 0 <= valPerc <= 100) or (not 0 <= testPerc <= 100):
                raise ValueError('Percentages of train/val/test are invalid!')

            for diag in datadict:
                num_of_patiens = len(datadict[diag])
                train_num = round(num_of_patiens * (trainPerc/100))
                val_num = round(num_of_patiens * (valPerc/100))
                divDict = {'training': datadict[diag][0: train_num], 'validation': datadict[diag][train_num: (train_num + val_num)], 'test': datadict[diag][(train_num + val_num) : ]}
                datadict[diag] = divDict

            allData = {'training': [], 'validation': [], 'test': []}
            for mod in allData:
                for diag in datadict:
                    allData[mod].extend(datadict[diag][mod])

    else:
        # TODO: Figure out and fix what to do if there is no patients dictionary
        trainPerc = args.trainPerc
        valPerc = args.valPerc
        testPerc = args.testPerc
        train_num = round(len(datalist) * (trainPerc/100))
        val_num = round(len(datalist) * (valPerc/100))
        if ((trainPerc + valPerc + testPerc) != 100) or (not 0 <= trainPerc <= 100) or (not 0 <= valPerc <= 100) or (not 0 <= testPerc <= 100):
            raise ValueError('Percentages of train/val/test are invalid!')

        allData = {'training': datalist[0: train_num], 'validation': datalist[train_num: (train_num + val_num)], 'test': datalist[(train_num + val_num) : ]}

    return allData

def create_dict(allData, baseDirs, args):
    final_dict = {'training': [], 'validation': [], 'test' : []}
    directory_ending = getattr(args, 'directory_ending', dict())
    patients_dict = yaml.load(Path(args.patients_dict).open('r'), Loader=yaml.CLoader)
    patients_dict = munch.Munch(patients_dict)
    for mod in final_dict:
        patients_list = allData[mod]
        for patient in patients_list:
            paths = {}
            for dir_name, dir in baseDirs.items():
                if dir_name == 'SUV_mips':
                    ending = directory_ending[dir_name] if dir_name in directory_ending.keys() else '0_SUV.nii.gz'
                elif dir_name == 'SEG_mips':
                    ending = directory_ending[dir_name] if dir_name in directory_ending.keys() else '0_SEG.nii.gz'
                elif dir_name == 'max_inds_mips':
                    ending = directory_ending[dir_name] if dir_name in directory_ending.keys() else '0_SUV_inds.nii.gz'
                elif dir_name == 'HGUO_mips':
                    ending = directory_ending[dir_name] if dir_name in directory_ending.keys() else '0_HGUO.nii.gz'
                elif dir_name == 'SUV_3D':
                    ending = directory_ending[dir_name] if dir_name in directory_ending.keys() else 'SUV.nii.gz'
                elif dir_name == 'SEG_3D':
                    ending = directory_ending[dir_name] if dir_name in directory_ending.keys() else 'SEG.nii.gz'
                elif dir_name == 'HGUO_3D':
                    ending = directory_ending[dir_name] if dir_name in directory_ending.keys() else 'HGUO.nii.gz'
                path = os.path.join(dir, patient, ending) if dir_name in baseDirs.keys() else ''
                if dir_name == 'label':
                    path = patients_dict[patient]['diagnosis']
                paths[dir_name] = path
            final_dict[mod].append(paths)
    return final_dict


def main(args):
    # Load dirs for different modalities
    baseDirs = args.baseDirs

    # Take care of ouput path + filename
    outPath = Path(getattr(args, 'save_path', '/mnt/sda1/PET/json_datasets'))
    outPath.mkdir(parents=True, exist_ok=True)
    jsonName = args.datasetName
    outPath = outPath.joinpath(jsonName)

    # Load list of patients
    datalist = open(args.datalist, 'r').read().splitlines()

    # Divide data into different diagnosis & train, val ,test
    allData = divide_data(datalist=datalist, args=args)

    if args.cross_validation_arguments['act']:
        for i, fold in enumerate(allData):
            jsonDict = create_dict(fold, baseDirs, args)

            outPath_folds = Path(str(outPath).replace(f'{jsonName}', f'{jsonName}_fold{i+1}')).with_suffix('.json')

            jsonDict = create_dict(fold, baseDirs, args)

            json.dump(jsonDict,
                      outPath_folds.open('w'),
                      indent=4)
            print(f'output file written to {outPath_folds}')
    else:
        outPath = outPath.with_suffix('.json')

        # Create JSON dictionary
        jsonDict = create_dict(allData, baseDirs, args)

        json.dump(jsonDict,
                  outPath.open('w'),
                  indent=4)
        print(f'output file written to {outPath}')


if __name__ == '__main__':
    conf = 'AllData3D.yaml'
    args = yaml.load(Path(f'/mnt/sda1/PET/json_configurations/dataset_configs',
                          conf).open('r'),
                     Loader=yaml.CLoader)
    args = munch.Munch(args)
    main(args)