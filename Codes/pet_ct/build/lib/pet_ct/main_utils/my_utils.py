"""
This file contains utility functions that might be used.
"""

from decimal import Decimal
from fractions import Fraction
from monai.inferers import sliding_window_inference
from monai.transforms import BatchInverseTransform, allow_missing_keys_mode
import copy
import inspect
import monai.data
import torch
import numpy as np

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def has_letters(inputString):
    return any(char.isalpha() for char in inputString)

def fix_args_if_needed(args):
    # Fixing fractions and list of floats
    if isinstance(args, list):
        for i, l in enumerate(args):
            if not isinstance(l, str):
                continue
            elif 'lambda ' in l:
                args[i] = exec(l)
                continue
            elif has_numbers(l) & has_letters(l):
                continue
            elif has_numbers(l):
                if '/' in l:
                    if '.' in l:
                        a, b = l.split('/')
                        a, b = Decimal(a), Decimal(b)
                        args[i] = float(a/b)
                    else:
                        args[i] = float(sum(Fraction(s) for s in l.split()))
                else:
                    args[i] = float(l)

    if not isinstance(args, dict):
        return args
    for a in args:
        if isinstance(args[a], str) \
                and (args[a] == 'None'):
            args[a] = None
        if isinstance(args[a], str) \
                and ('e-' in args[a]):
            args[a] = float(args[a])
        if isinstance(args[a], str) \
                and ('e+' in args[a]):
            args[a] = int(args[a])
        if isinstance(args[a], str) and 'lambda ' in args[a]:
            args[a] = eval(args[a])
        if isinstance(args[a], str) and 'eval' in args[a]:
            args[a] = eval(args[a])
        args[a] = fix_args_if_needed(args[a])
    return args


def Sliding_window_inference(input, model, args):
    inputArgs = inspect.getfullargspec(sliding_window_inference).args
    myArgs = getattr(args, 'sliding_window_inference_params')
    overlappingArgs = {k: myArgs[k] for k in myArgs if k in inputArgs}
    overlappingArgs['predictor'] = model
    overlappingArgs['inputs'] = input
    output = sliding_window_inference(**overlappingArgs)
    return output


def InverseTransforms(input: monai.data.MetaTensor,
                      pred: monai.data.MetaTensor,
                      target: monai.data.MetaTensor,
                      transforms,
                      dataloader,
                      timing: str,
                      set: str,
                      args: dict,
                      device: str):
    '''
    target_name = args.input_name  #args.target_name
    target = input
    # TODO: Get rid of the last two changes above!

    inverse_transforms_args = getattr(args, f'{set}_inverse_transforms', None)
    # If no inverse transforms were noted in the configuration file
    if inverse_transforms_args is None:
        raise ValueError('No inverse transforms noted in config')

    # If this is not the time to apply the transforms then do nothing
    if (not inverse_transforms_args['activation']) or (inverse_transforms_args['timing'] != timing):
        return pred

    # Calculate batch size
    # TODO: Change this number
    batch_size = 3  #pred.shape[0]

    # Get relevant transforms and dataloader
    relevant_transforms = getattr(transforms, args['set2transforms'][set])

    # if batch_size > 1:
    # Get applied operations for our predictions (and filter what we want)
    num_of_reverses = inverse_transforms_args['num_of_reverses']
    pred.applied_operations = target.applied_operations if num_of_reverses == -1 else [target.applied_operations[j][-num_of_reverses:] for j in range(len(target.applied_operations))]
    segs_dict = {target_name: pred}
    # dataloader2 = copy.deepcopy(dataloader)
    batch_inverter = BatchInverseTransform(relevant_transforms, dataloader)
    batch_inverter.transform.transforms = batch_inverter.transform.transforms#[1:] # Deleting first 2
    with allow_missing_keys_mode(relevant_transforms):
        inverted_pred = batch_inverter(segs_dict)
    inverted_pred = inverted_pred[0][target_name]
    inverted_pred = torch.unsqueeze(inverted_pred, dim=0)

    # else:
        # # Get applied operations for our predictions (and filter what we want)
        # num_of_reverses = inverse_transforms_args['num_of_reverses']
        # pred.applied_operations = target.applied_operations if num_of_reverses == -1 else [target.applied_operations[0][-num_of_reverses:]]
        #
        # seg_dict = {target_name: pred[0]}
        # with allow_missing_keys_mode(relevant_transforms):
        #     inverted_pred = relevant_transforms.inverse(seg_dict)[target_name]
        #
        # # Return data to MetaTensor type on cuda device
        # inverted_pred = torch.Tensor(inverted_pred).unsqueeze(dim=0).to(device)
    '''

    inverse_transforms_args = getattr(args, f'{set}_inverse_transforms', None)
    if (not inverse_transforms_args['activation']) or (inverse_transforms_args['timing'] != timing):
        return pred

    target_name = args.input_name



    relevant_transforms = getattr(transforms, args['set2transforms'][set])
    segs_dict = {target_name: pred[0]}
    # batch_inverter = BatchInverseTransform(relevant_transforms, dataloader)
    with allow_missing_keys_mode(relevant_transforms):
        inverted_pred = relevant_transforms.inverse(segs_dict)

        # inverted_pred = batch_inverter(segs_dict)
    inverted_pred = torch.unsqueeze(torch.Tensor(inverted_pred[target_name]), dim=0)

    return inverted_pred
