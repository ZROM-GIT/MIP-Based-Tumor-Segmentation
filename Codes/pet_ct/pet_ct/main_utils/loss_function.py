"""
This file contains a loss_function class that holds a loss function from MONAI.
It checks which arguments the MONAI loss function accepts and checks mutual arguments from configuration file.
"""

import monai
import torch
from inspect import getfullargspec, isclass
from pet_ct.main_utils import my_loss_functions


class LossFunction:
    def __init__(self, args):
        self.args = args
        self.lf_args = self.args['loss_function_arguments']
        self.loss_function_name = self.lf_args['loss_function_name']

    def create_loss_function(self):
        self._loss_function()
        self._get_input_args_from_lossFunc()
        self._get_mutual_args()
        return self.lossFunc(**self.lossArgs)

    def _loss_function(self):
        if hasattr(monai.losses, self.loss_function_name):
            lossFunc = getattr(monai.losses, self.loss_function_name)
        elif hasattr(torch.nn, self.loss_function_name):
            lossFunc = getattr(torch.nn, self.loss_function_name)
        elif hasattr(my_loss_functions, self.loss_function_name):
            return
        else:
            raise ValueError('Name of Loss function not available in MONAI nor Torch!')
        setattr(self, 'lossFunc', lossFunc)

    def _get_input_args_from_lossFunc(self):
        inputArgs = getfullargspec(self.lossFunc).args
        setattr(self, 'inputArgs', inputArgs)

    def _get_mutual_args(self):
        lossArgs = {k: self.lf_args[k] for k in self.lf_args if k in self.inputArgs}
        setattr(self, 'lossArgs', lossArgs)
