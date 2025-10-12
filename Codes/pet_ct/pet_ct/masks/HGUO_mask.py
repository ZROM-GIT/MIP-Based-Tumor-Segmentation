"""
This file creates an instance of a HGUO mask model with weights and returns to the trainer/tester
It allows to call the class to activate the mask.
"""
import warnings

import cc3d
import numpy as np
import torch
from pet_ct.main_utils.load_args import load_args
from pet_ct.main_utils.model import Model


class HGUO_mask:
    def __init__(self, args):
        self.args = args
        self.get_args()
        self.load_model()

    def get_args(self):
        if 'num_of_mips' not in self.args.keys():
            raise warnings.warn('num_of_mips not specified in config file, choose between (16, 32), 16 taken as default')
            self.num_of_mips = 16
        else:
            self.num_of_mips = self.args['num_of_mips']

        if 'k' not in self.args.keys():
            raise warnings.warn('k (number of regions segmented) not specified in config file, 2 taken as default')
            self.k = 2
        else:
            self.k = self.args['k']

        self.args_path = self.args['HGUO_args_path']
        self.HGUO_args = load_args(self.args_path)

    def load_model(self):

        # Get CUDA device if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Import relevant model
        self.model = (Model(args=self.HGUO_args).create_model().to(self.device))

        # Load trained weights
        self.model.load_state_dict(torch.load(getattr(self.HGUO_args, f'weights_path{self.num_of_mips}')))

    def __call__(self, input, output):
        # Run inference on model
        mask_logits = self.model(input)
        mask_prob = torch.nn.functional.softmax(mask_logits, dim=1)
        mask = torch.argmax(input=mask_prob, dim=1)

        # Take torch tensor out of CPU and turn to numpy array
        mask_np = np.array(mask.cpu()).squeeze()

        # Leave brain and bladder only
        labels_out = cc3d.largest_k(
            mask_np, k=self.k)
        labels_out[labels_out > 0] = 1

        # Return to torch and cuda
        labels_out = torch.tensor(labels_out.astype('int8')).to(self.device)

        # Reverse mask labels (0/1)
        labels_out = ~labels_out + 2

        # Return mask to one-hot vector in the same size as output
        labels_out = torch.nn.functional.one_hot(torch.unsqueeze(input=labels_out, dim=0).long())
        labels_out = torch.movedim(input=labels_out, source=-1, destination=1)

        # Final mask element-wise multiplication
        output = output * labels_out

        return output
