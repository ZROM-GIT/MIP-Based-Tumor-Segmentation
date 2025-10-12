"""
This file contains a gradient clipper class that holds a clip_grad_norm function from pytorch.
It checks which arguments the function accepts and checks mutual arguments from configuration file.
"""

from inspect import getfullargspec, isclass
import torch
from torch.nn.utils import clip_grad_norm_


class GradientClipper:
    def __init__(self, args, optimizer):
        gradient_clipping_args = getattr(args, 'gradient_clipping_args', None)
        grad_clip_function = clip_grad_norm_
        inputArgs = getfullargspec(grad_clip_function).args
        GradClipArgs = {k: gradient_clipping_args[k] for k in gradient_clipping_args if k in inputArgs}
        return grad_clip_function(**GradClipArgs)