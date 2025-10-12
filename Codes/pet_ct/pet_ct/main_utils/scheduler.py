"""
This file contains a scheduler class that holds a scheduler from pytorch.
It checks which arguments the pytorch scheduler accepts and checks mutual arguments from configuration file.
"""

from inspect import getfullargspec, isclass
import torch

class Scheduler:
    def __init__(self, args, optimizer):
        self.args = args
        self.optimizer = optimizer
        self.scheduler_args = getattr(self.args, 'scheduler_arguments', None)
        self.scheduler = self.scheduler_args.get('scheduler', None)
        
    def create_scheduler(self):
        if (self.scheduler_args is None) or (self.scheduler is None):
            return None
        self._create_scheduler_function()
        self._get_input_args_from_scheduler()
        self._get_mutual_args()
        return self.schedulerFunc(**self.schedulerArgs)

    def _create_scheduler_function(self):
        torch_schedulers = [attr for attr in dir(torch.optim.lr_scheduler) if isclass(getattr(torch.optim.lr_scheduler, attr))]
        if self.scheduler not in torch_schedulers:
            raise ValueError(f'Scheduler {self.scheduler} is not supported (not in torch.optim.lr_scheduler).')
        schedulerFunc = getattr(torch.optim.lr_scheduler, self.scheduler)
        setattr(self, 'schedulerFunc', schedulerFunc)

    def _get_input_args_from_scheduler(self):
        inputArgs = getfullargspec(self.schedulerFunc).args
        setattr(self, 'inputArgs', inputArgs)

    def _get_mutual_args(self):
        schedulerArgs = {k: self.scheduler_args[k] for k in self.scheduler_args if k in self.inputArgs}
        schedulerArgs['optimizer'] = self.optimizer
        setattr(self, 'schedulerArgs', schedulerArgs)

