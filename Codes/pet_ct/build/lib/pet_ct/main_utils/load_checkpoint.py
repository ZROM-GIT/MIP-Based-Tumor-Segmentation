"""
This file contains a function that loads a checkpoint to an existing model.
"""

import torch

from pathlib import Path

def load_checkpoint(model, optimizer, scheduler, name, args):
    if getattr(args, 'checkpoint_to_resume_from', None) is not None:
        path = Path(args.project_dir) / Path(args.checkpoint_to_resume_from)
    else:
        path = Path(args.project_dir) / Path(args.save_checkpoints_path) / Path(name + '.pt')
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint.pop('model_state_dict'))
    optimizer.load_state_dict(checkpoint.pop('optimizer_state_dict'))
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint.pop('scheduler_state_dict'))
    args.update(checkpoint)
