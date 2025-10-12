

import torch

from pathlib2 import Path

def load_weights(args, model):
    # Check if we are loading from checkpoint or from weights .pt file.
    weights_source = getattr(args, 'weights_source')
    checkpoint_path = Path(args.project_dir) / Path(getattr(args, 'checkpoint'))
    weights_path = Path(args.project_dir) / getattr(args, 'weights')

    if weights_source == 'W':
        model.load_state_dict(torch.load(weights_path, weights_only=False))
    elif weights_source == 'C':
        model.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model_state_dict'])
