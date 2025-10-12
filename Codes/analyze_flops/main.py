"""
This file is the main file for training a model.
"""

/import argparse
import os
import torch
import wandb

from pet_ct.main_utils.dataloader import Dataloader
from pet_ct.main_utils.determinism import determinism
from pet_ct.main_utils.get_project_dir import find_pycharm_project_root
from pet_ct.main_utils.load_args import load_args
from pet_ct.main_utils.load_checkpoint import load_checkpoint
from pet_ct.main_utils.loss_function import LossFunction
from pet_ct.main_utils.load_transforms import load_transforms
from pet_ct.main_utils.model import Model
from pet_ct.main_utils.optimizer import Optimizer
from pet_ct.main_utils.scheduler import Scheduler
from test_flops import Tester

def parse_args():
    """ Parses command-line arguments """
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", type=str, required=False, help="Path to config file")
    return parser.parse_args()

def main(conf):
    # Parse command-line arguments
    cli_args = parse_args()

    # Import default configuration arguments
    project_dir = find_pycharm_project_root()
    args = load_args(conf, cli_args, project_dir)

    # Initialize device for training
    device = torch.device(f"cuda:{args.device}") if isinstance(args.device, int) else torch.device("cpu")

    # Set determinism
    determinism(args)

    # Load all transforms to one object
    transforms = load_transforms(args)
    transforms.init_transforms()

    # Creating dataloader
    dataloader = Dataloader(args=args, transforms=transforms)
    dataloader.create_dataloader()

    # Creating instance of the model
    model = Model(args).create_model().to(device)

    tester = Tester(args=args,
                    model=model,
                    test_dataloader=dataloader.test_dataloader,
                    device=device,
                    transforms=transforms)
    tester.test_loop()

if __name__ == "__main__":
    config_list = [
        'Codes/analyze_flops/AttUnet_16MIPs_fold1.yaml',
        'Codes/analyze_flops/AttUnet_32MIPs_fold1.yaml',
        'Codes/analyze_flops/AttUnet_48MIPs_fold1.yaml',
        'Codes/analyze_flops/AttUnet_64MIPs_fold1.yaml',
        'Codes/analyze_flops/AttUnet_80MIPs_fold1.yaml',
        # 'Codes/analyze_flops/PET_non_healthy_swin_pet_only_fold1.yaml',
    ]

    for conf_path in config_list:
        print(f"Running experiment with config: {conf_path}")
        main(conf_path)



