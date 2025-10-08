# This file is the main file for testing a model #

import argparse
import os
import torch

from pet_ct.main_utils.dataloader import Dataloader
from pet_ct.main_utils.model import Model
from pet_ct.main_utils.load_args import load_args
from pet_ct.main_utils.load_transforms import load_transforms
from pet_ct.main_utils.load_weights import load_weights
from tester import Tester

def parse_args():
    """ Parses command-line arguments """
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", type=str, required=False, help="Path to config file")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    cli_args = parse_args()

    # Import default configuration arguments
    project_dir = os.path.dirname(os.path.dirname(__file__))
    conf = 'Configs/experiment_configurations/test/AttentionUnet_48mips_original/AttUnet_48MIPs_TrainAS_TestOriginal_fold5.yaml'
    args = load_args(conf, cli_args, project_dir)

    # Initialize device for training
    device = torch.device(f"cuda:{args.device}") if isinstance(args.device, int) else torch.device("cpu")

    # Load all transforms to one object
    transforms = load_transforms(args)
    transforms.init_transforms()

    # Creating dataloader
    dataloader = Dataloader(args=args, transforms=transforms)
    dataloader.create_dataloader()

    # Creating instance of the model
    model = Model(args).create_model().to(device)

    # Load weights to model
    load_weights(args, model)

    # Testing
    tester = Tester(args=args,
                    model=model,
                    dataloader=dataloader.test_dataloader,
                    transforms=transforms)
    tester.test_loop()

if __name__ == "__main__":
    main()
