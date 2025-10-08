"""
This file is the main file for training a model.
"""

import argparse
import os
import torch
import wandb

from pet_ct.main_utils.dataloader import Dataloader
from pet_ct.main_utils.determinism import determinism
from pet_ct.main_utils.get_project_dir import find_pycharm_project_root
from pet_ct.main_utils.load_trainer_and_tester import get_trainer_and_tester
from pet_ct.main_utils.load_args import load_args
from pet_ct.main_utils.load_checkpoint import load_checkpoint
from pet_ct.main_utils.loss_function import LossFunction
from pet_ct.main_utils.load_transforms import load_transforms
from pet_ct.main_utils.model import Model
from pet_ct.main_utils.optimizer import Optimizer
from pet_ct.main_utils.scheduler import Scheduler

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
    Trainer, Tester = get_trainer_and_tester(args)

    # Start cometML experiment
    if args.resume_training:
        run = wandb.init(project=args.project_name,
                         id=args.run_id,
                         resume='must')
    elif args.log_experiment:
        wandb.login()
        run = wandb.init(
            project=args.project_name,
            name=args.full_experiment_name,
            config=args,
        )
    else:
        run = None

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

    # Assigning optimizer and its inputs
    optimizer = Optimizer(args, model=model).create_optimizer()

    # Creating a scheduler
    scheduler = Scheduler(args, optimizer).create_scheduler() if args.use_scheduler else None

    # Assigning loss function and its inputs
    lossFunc = LossFunction(args).create_loss_function()

    # Resume training if specified
    if args.resume_training:
        load_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, name=args.full_experiment_name, args=args) # Load Training Checkpoint

    if args.train:
        # Training the model
        trainer = Trainer(args=args,
                        model=model,
                        training_dataloader=dataloader.training_dataloader,
                        validation_dataloader=dataloader.validation_dataloader,
                        test_dataloader=dataloader.test_dataloader,
                        device=device,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=lossFunc,
                        logger=run,
                        transforms=transforms)
        trainer.train_loop()

    if args.test:
        tester = Tester(args=args,
                        model=model,
                        test_dataloader=dataloader.test_dataloader,
                        loss_fn=lossFunc,
                        device=device,
                        logger=run,
                        transforms=transforms)
        tester.test_loop()

    if args.log_experiment:
        run.finish()

if __name__ == "__main__":
    config_list = [
        'Configs/experiment_configurations/train/classifier/volumetric_binary_classifier_fold1.yaml',
    ]

    for conf_path in config_list:
        print(f"Running experiment with config: {conf_path}")
        main(conf_path)



