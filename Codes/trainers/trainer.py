import gc
import os
import matplotlib.pyplot as plt
import monai.metrics
import nibabel as nib
import random
import warnings
from pathlib2 import Path
from tqdm import tqdm
from pet_ct.main_utils.load_metrics import LoadMetrics
from pet_ct.main_utils.my_utils import *
from pet_ct.masks.masks_loader import masks
from pet_ct.secondary_utils.log_temps import log_temps
from pet_ct.secondary_utils.AffineMatrices import affine_matrices
from pet_ct.main_utils.gradient_clipper import GradientClipper

class Trainer:
    def __init__(self,
                 args: dict,
                 model,
                 training_dataloader: monai.data.DataLoader,
                 validation_dataloader: monai.data.DataLoader,
                 optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler,
                 loss_fn: monai.losses,
                 device='cuda',
                 logger=None,
                 transforms=None):

        self.args = args
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        self.transforms = transforms

    def initialization(self):
        self.epoch = getattr(self.args, 'epoch', 0)
        self.batch_size = getattr(self.args, 'batch_size', 1)
        self.weights_path = getattr(self.args, 'save_weights_path')
        self.checkpoints_path = getattr(self.args, 'save_checkpoints_path')
        self.use_amp = getattr(self.args, 'use_amp', False)
        self.use_sliding_window_inference = getattr(self.args, 'use_sliding_window_inference', [])
        self.amp_dtype = getattr(torch, getattr(self.args, 'amp_dtype', 'float32'))
        self.final_activation = getattr(self.args, 'final_activation', None)  # Optional final activation for model
        self.masks = getattr(self.args, 'masks', False)
        self.num_accumulation_steps = getattr(self.args, 'accum_iter', self.batch_size)
        self.clip_gradients = getattr(self.args, 'use_gradient_clipping', False)
        self.save_val_every_n = getattr(self.args, 'save_val_every_n', False)
        self.save_checkpoint_every_n = getattr(self.args, 'save_checkpoint_every_n', np.inf)
        self.save_checkpoint_when_val_improves = getattr(self.args, 'save_checkpoint_when_val_improves', False)
        self.apply_inverse_transforms = getattr(self.args, 'apply_inverse_transforms', False)


        if self.final_activation:
            self.final_activation = getattr(torch.nn.functional, self.final_activation)
        if self.clip_gradients:
            self.gradient_clipper = GradientClipper(self.args)
        if self.masks:
            self.masks = masks(self.args)

    def pred_binary(self, pred, threshold=0.5):
        pred = torch.threshold(pred, threshold=threshold, value=0)
        pred[pred >= threshold] = 1
        return pred

    def visualize_data(self, input, target):
        np_fin = monai.visualize.utils.blend_images(input[:, 0, 0, :, :], target[:, 0, 0, :, :], alpha=0.1, cmap='hot')
        monai.visualize.matshow3d(np_fin, every_n=5, cmap='hot', clim=[0, 100])
        plt.show()
        return

    def model_improves(self):
        # Metric computed to understand if model improved
        mean_value = np.mean(np.concatenate(self.metrics_values['validation'][self.args.metricForSaving]).ravel())
        if mean_value > self.metric_value_max:
            self.metric_value_max = mean_value
            return True
        else:
            return False

    def create_new_metrics_dictionary(self):
        if not hasattr(self, 'metrics_values'):
            self.metrics_values = {'training': {k: [] for k in self.metrics_obj.metrics},
                                   'validation': {k: [] for k in self.metrics_obj.metrics},
                                   'test': {k: [] for k in self.metrics_obj.metrics}}

    def load_metrics(self):
        self.metrics_obj = LoadMetrics(self.args)
        self.metrics_obj.create_metrics()

        # Create variable to keep score of optimized MetricForSaving
        self.metric_value_max = getattr(self.args, 'metric_value_max', 0)

    def compute_metrics(self, y_pred, y, key='training'):
        y, y_pred = y.detach().cpu(), y_pred.detach().cpu()
        for metricName, metricFunc in self.metrics_obj.metrics.items():
            metric_result = metricFunc(y_pred=y_pred.to(torch.float64), y=y.to(torch.float64),)
            if metric_result.isnan().any():
                print('NANANANANASSSSSSS')
            self.metrics_values[key][metricName].append(np.array(metric_result))

    def log_metrics(self, keys=['training', 'validation']):
        for key in keys:
            for metric in self.metrics_values[key]:
                mean_value = np.mean(np.concatenate(self.metrics_values[key][metric]).ravel())
                self.log(name=metric, value=mean_value, key=key, step=self.epoch)

    def log(self, name, value, key='training', step=0):
        if self.logger is not None:
            if step is not None:
                self.logger.log({f'{key}/{name}': value}, step=step)
            else:
                self.logger.log({f'{key}/{name}': value})

    def load_best_weights(self):
        path = Path(f'{self.args.project_dir}/{self.weights_path}/{self.args.subproject_name}/{self.args.full_experiment_name}/best_weights.pt')
        if not path.exists():
            warnings.warn(f'Best weights not found at {path}. Please check the path or run training first. Running on random weights.')
        else:
            self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_weights(self):
        path = Path(f'{self.args.project_dir}/{self.weights_path}/{self.args.subproject_name}/{self.args.full_experiment_name}')
        path.mkdir(parents=True, exist_ok=True)
        torch.save(obj=self.model.state_dict(),
                   f=path / f'best_weights.pt'
                   )

    def save_checkpoint(self, best=False, last=False):
        path = Path(f'{self.args.project_dir}/{self.checkpoints_path}/{self.args.subproject_name}/{self.args.full_experiment_name}')
        path.mkdir(parents=True, exist_ok=True)
        obj = {
            'config': self.args,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.validation_loss,
            'metric_name': self.args.metricForSaving,
            'metric_value_max': self.metric_value_max,
        }
        obj = obj.update({'scheduler_state_dict': self.scheduler.state_dict()}) if self.scheduler is not None else obj
        if last:
            f = f'{path}/last_checkpoint.pt'
            torch.save(obj=obj, f=f)
        else:
            f = f'{path}/best_checkpoint.pt' if best else f'{path}/epoch{self.epoch}.pt'
            torch.save(obj=obj, f=f)

    def save_prediction_as_nifti(self, input, target, seg_prediction, mode='nib'):
        path = Path(self.args.project_dir) / Path(self.args.save_val_predictions_path)
        saving_path = path / f'PET{self.args.experiment_number}_{self.args.experiment_name}'

        patient_dict = self.validation_dataloader.dataset.data[self.batch_num - 1]
        patient_path = Path(patient_dict['CaseID']) if 'CaseID' in patient_dict else Path(patient_dict[random.choice(list(patient_dict))])
        if 'PACS' in patient_path.parts:
            patient_name = patient_path.parts[-3]
            saving_path = saving_path / patient_name
        else:
            saving_path = saving_path / Path('epoch' + str(self.epoch)) / Path(patient_path)

        # Open new folder for experiment
        if not os.path.exists(saving_path):
            # Create saving function
            os.makedirs(saving_path)

        # Transform data
        input = input.squeeze()
        if hasattr(input, 'affine'):
            input.affine = input.affine.squeeze()
        else:
            input.affine = affine_matrices['SLA']
        target = target.argmax(axis=1).squeeze()
        seg_prediction = seg_prediction.argmax(axis=1).squeeze()

        input, target, seg_prediction = input.detach().cpu(), target.detach().cpu(), seg_prediction.detach().cpu()

        # seg_and_pred = np.zeros(target.shape)
        # for label in range(0, target.max() + 1):
        seg_and_pred = target.clone()
        seg_and_pred[seg_prediction == 1] = 3
        seg_and_pred[seg_prediction == 2] = 4
        seg_and_pred[(target == 1) & (seg_prediction == 1)] = 5
        seg_and_pred[(target == 2) & (seg_prediction == 2)] = 6

        if len(input.shape) > 3:
            input = torch.movedim(input, source=0, destination=-1)

        match mode:
            case 'monai':
                if self.batch_num == 1:
                    self.save_SUV = monai.transforms.SaveImage(output_dir=str(saving_path), output_postfix='SUV',
                                                               output_ext='.nii.gz',
                                                               separate_folder=False, channel_dim=None)
                    self.save_SEG_prediction = monai.transforms.SaveImage(output_dir=str(saving_path), output_postfix='PRED',
                                                                          output_ext='.nii.gz',
                                                                          separate_folder=False, channel_dim=None)
                    self.save_ALL_prediction = monai.transforms.SaveImage(output_dir=str(saving_path), output_postfix='ALL',
                                                                          output_ext='.nii.gz',
                                                                          separate_folder=False, channel_dim=None)

                    self.save_SEG = monai.transforms.SaveImage(output_dir=str(saving_path), output_postfix='SEG',
                                                               output_ext='.nii.gz',
                                                               separate_folder=False, channel_dim=None)
                # Save data
                self.save_SUV(input)
                self.save_SEG_prediction(seg_prediction)
                self.save_SEG(target)
                self.save_ALL_prediction(seg_and_pred)

            case 'nib':
                # affine = affine_matrices['SLA']
                affine = input.affine

                input_nif = nib.Nifti1Image(np.array(input, dtype=np.float32), affine=affine)
                target_nif = nib.Nifti1Image(target.numpy(), affine=affine, dtype=np.int16)
                seg_pred_nif = nib.Nifti1Image(seg_prediction.numpy(), affine=affine, dtype=np.int16)
                all_nif = nib.Nifti1Image(seg_and_pred.numpy(), affine=affine, dtype=np.int16)

                nib.save(input_nif, filename=f'{saving_path}/input.nii.gz')
                nib.save(target_nif, filename=f'{saving_path}/target.nii.gz')
                nib.save(seg_pred_nif, filename=f'{saving_path}/prediction.nii.gz')
                nib.save(all_nif, filename=f'{saving_path}/all.nii.gz')

    def eval_step(self, batch):
        # Load input and target
        input = batch[f'{self.args.input_name}']
        target = batch[f'{self.args.target_name}']

        # Sending data to GPU if possible
        input, target, self.model = input.to(self.device), target.to(self.device), self.model.to(self.device)

        with torch.amp.autocast(device_type=str(self.device), dtype=self.amp_dtype, enabled=self.use_amp):
            with torch.no_grad():
                if 'validation' in self.use_sliding_window_inference:
                    pred = Sliding_window_inference(input=input, model=self.model, args=self.args)
                else:
                    pred = self.model(input)

                # Make sure there are no NaNs
                if torch.isnan(pred).any():
                    print('There are NaNs')

                if self.final_activation:
                    pred = self.final_activation(pred, dim=1)
                pred_target = self.pred_binary(pred, threshold=0.5)

                if self.masks:
                    pred_target = self.masks(input=input, output=pred_target)

                # Compute the loss and its gradients
                loss = self.loss_fn(pred, target).mean()

        # Inverse transforms (Pre-metrics)
        pred_target = InverseTransforms(input=input, pred=pred_target, target=target, transforms=self.transforms,
                                        dataloader=self.validation_dataloader, timing='pre', set='validation',
                                        args=self.args, device=self.device)

        # Compute metrics
        for i, x in enumerate(target):
            if target[i, 1].any():  # 2nd dim: First channel = Background, Second channel = Foreground
                self.compute_metrics(y=torch.unsqueeze(target[i], 0), y_pred=pred_target[i].unsqueeze(0), key='validation')

        # Inverse transforms (Post-metrics)
        pred_target = InverseTransforms(input=input, pred=pred_target, target=target, transforms=self.transforms,
                                        dataloader=self.validation_dataloader, timing='post', set='validation',
                                        args=self.args, device=self.device)

        if self.save_val_every_n is not False:
            if self.batch_num % self.save_val_every_n == 0:
                self.save_prediction_as_nifti(input=input, target=target, seg_prediction=pred_target)

        del input, target, pred, pred_target, batch

        return loss.item()

    def train_step(self, batch):
        # Load input and target
        input = batch[f'{self.args.input_name}']
        target = batch[f'{self.args.target_name}']

        # Sending data to device
        input, target, self.model = input.to(self.device), target.to(self.device), self.model.to(self.device)

        with torch.amp.autocast(device_type=str(self.device), dtype=self.amp_dtype, enabled=self.use_amp):
            if 'training' in self.use_sliding_window_inference:
                pred = Sliding_window_inference(input=input, model=self.model, args=self.args)
            else:
                pred = self.model(input)

            # Make sure there are no NaNs
            if torch.isnan(pred).any():
                print('There are NaNs')

            if self.final_activation:
                pred = self.final_activation(pred, dim=1)

            pred_target = self.pred_binary(pred, threshold=0.5)

            if self.masks:
                pred_target = self.masks(input=input, output=pred_target)

            # Compute the loss and its gradients
            self.loss = self.loss_fn(pred, target).mean() / (self.num_accumulation_steps / self.batch_size)

        # Scaler backward
        if self.use_amp:
            self.scaler.scale(self.loss).backward()
        else:
            self.loss.backward()

        # Clip gradients if using
        self.gradient_clipper(self.model.parameters()) if self.clip_gradients else None

        if (self.batch_num % self.num_accumulation_steps == 0) or (self.batch_num == len(self.training_dataloader)):
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)  # Zero gradients

        # Inverse transforms (pre-metrics)
        pred_target = InverseTransforms(input=input, pred=pred_target, target=target, transforms=self.transforms,
                                        dataloader=self.training_dataloader, timing='pre', set='training',
                                        args=self.args, device=self.device)

        # Compute metrics
        for i, x in enumerate(target):
            if target[i, 1].any(): # 2nd dim: First channel = Background, Second channel = Foreground
                self.compute_metrics(y=torch.unsqueeze(target[i], 0), y_pred=pred_target[i].unsqueeze(0), key='training')

        # Inverse transforms (Post-metrics)
        pred_target = InverseTransforms(input=input, pred=pred_target, target=target, transforms=self.transforms,
                                        dataloader=self.training_dataloader, timing='post', set='training',
                                        args=self.args, device=self.device)

        # Cleaning up
        del input, target, pred, pred_target, batch
        torch.cuda.empty_cache()
        gc.collect()

        return self.loss.item()

    def train_loop(self):
        # Initialize arguments
        self.initialization()

        # Load metrics
        self.load_metrics()

        # Assign model to training mode
        self.model.train()

        # Gradscaler if device is GPU
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        for epoch in range(1, self.args.epochs + 1):
            # Keep epoch number
            self.epoch += 1

            # Restart metrics values
            self.create_new_metrics_dictionary()

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            self.model.train()
            train_loader = tqdm(self.training_dataloader, colour='magenta')
            running_loss_train = []

            for i, batch in enumerate(train_loader):
                # Save batch number
                self.batch_num = i + 1

                # Log temperatures and progression
                log_temps(epoch=self.epoch, batch=self.batch_num, set_name='training', args=self.args)

                # train loop & compute metrics on training
                loss = self.train_step(batch)

                # loss tracking
                train_loader.set_description(f'Training   | Epoch {self.epoch}: batch loss = {loss}')
                running_loss_train.append(loss)

            # Training loss
            loss_mean = np.mean(running_loss_train)
            self.training_loss = loss_mean
            self.log('loss', self.training_loss, key='training', step=self.epoch)

            # Change model to eval mode
            self.model.eval()
            val_loader = tqdm(self.validation_dataloader, colour='white')
            running_loss_val = []

            for i, batch in enumerate(val_loader):
                # Save batch number
                self.batch_num = i + 1

                # Validation step
                loss_val = self.eval_step(batch)
                running_loss_val.append(loss_val)

                # Log temperatures
                log_temps(epoch=self.epoch, batch=self.batch_num, set_name='validation', args=self.args)

                val_loader.set_description(f'Validation | Epoch {self.epoch}: batch loss = {loss_val}')

            # Checking on Validation
            loss_val = np.mean(running_loss_val)
            self.validation_loss = loss_val
            self.log('loss', self.validation_loss, key='validation', step=self.epoch)

            # Update and log scheduler
            if self.scheduler is not None:
                self.log(name='Scheduler', value=self.scheduler.get_last_lr()[0], key='LR', step=self.epoch)
                self.scheduler.step()
            else:
                self.log(name='Constant', value=self.optimizer.defaults['lr'], key='LR', step=self.epoch)

            # Log metrics
            self.log_metrics(keys=['training', 'validation'])

            # Save checkpoint for last epoch
            self.save_checkpoint(best=False, last=True)

            # Save state_dict if model improves
            if self.model_improves():
                self.save_weights()
                self.save_checkpoint(best=True, last=False)
            elif self.epoch % self.save_checkpoint_every_n == 0:
                self.save_checkpoint(best=False, last=False)

            torch.cuda.empty_cache()
            gc.collect()
    
    # def test_loop(self):
    #     # Re-Initialize arguments
    #     self.initialization()
    #
    #     # Load best weights
    #     self.load_best_weights()
    #
    #     # Load metrics
    #     self.load_metrics()
    #
    #     # Restart metrics values if test only
    #     self.create_new_metrics_dictionary()
    #
    #     # Assign model to evaluation mode
    #     self.model.eval()
    #
    #     test_loader = tqdm(self.test_dataloader, colour='blue')
    #     running_loss_test = []
    #
    #     for i, batch in enumerate(test_loader):
    #         self.batch_num = i + 1
    #
    #         # Load input and target
    #         input = batch[f'{self.args.input_name}']
    #         target = batch[f'{self.args.target_name}']
    #
    #         # Sending data to GPU if possible
    #         input, target, self.model = input.to(self.device), target.to(self.device), self.model.to(self.device)
    #
    #         with torch.amp.autocast(device_type=str(self.device), dtype=self.amp_dtype, enabled=self.use_amp):
    #             with torch.no_grad():
    #                 if 'test' in self.use_sliding_window_inference:
    #                     pred = Sliding_window_inference(input=input, model=self.model, args=self.args)
    #                 else:
    #                     pred = self.model(input)
    #
    #                 # Make sure there are no NaNs
    #                 if torch.isnan(pred).any():
    #                     print('There are NaNs')
    #
    #                 if self.final_activation:
    #                     pred = self.final_activation(pred, dim=1)
    #                 pred_target = self.pred_binary(pred, threshold=0.5)
    #
    #                 if self.masks:
    #                     pred_target = self.masks(input=input, output=pred_target)
    #
    #                 # Compute the loss and its gradients
    #                 loss = self.loss_fn(pred, target).mean()
    #
    #         # Inverse transforms (Pre-metrics)
    #         pred_target = InverseTransforms(input=input, pred=pred_target, target=target, transforms=self.transforms,
    #                                         dataloader=self.validation_dataloader, timing='pre', set='validation',
    #                                         args=self.args, device=self.device)
    #
    #         # Compute metrics
    #         for i, x in enumerate(target):
    #             if target[i, 1].any():  # 2nd dim: First channel = Background, Second channel = Foreground
    #                 self.compute_metrics(y=torch.unsqueeze(target[i], 0), y_pred=pred_target[i].unsqueeze(0),
    #                                      key='test')
    #
    #         # Inverse transforms (Post-metrics)
    #         pred_target = InverseTransforms(input=input, pred=pred_target, target=target, transforms=self.transforms,
    #                                         dataloader=self.validation_dataloader, timing='post', set='validation',
    #                                         args=self.args, device=self.device)
    #
    #         # TODO: SHOULD SAVE TEST PREDICTIONS IN NIFTI FORMAT
    #         # if self.save_val_every_n is not False:
    #         #     if self.batch_num % self.save_val_every_n == 0:
    #         #         self.save_prediction_as_nifti(input=input, target=target, seg_prediction=pred_target)
    #
    #         del input, target, pred, pred_target, batch
    #
    #         loss_test = loss.item()
    #         # loss tracking
    #         running_loss_test.append(loss_test)
    #
    #         # Logs
    #         test_loader.set_description(f'Testing: batch loss = {loss_test}')
    #
    #     # Logging test loss
    #     loss_test = np.mean(running_loss_test)
    #     self.test_loss = loss_test
    #     self.log('loss', self.test_loss, key='test', step=None)
    #
    #     # Log metrics
    #     self.log_metrics(keys=['test'])
