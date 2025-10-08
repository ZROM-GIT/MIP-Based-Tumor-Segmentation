import gc
import os
import json
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
                 test_dataloader: monai.data.DataLoader,
                 optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler,
                 loss_fn: monai.losses,
                 device='cuda',
                 logger=None,
                 transforms=None):

        self.args = args
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
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
        self.amp_dtype = getattr(torch, getattr(self.args, 'amp_dtype', 'float32'))
        self.final_activation = getattr(self.args, 'final_activation', None)  # Optional final activation for model
        self.num_accumulation_steps = getattr(self.args, 'accum_iter', self.batch_size)
        self.clip_gradients = getattr(self.args, 'use_gradient_clipping', False)
        self.save_val_every_n = getattr(self.args, 'save_val_every_n', False)
        self.save_checkpoint_every_n = getattr(self.args, 'save_checkpoint_every_n', np.inf)
        self.save_checkpoint_when_val_improves = getattr(self.args, 'save_checkpoint_when_val_improves', False)

        if self.final_activation:
            self.final_activation = getattr(torch.nn.functional, self.final_activation)
        if self.clip_gradients:
            self.gradient_clipper = GradientClipper(self.args)

    def pred_binary(self, pred, threshold=0.5):
        pred = torch.threshold(pred, threshold=threshold, value=0)
        pred[pred >= threshold] = 1
        return pred

    def init_metrics(self):
        self.metrics = self.metrics_obj.metrics
        # self.metrics_names = [k for k,v in self.metrics_obj.metrics.items()]
        # self.metrics = [v for k,v in self.metrics_obj.metrics.items()]

    def reset_metrics(self):
        for metricName, metricFunc in self.metrics.items():
            metricFunc.reset()

    def load_metrics(self):
        self.metrics_obj = LoadMetrics(self.args)
        self.metrics_obj.create_metrics()

        # Create variable to keep score of optimized MetricForSaving
        self.metric_value_max = getattr(self.args, 'metric_value_max', 0)

    def compute_metrics(self, y_pred, y, key='training'):
        y, y_pred = y.detach().cpu(), y_pred.detach().cpu()
        for metricName, metricFunc in self.metrics_obj.metrics.items():
            metric_result = metricFunc(y_pred=y_pred.to(torch.float64), y=y.to(torch.float64), )
            if metric_result.isnan().any():
                print('NANANANANASSSSSSS')
            self.metrics_values[key][metricName].append(np.array(metric_result))

    def log(self, name, value, key='training', step=0, test_dict=None):
        if self.logger is not None:
            self.logger.log({f'{key}/{name}': value}, step=step)
        if test_dict is not None:
            test_dict[f'{key}/{name}'] = value

    def load_best_weights(self):
        path = Path(
            f'{self.args.project_dir}/{self.weights_path}/{self.args.subproject_name}/{self.args.full_experiment_name}/best_weights.pt')
        if not path.exists():
            warnings.warn(
                f'Best weights not found at {path}. Please check the path or run training first. Running on random weights.')
        else:
            self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_weights(self):
        path = Path(
            f'{self.args.project_dir}/{self.weights_path}/{self.args.subproject_name}/{self.args.full_experiment_name}')
        path.mkdir(parents=True, exist_ok=True)
        torch.save(obj=self.model.state_dict(),
                   f=path / f'best_weights.pt'
                   )

    def save_checkpoint(self, best=False, last=False):
        path = Path(
            f'{self.args.project_dir}/{self.checkpoints_path}/{self.args.subproject_name}/{self.args.full_experiment_name}')
        path.mkdir(parents=True, exist_ok=True)
        obj = {
            # 'config': self.args,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.validation_loss,
            'metric_name': 'macro F1',
            'metric_value_max': self.metric_value_max,
        }
        obj = obj.update({'scheduler_state_dict': self.scheduler.state_dict()}) if self.scheduler is not None else obj
        if last:
            f = f'{path}/last_checkpoint.pt'
            torch.save(obj=obj, f=f)
        else:
            f = f'{path}/best_checkpoint.pt' if best else f'{path}/epoch{self.epoch}.pt'
            torch.save(obj=obj, f=f)

    def eval_step(self, batch):
        # Load input and target
        input = batch[f'{self.args.input_name}']
        target = batch[f'{self.args.target_name}']

        # Sending data to GPU if possible
        input, target, self.model = input.to(self.device), target.to(self.device), self.model.to(self.device)

        with torch.amp.autocast(device_type=str(self.device), dtype=self.amp_dtype, enabled=self.use_amp):
            with torch.no_grad():
                pred = self.model(input)

                # Make sure there are no NaNs
                if torch.isnan(pred).any():
                    print('There are NaNs')

                if self.final_activation:
                    pred = self.final_activation(pred, dim=1)

                pred_target = self.model.predict_class(pred)

                # Compute the loss and its gradients
                loss = self.loss_fn(pred, target).mean()

        # Compute metrics
        for metricName, metricFunc in self.metrics.items():
            metricFunc = metricFunc.to(self.device)
            x = pred_target.clone().to(torch.int).detach().cpu()
            y = target.clone().argmax(dim=1).detach().cpu()
            metricFunc.update(x, y)

        # del input, target, pred, pred_target, batch

        return loss.item()

    def train_step(self, batch):
        # Load input and target
        input = batch[f'{self.args.input_name}']
        target = batch[f'{self.args.target_name}']

        # Sending data to device
        input, target, self.model = input.to(self.device), target.to(self.device), self.model.to(self.device)

        with torch.amp.autocast(device_type=str(self.device), dtype=self.amp_dtype, enabled=self.use_amp):
            pred = self.model(input)

            # Make sure there are no NaNs
            if torch.isnan(pred).any():
                print('There are NaNs')

            if self.final_activation:
                pred = self.final_activation(pred, dim=1)

            pred_target = self.model.predict_class(pred)

            # Compute the loss and its gradients
            self.loss_fn = self.loss_fn.to(self.device)
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

        # Compute metrics
        for metricName, metricFunc in self.metrics.items():
            metricFunc = metricFunc.to(self.device)
            x = pred_target.clone().to(torch.int).detach().cpu()
            y = target.clone().argmax(dim=1).detach().cpu()
            metricFunc.update(x, y)

        # # Add synchronization before cleanup
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        #
        # # Cleaning up
        # del input, target, pred, pred_target, batch
        # torch.cuda.empty_cache()
        # gc.collect()

        return self.loss.item()

    def train_loop(self):
        # Initialize arguments
        self.initialization()

        # Load metrics
        self.load_metrics()
        self.init_metrics()

        # Assign model to training mode
        self.model.train()

        # Gradscaler if device is GPU
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        for epoch in range(1, self.args.epochs + 1):
            # Keep epoch number
            self.epoch += 1

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

            # Log training metrics
            key = 'training'
            for metricName, metricFunc in self.metrics.items():
                if metricName == 'MulticlassConfusionMatrix':
                    conf_matrix = metricFunc.compute()
                    num_classes = conf_matrix.shape[0]

                    TP = torch.diag(conf_matrix)
                    FP = conf_matrix.sum(dim=0) - TP
                    FN = conf_matrix.sum(dim=1) - TP

                    precision = TP / (TP + FP + 1e-8)  # avoid divide-by-zero
                    recall = TP / (TP + FN + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)

                    macro_precision = precision.mean()
                    macro_recall = recall.mean()
                    macro_f1 = f1.mean()

                    for i in range(num_classes):
                        self.log(name=f'Precision_{i}', value=precision[i].item(), key=key, step=self.epoch)
                        self.log(name=f'Recall_{i}', value=recall[i].item(), key=key, step=self.epoch)
                        self.log(name=f'F1_{i}', value=f1[i].item(), key=key, step=self.epoch)

                    self.log(name='MacroPrecision', value=macro_precision, key=key, step=self.epoch)
                    self.log(name='MacroRecall', value=macro_recall, key=key, step=self.epoch)
                    self.log(name='MacroF1', value=macro_f1, key=key, step=self.epoch)

                elif metricName == 'BinaryConfusionMatrix':
                    conf_matrix = metricFunc.compute()
                    TP = conf_matrix[0, 0]
                    TN = conf_matrix[1, 1]
                    FP = conf_matrix[1, 0]
                    FN = conf_matrix[0, 1]

                    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
                    precision = TP / (TP + FP + 1e-8)
                    recall = TP / (TP + FN + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)

                    self.log(name='Accuracy', value=accuracy, key=key, step=self.epoch, test_dict=None)
                    self.log(name=f'Precision', value=precision, key=key, step=self.epoch, test_dict=None)
                    self.log(name=f'Recall', value=recall, key=key, step=self.epoch, test_dict=None)
                    self.log(name=f'F1', value=f1, key=key, step=self.epoch, test_dict=None)

                if metricName=='BinaryAccuracy':
                    metric_value = metricFunc.compute()
                    self.log(name=metricName, value=metric_value.item(), key=key, step=self.epoch)

            # Training loss
            loss_mean = np.mean(running_loss_train)
            self.training_loss = loss_mean
            self.log('loss', self.training_loss, key='training', step=self.epoch)

            # Reset metrics after training
            self.reset_metrics()

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

            # Log validation metrics
            key = 'validation'
            self.model_improves = False
            for metricName, metricFunc in self.metrics.items():
                if metricName == 'MulticlassConfusionMatrix':
                    conf_matrix = metricFunc.compute()
                    num_classes = conf_matrix.shape[0]

                    TP = torch.diag(conf_matrix)
                    FP = conf_matrix.sum(dim=0) - TP
                    FN = conf_matrix.sum(dim=1) - TP

                    precision = TP / (TP + FP + 1e-8)  # avoid divide-by-zero
                    recall = TP / (TP + FN + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)

                    macro_precision = precision.mean()
                    macro_recall = recall.mean()
                    macro_f1 = f1.mean()

                    for i in range(num_classes):
                        self.log(name=f'Precision_{i}', value=precision[i].item(), key=key, step=self.epoch)
                        self.log(name=f'Recall_{i}', value=recall[i].item(), key=key, step=self.epoch)
                        self.log(name=f'F1_{i}', value=f1[i].item(), key=key, step=self.epoch)

                    self.log(name='MacroPrecision', value=macro_precision, key=key, step=self.epoch)
                    self.log(name='MacroRecall', value=macro_recall, key=key, step=self.epoch)
                    self.log(name='MacroF1', value=macro_f1, key=key, step=self.epoch)

                elif metricName == 'BinaryConfusionMatrix':
                    conf_matrix = metricFunc.compute()
                    TP = conf_matrix[0, 0]
                    TN = conf_matrix[1, 1]
                    FP = conf_matrix[1, 0]
                    FN = conf_matrix[0, 1]

                    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
                    precision = TP / (TP + FP + 1e-8)
                    recall = TP / (TP + FN + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)

                    self.log(name='Accuracy', value=accuracy, key=key, step=self.epoch, test_dict=None)
                    self.log(name=f'Precision', value=precision, key=key, step=self.epoch, test_dict=None)
                    self.log(name=f'Recall', value=recall, key=key, step=self.epoch, test_dict=None)
                    self.log(name=f'F1', value=f1, key=key, step=self.epoch, test_dict=None)

                    metric_value = float(accuracy)
                    if metric_value > self.metric_value_max:
                        self.metric_value_max = metric_value
                        self.model_improves = True
                    else:
                        self.model_improves = False

                if metricName == 'BinaryAccuracy':
                    metric_value = metricFunc.compute()
                    self.log(name=metricName, value=metric_value.item(), key=key, step=self.epoch)



            # Reset metrics after validation
            self.reset_metrics()

            # Update and log scheduler
            if self.scheduler is not None:
                self.log(name='Scheduler', value=self.scheduler.get_last_lr()[0], key='LR', step=self.epoch)
                self.scheduler.step()
            else:
                self.log(name='Constant', value=self.optimizer.defaults['lr'], key='LR', step=self.epoch)

            # Save checkpoint for last epoch
            self.save_checkpoint(best=False, last=True)

            # Save state_dict if model improves
            if self.model_improves:
                self.save_weights()
                self.save_checkpoint(best=True, last=False)
            elif self.epoch % self.save_checkpoint_every_n == 0:
                self.save_checkpoint(best=False, last=False)

            torch.cuda.empty_cache()
            gc.collect()

    def test_loop(self):
        # Re-Initialize arguments
        self.initialization()

        # Load metrics
        self.load_metrics()
        self.init_metrics()
        self.reset_metrics()

        # Load best weights if specified
        self.load_best_weights()

        # Assign model to evaluation mode
        self.model.eval()

        test_loader = tqdm(self.test_dataloader, colour='blue')
        running_loss_test = []

        for i, batch in enumerate(test_loader):
            self.batch_num = i + 1

            # Load input and target
            input = batch[f'{self.args.input_name}']
            target = batch[f'{self.args.target_name}']

            # Sending data to GPU if possible
            input, target, self.model = input.to(self.device), target.to(self.device), self.model.to(self.device)

            with torch.amp.autocast(device_type=str(self.device), dtype=self.amp_dtype, enabled=self.use_amp):
                with torch.no_grad():
                    pred = self.model(input)

                    # Make sure there are no NaNs
                    if torch.isnan(pred).any():
                        print('There are NaNs')

                    if self.final_activation:
                        pred = self.final_activation(pred, dim=1)

                    pred_target = self.model.predict_class(pred)

                    # Compute the loss and its gradients
                    loss = self.loss_fn(pred, target).mean()

            # Compute metrics
            for metricName, metricFunc in self.metrics.items():
                metricFunc = metricFunc.to(self.device)
                x = pred_target.clone().to(torch.int).detach().cpu()
                y = target.clone().argmax(dim=1).detach().cpu()
                metricFunc.update(x, y)

            loss_test = loss.item()
            # loss tracking
            running_loss_test.append(loss_test)

            # Logs
            test_loader.set_description(f'Testing: batch loss = {loss_test}')

        # Logging test loss
        loss_test = np.mean(running_loss_test)
        self.test_loss = loss_test
        self.log('loss', self.test_loss, key='test', step=self.epoch)

        # Log testmetrics
        key = 'test'
        path = Path(f'{self.args.project_dir}/TestResults/{self.args.subproject_name}/{self.args.full_experiment_name}/results.json')
        path.mkdir(parents=True, exist_ok=True)
        test_dict = {}
        for metricName, metricFunc in self.metrics.items():
            if metricName == 'MulticlassConfusionMatrix':
                conf_matrix = metricFunc.compute()
                num_classes = conf_matrix.shape[0]

                TP = torch.diag(conf_matrix)
                FP = conf_matrix.sum(dim=0) - TP
                FN = conf_matrix.sum(dim=1) - TP

                precision = TP / (TP + FP + 1e-8)  # avoid divide-by-zero
                recall = TP / (TP + FN + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)

                macro_precision = precision.mean()
                macro_recall = recall.mean()
                macro_f1 = f1.mean()

                for i in range(num_classes):
                    self.log(name=f'Precision_{i}', value=precision[i].item(), key=key, step=self.epoch, test_dict=test_dict)
                    self.log(name=f'Recall_{i}', value=recall[i].item(), key=key, step=self.epoch, test_dict=test_dict)
                    self.log(name=f'F1_{i}', value=f1[i].item(), key=key, step=self.epoch, test_dict=test_dict)

                self.log(name='MacroPrecision', value=macro_precision, key=key, step=self.epoch, test_dict=test_dict)
                self.log(name='MacroRecall', value=macro_recall, key=key, step=self.epoch, test_dict=test_dict)
                self.log(name='MacroF1', value=macro_f1, key=key, step=self.epoch, test_dict=test_dict)

            elif metricName == 'BinaryAccuracy':
                metric_value = metricFunc.compute()
                self.log(name=metricName, value=metric_value.item(), key=key, step=self.epoch, test_dict=test_dict)

        # Save test results to file
        with open(path, 'w') as f:
            json.dump(test_dict, f, indent=4)