import json
import warnings
import monai

from tqdm import tqdm
from pathlib2 import Path

from pet_ct.main_utils.my_metrics import accuracy
from pet_ct.main_utils.load_metrics import LoadMetrics
from pet_ct.main_utils.my_utils import *


class Tester:
    def __init__(self,
                 args: dict,
                 model,
                 test_dataloader: monai.data.DataLoader,
                 loss_fn: monai.losses,
                 device='cuda',
                 logger=None,
                 transforms=None):

        self.args = args
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        self.transforms = transforms

    def initialization(self):
        self.epoch = getattr(self.args, 'epoch', 0)
        self.batch_size = getattr(self.args, 'batch_size', 1)
        self.weights_path = getattr(self.args, 'save_weights_path')
        self.checkpoints_path = getattr(self.args, 'save_checkpoints_path')
        self.final_activation = getattr(self.args, 'final_activation', None)  # Optional final activation for model
        self.use_amp = getattr(self.args, 'use_amp', False)
        self.amp_dtype = getattr(torch, getattr(self.args, 'amp_dtype', 'float32'))

        if self.final_activation:
            self.final_activation = getattr(torch.nn.functional, self.final_activation)

    def pred_binary(self, pred, threshold=0.5):
        pred = torch.threshold(pred, threshold=threshold, value=0)
        pred[pred >= threshold] = 1
        return pred

    def init_metrics(self):
        self.metrics = self.metrics_obj.metrics

    def reset_metrics(self):
        for metricName, metricFunc in self.metrics.items():
            metricFunc.reset()

    def load_metrics(self):
        self.metrics_obj = LoadMetrics(self.args)
        self.metrics_obj.create_metrics()

        # Create variable to keep score of optimized MetricForSaving
        self.metric_value_max = getattr(self.args, 'metric_value_max', 0)

    def log(self, name, value, key='training', step=0, test_dict=None):
        if self.logger is not None:
            self.logger.log({f'{key}/{name}': value}, step=step)
        if test_dict is not None:
            test_dict[f'{key}/{name}'] = float(value)

    def load_best_weights(self):
        path = Path(
            f'{self.args.project_dir}/{self.weights_path}/{self.args.subproject_name}/{self.args.full_experiment_name}/best_weights.pt')
        if not path.exists():
            warnings.warn(
                f'Best weights not found at {path}. Please check the path or run training first. Running on random weights.')
        else:
            self.model.load_state_dict(torch.load(path, map_location=self.device))

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
        path = Path(
            f'{self.args.project_dir}/TestResults/{self.args.subproject_name}/{self.args.full_experiment_name}')
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f'test_results.json'
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
                    self.log(name=f'Precision_{i}', value=precision[i].item(), key=key, step=self.epoch,
                             test_dict=test_dict)
                    self.log(name=f'Recall_{i}', value=recall[i].item(), key=key, step=self.epoch, test_dict=test_dict)
                    self.log(name=f'F1_{i}', value=f1[i].item(), key=key, step=self.epoch, test_dict=test_dict)

                self.log(name='MacroPrecision', value=macro_precision, key=key, step=self.epoch, test_dict=test_dict)
                self.log(name='MacroRecall', value=macro_recall, key=key, step=self.epoch, test_dict=test_dict)
                self.log(name='MacroF1', value=macro_f1, key=key, step=self.epoch, test_dict=test_dict)

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

                self.log(name='Accuracy', value=accuracy, key=key, step=self.epoch, test_dict=test_dict)
                self.log(name=f'Precision', value=precision, key=key, step=self.epoch, test_dict=test_dict)
                self.log(name=f'Recall', value=recall, key=key, step=self.epoch, test_dict=test_dict)
                self.log(name=f'F1', value=f1, key=key, step=self.epoch, test_dict=test_dict)

            elif metricName == 'BinaryAccuracy':
                metric_value = metricFunc.compute()
                self.log(name=metricName, value=metric_value.item(), key=key, step=self.epoch, test_dict=test_dict)

        # Save test results to file
        with open(filepath, 'w') as f:
            json.dump(test_dict, f, indent=4)