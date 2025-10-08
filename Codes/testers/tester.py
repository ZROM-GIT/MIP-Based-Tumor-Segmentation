import json
import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import os
import pickle

from pet_ct.main_utils.load_metrics import LoadMetrics
from pathlib import Path
from pet_ct.masks.masks_loader import masks
from pet_ct.secondary_utils.AffineMatrices import affine_matrices
from pet_ct.secondary_utils.createMIP_new import create_mip_new
# from pet_ct.secondary_utils.mip_tumor_precision_recall_detection import per_tumor_mip_detection
from pet_ct.main_utils.my_utils import *
from tqdm import tqdm

class Tester:
    def __init__(self,
                 args: dict,
                 model,
                 test_dataloader: torch.utils.data.DataLoader,
                 loss_fn,
                 device: str = 'cuda',
                 logger=None,
                 transforms=None):

        self.args = args
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.model = model
        self.device = device
        self.logger = logger
        self.transforms = transforms


    def initialization(self):
        self.apply_inverse_transforms = getattr(self.args, 'apply_inverse_transforms', False)
        self.save_prediction = getattr(self.args, 'save_prediction', False)
        self.save_analysis = getattr(self.args, 'save_analysis', True)
        self.final_activation = getattr(self.args, 'final_activation', None)  # Optional final activation for model
        self.use_sliding_window_inference = getattr(self.args, 'use_sliding_window_inference', False)
        self.masks = getattr(self.args, 'masks', False)
        self.detect_tumors = getattr(self.args, 'detect_tumors', False)
        self.prediction_threshold = getattr(self.args, 'prediction_threshold', 0.5)
        self.use_sliding_window_inference = getattr(self.args, 'use_sliding_window_inference', False)
        self.use_amp = getattr(self.args, 'use_amp', False)
        self.amp_dtype = getattr(torch, getattr(self.args, 'amp_dtype', 'float32'))
        self.create_mips = getattr(self.args, 'create_mips', False)

        if self.final_activation:
            self.final_activation = getattr(torch.nn.functional, self.final_activation)
        if self.masks:
            self.masks = masks(self.args)
        if self.detect_tumors:
            # Initialize dataframe
            self.data = None
            self.Precision = []
            self.Recall = []

    def pred_binary(self, pred, threshold=0.5):
        pred = torch.threshold(pred, threshold=threshold, value=0)
        pred[pred >= threshold] = 1
        return pred

    def load_metrics(self):
        self.metrics_obj = LoadMetrics(self.args)
        self.metrics_obj.create_metrics()

        # Creating a dictionary to keep metrics values
        self.metrics_values = {k: [] for k in self.metrics_obj.metrics}

    def compute_metrics(self, y, y_pred):
        y, y_pred = y.detach().cpu(), y_pred.detach().cpu()
        y, y_pred = y.to(dtype=torch.float), y_pred.to(dtype=torch.float)
        for metricName, metricFunc in self.metrics_obj.metrics.items():
            self.metrics_values[metricName].append(np.array(metricFunc(y_pred=y_pred, y=y)))

    def compute_mean_metrics(self):
        for metricName, metricFunc in self.metrics_obj.metrics.items():
            self.metrics_values[metricName] = np.mean(self.metrics_values[metricName])


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

    def create_mips(self, input, target, pred_target, num_of_mips=16):
        W, D, H = pred_target.shape[-3:] #input.shape[-3:]
        B, C = target.shape[:2]
        if hasattr(pred_target, 'meta'):
            pred_target = monai.data.MetaTensor(pred_target, meta=pred_target.meta, applied_operations=pred_target.applied_operations)
        else:
            pred_target = monai.data.MetaTensor(pred_target, meta=input.meta, applied_operations=input.applied_operations)
        orient_forward = monai.transforms.Orientation(axcodes='ILP')
        pred_target_mips = torch.zeros(B, C, H, W, num_of_mips)

        for i, pred_target_i in enumerate(pred_target):
            affine_i = pred_target.affine[i]
            pred_target_i.affine = affine_i
            pred_target_i = orient_forward(pred_target_i)
            pred_target_i = torch.argmax(pred_target_i, dim=0).squeeze()

            all_angles = np.linspace(0, 180 - 180 / num_of_mips, num=num_of_mips)
            # W, D, H = pred_target_i.shape
            pred_target_mips_i = torch.zeros(H, W, num_of_mips)

            for j in range(0, num_of_mips):
                horizontal_angle = all_angles[j]
                pred_target_mip_i = create_mip_new(img=pred_target_i, return_inds=False,
                                                   horizontal_angle=horizontal_angle, modality='seg', device='cuda')

                pred_target_mips_i[:, :, j] = torch.Tensor(pred_target_mip_i)

            pred_target_mips_i = torch.nn.functional.one_hot(pred_target_mips_i.to(dtype=torch.int64), num_classes=2)
            pred_target_mips_i = pred_target_mips_i.movedim(source=-1, destination=0)
            pred_target_mips[i] = pred_target_mips_i

        return pred_target_mips


    def write_output(self, data: dict):
        OUTPUT_DIR = Path(self.args.project_dir) / getattr(self.args, 'save_test_results_path') / f"{self.args.subproject_name}"

        os.makedirs(OUTPUT_DIR) if not os.path.isdir(OUTPUT_DIR) else None

        output_filename =  f"{self.args.full_experiment_name}.json"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)

        print(f"Writing output to {output_filepath}")
        # Load output file
        if os.path.exists(output_filepath):
            # pylint: disable=C0103
            with open(output_filepath, 'r', encoding='utf-8') as f:
                all_output_data = json.load(f)
        else:
            all_output_data = {}

        # Add new data and write to file
        all_output_data.update(data)

        # correction
        all_output_data = {key: str(val) for key, val in all_output_data.items()}

        # pylint: disable=C0103
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, indent=4)

    def create_analysis(self):
        self.write_output(data=self.metrics_values)

    def save_prediction_as_nifti(self, input, target, seg_prediction, mode='nib'):
        saving_path = Path(self.args.project_dir) / getattr(self.args, 'save_test_predictions_path') / self.args.subproject_name / self.args.full_experiment_name

        patient_path = Path(self.test_dataloader.dataset.data[self.batch_num - 1][self.args.input_name])
        if 'PACS' in patient_path.parts:
            patient_name = patient_path.parts[-3]
            saving_path = saving_path / patient_name
        else:
            saving_path = saving_path / patient_path.parts[-3] / patient_path.parts[-2]

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
                affine = affine_matrices['SLA']

                input_nif = nib.Nifti1Image(np.array(input, dtype=np.float32), affine=affine)
                target_nif = nib.Nifti1Image(target.numpy(), affine=affine, dtype=np.int16)
                seg_pred_nif = nib.Nifti1Image(seg_prediction.numpy(), affine=affine, dtype=np.int16)
                all_nif = nib.Nifti1Image(seg_and_pred.numpy(), affine=affine, dtype=np.int16)

                nib.save(input_nif, filename=f'{saving_path}/input.nii.gz')
                nib.save(target_nif, filename=f'{saving_path}/target.nii.gz')
                nib.save(seg_pred_nif, filename=f'{saving_path}/prediction.nii.gz')
                nib.save(all_nif, filename=f'{saving_path}/all.nii.gz')

    def eval(self, batch):

        # Load input and target
        input = batch[f'{self.args.input_name}']
        target = batch[f'{self.args.target_name}']

        # Sending data to GPU if possible
        input, target, self.model = input.to(self.device), target.to(self.device), self.model.to(self.device)

        with torch.amp.autocast(device_type=str(self.device), dtype=self.amp_dtype, enabled=self.use_amp):
            with torch.no_grad():
                if 'test' in self.use_sliding_window_inference:
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
                                        dataloader=self.test_dataloader, timing='pre', set='test',
                                        args=self.args, device=self.device)

        if self.create_mips:
            pred_target = self.create_mips(input, target, pred_target, num_of_mips=48)

        # Compute metrics
        for i, _ in enumerate(target):
            if target[i, 1].any(): # 2nd dim: First channel = Background, Second channel = Foreground
                self.compute_metrics(y=torch.unsqueeze(target[i], 0), y_pred=pred_target[i].unsqueeze(0))

        # if self.detect_tumors:
        #     self.data, Precision, Recall = (
        #         per_tumor_mip_detection(suv=batch['SUV_3D'],
        #                                 seg=batch['SEG_3D'],
        #                                 pred=pred_target,
        #                                 start_angle=0,
        #                                 end_angle=180,
        #                                 num_of_mips=16,
        #                                 ver_threshold=0.75,
        #                                 pixel_size_threshold=3,
        #                                 split_tumors=True,
        #                                 visualSNR_threshold=8,
        #                                 IOU_threshold=0.3,
        #                                 data=self.data
        #                                 ))
        #     self.Precision.append(Precision), self.Recall.append(Recall)

        # Inverse transforms (Post-metrics)
        pred_target = InverseTransforms(input=input, pred=pred_target, target=target, transforms=self.transforms,
                                        dataloader=self.test_dataloader, timing='post', set='test',
                                        args=self.args, device=self.device)

        # Save prediction if necessary
        if self.save_prediction:
            self.save_prediction_as_nifti(input=input, target=target, seg_prediction=pred_target)

        return loss.item()

    def test_loop(self):
        # Initialize needed objects
        self.initialization()

        # Load metrics
        self.load_metrics()

        test_loader = tqdm(self.test_dataloader, colour='blue')
        running_loss_test = []

        for i, batch in enumerate(test_loader):
            self.batch_num = i + 1
            test_loader.set_description(f'Batch {self.batch_num} in process')

            # Evaluate data
            loss = self.eval(batch=batch)
            running_loss_test.append(loss)
            test_loader.set_description(f'Testing: batch loss = {loss}')


        # Logging test loss
        loss_test = np.mean(running_loss_test)
        self.test_loss = loss_test
        self.log('loss', self.test_loss, key='test', step=None)

        # Log metrics
        self.log_metrics(keys=['test'])

        if self.detect_tumors:
            if hasattr(self.args, 'detection_save_path'):
                self.data.to_csv(Path(self.args.detection_save_path) / f'conf{self.args.experiment_number}_{self.prediction_threshold}PredThresh.csv')
                with open(Path(self.args.detection_save_path) / f'conf{self.args.experiment_number}_{self.prediction_threshold}PredThresh_Precision.pkl', 'wb') as f:
                    pickle.dump(self.Precision, f)
                with open(Path(self.args.detection_save_path) / f'conf{self.args.experiment_number}_{self.prediction_threshold}PredThresh_Recall.pkl', 'wb') as f:
                    pickle.dump(self.Recall, f)

        self.compute_mean_metrics()
        if self.save_analysis:
            self.create_analysis()