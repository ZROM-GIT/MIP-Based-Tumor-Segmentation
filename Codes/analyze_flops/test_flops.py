import json
import warnings
from calendar import month_abbr

import monai
import time

from math import ceil
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info
from tqdm import tqdm
from pathlib2 import Path
from pet_ct.secondary_utils.createMIP_new import create_mip_new_torch

from build.lib.pet_ct.main_utils.my_metrics import accuracy
from pet_ct.main_utils.load_metrics import LoadMetrics
from pet_ct.main_utils.my_utils import *


class Tester:
    def __init__(self,
                 args: dict,
                 model,
                 test_dataloader: monai.data.DataLoader,
                 device='cuda',
                 transforms=None):

        self.args = args
        self.test_dataloader = test_dataloader
        self.model = model
        self.device = device
        self.transforms = transforms

    def initialization(self):
        self.epoch = getattr(self.args, 'epoch', 0)
        self.batch_size = getattr(self.args, 'batch_size', 1)
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

        # Assign model to evaluation mode
        self.model.eval()

        test_loader = tqdm(self.test_dataloader, colour='blue')
        running_loss_test = []

        all_gflops = []
        inference_times = [] # seconds
        for i, batch in enumerate(test_loader):
            self.batch_num = i + 1

            # Load input and target
            input = batch[f'{self.args.input_name}']
            target = batch[f'{self.args.target_name}']

            # Sending data to GPU if possible
            input, target, self.model = input.to(self.device), target.to(self.device), self.model.to(self.device)

            with torch.amp.autocast(device_type=str(self.device), dtype=self.amp_dtype, enabled=self.use_amp):
                with torch.no_grad():
                    # Forward pass through the model
                    if self.args.use_sliding_window_inference == []:
                        # Warm-up (especially important for GPU)
                        if self.batch_num == 1:
                            warmup_input_tensor = torch.randn((1,1,400, 400, self.args.num_of_mips)).to(self.device)
                            for _ in range(50):
                                _ = self.model(warmup_input_tensor)

                        W, D, H = input.shape[-3:]  # Input image shape
                        mips = torch.zeros((1, self.args.num_of_mips, W, H), dtype=torch.float32, device=self.device)

                        all_angles = np.linspace(start=0, stop=(180 - 180/ self.args.num_of_mips), num=self.args.num_of_mips)
                        torch.cuda.synchronize()
                        start_time = time.time()
                        for j in range(self.args.num_of_mips):
                            mips[0, j] = create_mip_new_torch(input[0], horizontal_angle=float(all_angles[j]), modality='suv', device='cuda')[0]

                        self.model(mips.unsqueeze(0).to(self.device))
                        torch.cuda.synchronize()
                        end_time = time.time()
                        inference_time = end_time - start_time

                        macs, params = get_model_complexity_info(self.model, tuple(mips.shape), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
                        macs, units =float(macs.split()[0]), str(macs.split()[1])
                        gflops = macs * 2
                        print(f"FLOPs (approx.): {gflops} GFLOPs")  # MACs ≈ FLOPs / 2
                        print(f"Params: {params}")


                    else:
                        torch.cuda.synchronize()
                        start_time = time.time()
                        output = sliding_window_inference(roi_size=(96, 96, 96), sw_batch_size=8,
                                                                            overlap=0.25, mode='constant',
                                                                            predictor=self.model, inputs=input)
                        # self.model(input)
                        torch.cuda.synchronize()
                        end_time = time.time()
                        inference_time = end_time - start_time

                        patch_shape = (1, 96, 96, 96)
                        macs, params = get_model_complexity_info(self.model, patch_shape, as_strings=True,
                                             print_per_layer_stat=True, verbose=True)


                        D, H, W = input.shape[-3:]  # Input image shape
                        pd, ph, pw = 96, 96, 96  # Patch size
                        sd, sh, sw = pd/4, ph/4, pw/4  # Stride / overlap

                        num_patches = (
                                              ceil((D - pd) / sd) + 1
                                      ) * (
                                              ceil((H - ph) / sh) + 1
                                      ) * (
                                              ceil((W - pw) / sw) + 1
                                      )
                        macs, units =float(macs.split()[0]), str(macs.split()[1])
                        macs_total = macs * num_patches
                        flops_total = macs_total * 2  # MACs ≈ FLOPs / 2

                        print(f"Macs (approx.): {macs_total} {units}")
                        print(f"FLOPs (approx.): {flops_total} 'GFLOPs'")  # FLOPs ≈ 2 x MACs
                        gflops = flops_total

            # Compute metrics
            all_gflops.append(float(gflops))
            inference_times.append(float(inference_time))

            # Logs
            test_loader.set_description(f'Testing: batch {i}')

        results = {}
        results['avg_gflops'] = np.mean(all_gflops)
        results['std_gflops'] = np.std(all_gflops)
        results['avg_inference_time (seconds)'] = np.mean(inference_times)
        results['std_inference_time (seconds)'] = np.std(inference_times)
        results['number of parameters'] = params

        filename = f'{self.args.full_experiment_name}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
