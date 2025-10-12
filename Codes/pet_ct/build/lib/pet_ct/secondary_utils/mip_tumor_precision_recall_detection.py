"""
This file is meant to create a function that creates mips generally for a transform (with different arguments available)
"""
import multiprocessing

import cc3d
import monai
import numpy as np
import pandas as pd
import torch
from torchmetrics import JaccardIndex
from pet_ct.main_utils.connected_components import cc
from munch import Munch


def per_tumor_mip_detection(suv: torch.Tensor,
                            seg: torch.Tensor,
                            pred: torch.tensor,
                            start_angle: int = 0,
                            end_angle: int = 180,
                            num_of_mips: int = 16,
                            ver_threshold: float = 0,
                            pixel_size_threshold: float = 0,
                            split_tumors=False,
                            visualSNR_threshold: float = 8,
                            IOU_threshold: float = 0.3,
                            data=None,
                            precision=None,
                            recall=None):

    """Creates dataframe including information about recall detection of mip tumors.

        Args:
            - suv: MetaTensor / torch.Tensor
            - seg: MetaTensor / torch.Tensor
            - start_angle: Starting angle of MIPs to take
            - end_angle: Ending angle of MIPs to take
            - num_of_mips: Number of MIPs to take between the angles mentioned above.
            - angles: The MIP angles required
            - num_of_mips: Number of MIPs to create from 3D SUV
            - pixel_size_threshold: Any tumor with smaller volume than this will be removed from 3D SUV

        Returns:
            - data (detection data)
        """

    if data is None:
        data = pd.DataFrame(columns=['tumor_size', 'tumor_volume',
                                         'tumor_mean_intensity', 'tumor_median_intensity',
                                         'tumor_max_intensity', 'tumor_min_intensity',
                                         'num_of_mips_seen', 'num_of_mips_hit',
                                         'hit_rate (recall)', 'IOUs'])

    # Initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialization
    spatial_dims = suv.shape[-3:]
    HEIGHT = spatial_dims[0]
    WIDTH = DEPTH = spatial_dims[1]
    pred_mips = torch.argmax(pred, dim=1)[0]
    spacing = np.float32(suv.pixdim[0])

    # Copy all torch tensors to CPU
    suv_cpu = np.float32(suv.cpu())[0][0]
    suv_cuda = torch.from_numpy(suv_cpu).to('cuda')
    seg_cpu = np.float32(seg.cpu())[0][0]

    # Extract CC3D, number of tumors in patient and create Dataframe to keep info
    CC3D, N_GT = cc(seg_cpu, dust=True, threshold_dust=pixel_size_threshold, return_N=True)

    # Calculating Radians from angles
    angles = np.deg2rad(np.linspace(start=start_angle, stop=(end_angle - (end_angle / num_of_mips)), num=num_of_mips))

    if not seg.any():  # If patient is NEGATIVE

        # All predictions are False Positives
        Precision_tree = {i: {} for i in range(num_of_mips)}

        for i, angle in enumerate(angles):
            CC_pred_mip, N_mip = cc(pred_mips[:, :, i].cpu(), dust=False, return_N=True)
            pred_mips_stats = cc3d.statistics(CC_pred_mip)

            for j in range(1, N_mip + 1):
                x1, x2 = pred_mips_stats['bounding_boxes'][j][0].start, pred_mips_stats['bounding_boxes'][j][0].stop
                y1, y2 = pred_mips_stats['bounding_boxes'][j][1].start, pred_mips_stats['bounding_boxes'][j][1].stop
                tumor_name = f'{x1}_{x2}_{y1}_{y2}'

                # Prediction is False Positive
                Precision_tree[i][tumor_name] = [0]

        return data, precision, recall

    # Extract statistics about 3D tumors
    CC3D_pixel_counts = cc3d.statistics(CC3D)['voxel_counts'].astype(np.int32)

    # Create munch of arrays to include TP, FP, FN
    info = Munch({'TP': np.zeros(shape=num_of_mips, dtype=np.int32),
                  'FP': np.zeros(shape=num_of_mips, dtype=np.int32),
                  'FN': np.zeros(shape=num_of_mips, dtype=np.int32),
                  'seg_N': np.zeros(shape=num_of_mips, dtype=np.int32),
                  'pred_N': np.zeros(shape=num_of_mips, dtype=np.int32)})

    # Create JACCARD function for IOU calculation and send to cuda
    jaccard = JaccardIndex(task='binary').to('cuda')

    # Create target array
    CC_target = torch.zeros_like(pred_mips, dtype=torch.int32)

    # Create list of dictionaries for all GT tumors
    # tumor_tree = [{j: {h:cc(pred_mips[:, :, h].cpu(), dust=False, return_N=True)[1] for h in range(num_of_mips)} for j in range(num_of_mips)} for i in range(N_GT)]
    # [tumor 1 (3D), tumor 2 (3D), ... , tumor N (3D)]
    # tumor 1 (3D) = {0 (MIP angle 0): {}, 1 (MIP angle 1): {}, ...}
    # In each MIP angle dictionary keep: {'minX_maxX_minY_maxY': IOU value},  'minX_maxX_minY_maxY' of tumor predictions
    Recall_tree = [{j: {} for j in range(num_of_mips)} for i in range(N_GT)]

    # Extract 3D tumors one by one
    for label, tumor_bin_3D in cc3d.each(CC3D, binary=True, in_place=True):  # For each tumor
        # Calculate stats about tumor
        num_of_pixels = int(CC3D_pixel_counts[label])
        volume = num_of_pixels * np.prod(spacing)
        tumor_intensities = suv_cpu[tumor_bin_3D]
        mean_intensity = np.mean(tumor_intensities)
        median_intensity = np.median(tumor_intensities)
        max_intensity = np.max(tumor_intensities)
        min_intensity = np.mean(tumor_intensities)

        # Create confusion matrix coefficients
        visible_in_mips = 0
        hit_in_mips = 0
        IOUs = []

        # list_of_tuples = create_tuples(suv=[suv]*num_of_mips, seg=[seg]*num_of_mips, pred=[pred_mips]*num_of_mips, angle=list(angles) ,i=list(range(num_of_mips)))
        # with Pool(4) as p:
        #     # result = p.starmap(print_stam, [(1, 2, 1, 1, 1), (3, 4, 1, 1, 1), (5, 6, 1, 1, 1), (7, 8, 1, 1, 1)])
        #     result = p.starmap(analyze_mip, list_of_tuples)

        tumor_bin_3D = np.int32(tumor_bin_3D)
        tumor_bin_3D_cuda = torch.Tensor(tumor_bin_3D).to('cuda')

        for i, angle in enumerate(angles):
            # Compute MIPs
            suv_mip, suv_mip_inds, CC_mip, suv_inds_ver = create_mips(suv=suv_cuda, tumor_bin_3D=tumor_bin_3D_cuda, angle=angle, device=device)

            # Compute verified pixels percentage
            ver_ratio = suv_inds_ver.sum() / CC_mip.sum()

            # Load Prediction MIP
            CC_pred_mip, N_mip = cc(pred_mips[:, :, i].cpu(), dust=False, return_N=True)
            pred_mips_stats = cc3d.statistics(CC_pred_mip)
            CC_pred_mip = torch.from_numpy(CC_pred_mip.astype('int32')).to('cuda')

            info.pred_N[i] = N_mip

            if ver_ratio >= ver_threshold:  # It's a legitimate tumor to check
                # Update counter
                visible_in_mips += 1

                # Leave only intersections
                intersections_numbers = torch.unique(CC_mip * CC_pred_mip)[1:]
                intersections_preds = torch.isin(CC_pred_mip, intersections_numbers)
                cc_iou = jaccard(intersections_preds * 1, CC_mip)

                # Update values
                IOUs.append(float(cc_iou))
                if cc_iou > IOU_threshold:
                    hit_in_mips += 1

                # Update target MIP
                CC_target[:, :, i][CC_mip == 1] = 1

                # Update tumors dictionary
                for h in intersections_numbers:
                    x1, x2 = pred_mips_stats['bounding_boxes'][h][0].start, pred_mips_stats['bounding_boxes'][h][0].stop
                    y1, y2 = pred_mips_stats['bounding_boxes'][h][1].start, pred_mips_stats['bounding_boxes'][h][1].stop
                    Recall_tree[label - 1][i][f'{x1}_{x2}_{y1}_{y2}'] = float(cc_iou.cpu())

            elif split_tumors:
                # Connected-components to the parts of the tumor that do come from the tumor
                CC_rest = cc(np.float32(suv_inds_ver.cpu()), dust=True, threshold_dust=pixel_size_threshold, return_N=False)
                stats = cc3d.statistics(CC_rest)

                if len(stats['voxel_counts']) > 1:
                    visible_in_mips += 1
                else:
                    continue

                for part_label, part_image in cc3d.each(CC_rest, binary=True, in_place=True):
                    # Get bounding boxes
                    bounding_boxes = stats['bounding_boxes']
                    x1, x2 = bounding_boxes[part_label][0].start, bounding_boxes[part_label][0].stop
                    y1, y2 = bounding_boxes[part_label][1].start, bounding_boxes[part_label][1].stop

                    part_image = torch.from_numpy(part_image.copy()).to('cuda')
                    foreground = suv_mip[part_image]
                    mask = ~part_image[x1 - 1:x2 + 1, y1 - 1:y2 + 1]
                    background = suv_mip[x1 - 1:x2 + 1, y1 - 1:y2 + 1]
                    background = background[mask]

                    # Create new verified ground truth
                    new_suv_inds_ver = torch.zeros_like(suv_inds_ver)

                    # Check contrast
                    if is_in_contrast(foreground=foreground, background=background, visualSNR_threshold=visualSNR_threshold):
                        new_suv_inds_ver += part_image

                # Leave only intersections
                intersections_numbers = torch.unique(new_suv_inds_ver * CC_pred_mip)[1:]
                intersections_preds = torch.isin(CC_pred_mip, intersections_numbers)
                cc_iou = jaccard(intersections_preds * 1, new_suv_inds_ver)
                IOUs.append(float(cc_iou))
                if cc_iou > IOU_threshold:
                    hit_in_mips += 1

                CC_target[:, :, i][new_suv_inds_ver == 1] = 1

                # Fix non-visible GT tumors
                if not new_suv_inds_ver.any():
                    Recall_tree[label - 1][i] = None

                # Update tumors dictionary
                for h in intersections_numbers:
                    x1, x2 = pred_mips_stats['bounding_boxes'][h][0].start, pred_mips_stats['bounding_boxes'][h][0].stop
                    y1, y2 = pred_mips_stats['bounding_boxes'][h][1].start, pred_mips_stats['bounding_boxes'][h][1].stop
                    Recall_tree[label - 1][i][f'{x1}_{x2}_{y1}_{y2}'] = float(cc_iou.cpu())

        row_to_add = {'tumor_size': num_of_pixels,
                      'tumor_volume': volume,
                      'tumor_mean_intensity': mean_intensity,
                      'tumor_median_intensity': median_intensity,
                      'tumor_max_intensity': max_intensity,
                      'tumor_min_intensity': min_intensity,
                      'num_of_mips_seen': visible_in_mips,
                      'num_of_mips_hit': hit_in_mips,
                      'hit_rate (recall)': np.inf if visible_in_mips == 0 else hit_in_mips/visible_in_mips,
                      'IOUs': IOUs}
        data.loc[len(data)] = row_to_add

    # Create Precision tree for predictions
    Precision_tree = {i:{} for i in range(num_of_mips)}

    for i, angle in enumerate(angles):
        CC_pred_mip, N_mip = cc(pred_mips[:, :, i].cpu(), dust=False, return_N=True)
        pred_mips_stats = cc3d.statistics(CC_pred_mip)

        for j in range(1, N_mip + 1):
            x1, x2 = pred_mips_stats['bounding_boxes'][j][0].start, pred_mips_stats['bounding_boxes'][j][0].stop
            y1, y2 = pred_mips_stats['bounding_boxes'][j][1].start, pred_mips_stats['bounding_boxes'][j][1].stop
            tumor_name = f'{x1}_{x2}_{y1}_{y2}'

            # Preallocate lists for predictions
            Precision_tree[i][tumor_name] = []

            for tumor_num in range(len(Recall_tree)):

                if Recall_tree[tumor_num][i] is not None:
                    if tumor_name in Recall_tree[tumor_num][i].keys():
                        Precision_tree[i][tumor_name].append(Recall_tree[tumor_num][i][tumor_name])

            if Precision_tree[i][tumor_name] == []:
                Precision_tree[i][tumor_name] = [0]

    return data, Precision_tree, Recall_tree


def is_in_contrast(foreground, background, visualSNR_threshold):
    background_median = torch.median(background)
    low_background = background[background < torch.median(background_median)]
    low_background_mean = torch.mean(low_background)
    foreground_mean = torch.mean(foreground)
    contrast = foreground_mean - low_background_mean
    visual_snr = contrast * np.sqrt(torch.numel(foreground))

    if visual_snr > visualSNR_threshold:
        return True
    else:
        return False


def create_mips(suv ,tumor_bin_3D, angle, device):
    rotation_bilinear = monai.transforms.Affine(rotate_params=(angle, 0, 0),
                                                image_only=True,
                                                padding_mode='zeros',
                                                mode='bilinear',
                                                dtype=np.float32)

    rotation_nearest = monai.transforms.Affine(rotate_params=(angle, 0, 0),
                                               image_only=True,
                                               padding_mode='zeros',
                                               mode='nearest',
                                               dtype=np.float32)

    # Bilinear rotation for suv
    suv_rot = rotation_bilinear(suv)

    # Nearest rotation for seg
    tumor_bin_3D_rot = rotation_nearest(tumor_bin_3D)

    match device.type:
         case 'cpu':
            suv_mip = np.max(suv_rot, axis=2)
            suv_inds = np.argmax(suv_rot, axis=2)
         case 'cuda':
            suv_mip, suv_inds = torch.max(suv_rot, dim=2)
            CC_mip, first_occurrences = torch.max(tumor_bin_3D_rot, dim=2)
            first_occurrences[first_occurrences == 0] = 1000

            # Cast CC_mip & first_occurrences to int32
            CC_mip, first_occurrences = CC_mip.to(dtype=torch.int32), first_occurrences.to(dtype=torch.int32)

            # Find the last occurrence along the third axis
            temp = torch.argmax(torch.flip(tumor_bin_3D_rot, dims=[2]), dim=2)
            temp[temp == 0] = tumor_bin_3D_rot.shape[2] - 1
            last_occurrences = tumor_bin_3D_rot.shape[2] - temp - 1
            last_occurrences[last_occurrences == 0] = -1

            suv_inds_ver = ((first_occurrences <= suv_inds) & (suv_inds <= last_occurrences)) * 1

    return suv_mip, suv_inds, CC_mip, suv_inds_ver