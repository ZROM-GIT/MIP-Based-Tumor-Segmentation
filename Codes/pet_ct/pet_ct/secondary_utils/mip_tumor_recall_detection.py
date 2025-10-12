"""
This file is meant to create a function that creates mips generally for a transform (with different arguments available)
"""

import cc3d
import monai
import numpy as np
import pandas as pd
import torch
from torchmetrics import JaccardIndex
from pet_ct.main_utils.connected_components import cc
import numba
from multiprocessing import Pool
import concurrent.futures

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
                            ):

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

    if not seg.any():  # If patient is NEGATIVE
        return data

    # Initialization
    spatial_dims = suv.shape[-3:]
    HEIGHT = spatial_dims[0]
    WIDTH = DEPTH = spatial_dims[1]
    pred_mips = torch.argmax(pred, dim=1)[0]

    # Strip 3D data from redundant dimensions
    suv, seg = suv[0][0], seg[0][0]
    spacing = suv.pixdim

    # Extract CC3D, number of tumors in patient and create Dataframe to keep info
    CC3D, N_GT = cc(np.float32(seg), dust=True, threshold_dust=pixel_size_threshold, return_N=True)

    # Calculating Radians from angles
    angles = np.deg2rad(np.linspace(start=start_angle, stop=(end_angle - (end_angle / num_of_mips)), num=num_of_mips))

    # Extract statistics about 3D tumors
    CC3D_pixel_counts = torch.from_numpy(cc3d.statistics(CC3D)['voxel_counts'].astype(np.int64))

    # Extract 3D tumors one by one
    for label, tumor_bin_3D in cc3d.each(CC3D, binary=True, in_place=True):  # For each tumor
        tumor_bin_3D = tumor_bin_3D.copy()  # Copy tumor to avoid non-writable numpy array
        # Calculate stats about tumor
        num_of_pixels = int(CC3D_pixel_counts[label])
        volume = float(num_of_pixels * torch.prod(spacing))
        tumor_intensities = suv[torch.from_numpy(tumor_bin_3D)].to(dtype=torch.float32)
        mean_intensity = float(tumor_intensities.mean())
        median_intensity = float(tumor_intensities.median())
        max_intensity = float(tumor_intensities.max())
        min_intensity = float(tumor_intensities.min())

        # Create confusion matrix coefficients
        visible_in_mips = 0
        hit_in_mips = 0
        IOUs = []

        for i, angle in enumerate(angles):  # For each angle to create MIP from
            # Create and apply rotation functions on suv & seg
            rotation_bilinear = monai.transforms.Affine(rotate_params=(angle, 0, 0),
                                                        image_only=True,
                                                        padding_mode='zeros',
                                                        mode='bilinear')

            rotation_nearest = monai.transforms.Affine(rotate_params=(angle, 0, 0),
                                                       image_only=True,
                                                       padding_mode='zeros',
                                                       mode='nearest')
            suv_rot = rotation_bilinear(suv)
            suv_mip, suv_inds = torch.max(suv_rot, dim=2)
            tumor_bin_3D_rot = rotation_nearest(tumor_bin_3D)

            # MIP of one tumor
            CC_mip = torch.from_numpy(np.max(tumor_bin_3D_rot, axis=2)) * 1

            # Find the first occurrence along the third axis
            first_occurrences = torch.from_numpy(np.argmax(tumor_bin_3D_rot == 1, axis=2)).to(torch.float)
            first_occurrences[first_occurrences == 0] = torch.inf

            # # Find the last occurrence along the third axis
            temp = np.argmax(torch.flip(tumor_bin_3D_rot, dims=[2]) == 1, axis=2)
            temp[temp == 0] = tumor_bin_3D_rot.shape[2] - 1
            last_occurrences = torch.Tensor(tumor_bin_3D_rot.shape[2] - 1 - temp)
            last_occurrences[last_occurrences == 0] = -1

            suv_inds_ver = ((first_occurrences <= suv_inds) & (suv_inds <= last_occurrences)) * 1
            ver_ratio = suv_inds_ver.sum() / CC_mip.sum()

            # Load Prediction MIP
            CC_pred_mip, N_mip = cc(pred_mips[:, :, i].cpu(), dust=False, return_N=True)
            CC_pred_mip = torch.from_numpy(CC_pred_mip.astype('int'))

            if ver_ratio >= ver_threshold:  # It's a legitimate tumor to check
                # Update counter
                visible_in_mips += 1

                # Leave only intersections
                intersections_numbers = torch.unique(CC_mip * CC_pred_mip)[1:]
                intersections_preds = torch.isin(CC_pred_mip, intersections_numbers)
                jaccard = JaccardIndex(task='binary')
                cc_iou = jaccard(intersections_preds * 1, CC_mip)
                IOUs.append(cc_iou)
                if cc_iou > IOU_threshold:
                    hit_in_mips += 1

            elif split_tumors:
                # Connected-components to the parts of the tumor that do come from the tumor
                CC_rest = cc(np.float32(suv_inds_ver), dust=True, threshold_dust=pixel_size_threshold, return_N=False)
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

                    foreground = suv_mip[torch.from_numpy(part_image.copy())]
                    mask = ~torch.from_numpy(part_image.copy())[x1 - 1:x2 + 1, y1 - 1:y2 + 1]
                    background = suv_mip[x1 - 1:x2 + 1, y1 - 1:y2 + 1]
                    background = background[mask]

                    # Create new verified ground truth
                    new_suv_inds_ver = torch.zeros_like(suv_inds_ver)

                    # Check contrast
                    if is_in_contrast(foreground=foreground, background=background, visualSNR_threshold=visualSNR_threshold):
                        new_suv_inds_ver += torch.from_numpy(part_image.copy())

                # Leave only intersections
                intersections_numbers = torch.unique(new_suv_inds_ver * CC_pred_mip)[1:]
                intersections_preds = torch.isin(CC_pred_mip, intersections_numbers)
                jaccard = JaccardIndex(task='binary')
                cc_iou = jaccard(intersections_preds * 1, new_suv_inds_ver)
                IOUs.append(cc_iou)
                if cc_iou > IOU_threshold:
                    hit_in_mips += 1

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

    return data


def is_in_contrast(foreground, background, visualSNR_threshold):
    background_median = torch.median(background)
    low_background = background[background < torch.median(background)]
    low_background_mean = torch.mean(low_background)
    foreground_mean = torch.mean(foreground)
    contrast = foreground_mean - low_background_mean
    visual_snr = contrast * np.sqrt(torch.numel(foreground))

    if visual_snr > visualSNR_threshold:
        return True
    else:
        return False