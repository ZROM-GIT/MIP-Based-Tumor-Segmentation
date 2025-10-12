"""
This file is meant to create segmentation mips inds
"""

import numpy as np
import monai

def create_seg_inds_mips(seg: np.array, return_inds: bool = False, horizontal_angle: int = 0, device: str = 'cpu') -> np.array:

    # Calculating Radians from angles
    rad_angle = (2 * np.pi) * (horizontal_angle / 360)

    mode = 'nearest'
    if device == 'cuda':
        # Rotating cupy image along horizontal axis
        rotation = monai.transforms.Affine(rotate_params=(rad_angle, 0, 0),
                                           image_only=True,
                                           padding_mode='zeros',
                                           mode=mode,
                                           device='cuda')

    else:
        # Rotating mumpy image along horizontal axis
        rotation = monai.transforms.Affine(rotate_params=(rad_angle, 0, 0),
                                           image_only=True,
                                           padding_mode='zeros',
                                           mode=mode)
        # Apply rotation on 3d segmentation
        seg_rot = np.array(rotation(seg))

        # Find the first occurrence along the third axis
        first_occurrences = np.argmax(seg_rot == 1, axis=2)

        # Find the last occurrence along the third axis
        temp = np.argmax(seg_rot[:, :, ::-1] == 1, axis=2)
        temp[temp == 0] = seg_rot.shape[2] - 1
        last_occurrences = seg_rot.shape[2] - 1 - temp

        return first_occurrences, last_occurrences

