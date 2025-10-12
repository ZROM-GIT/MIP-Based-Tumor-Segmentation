# This file is meant to turn 3D axial image data into 2D MIPs

# IMPORTS
import numpy as np
import monai


def createMIP(np_img: np.array, return_inds: bool = False, horizontal_angle: int = 0, modality: str = 'suv') -> np.array:
    '''Creates 2D MIP image from 3D image data

    Input:
    - np_img: 3D numpy.array image.
    - return_inds: Whether to return index of pixel take in MIP (True/False)
    - horizontal_angle: The angle of MIP required in degrees.
    - modality: The modality you want to create the MIP from.
                This is important because the effects of this code on different modalities differ.
                For 'seg' (segmentation) or 'hguo' (high glucose uptaking organs), we use the 'nearest' interpolation method.
                For 'img' (image) or 'suv' (standardized uptaking values), we use the 'bilinear' interpolation method.

     Output:
     - 2D numpy.array MIP image.
     - (Optional) Index matrix of the maximum intensity pixels
     '''

    # Calculating Radians from angles
    rad_angle = (2 * np.pi) * (horizontal_angle / 360)

    # Alligning modality with date type
    if modality == 'seg' or modality == 'hguo':
        mode = 'nearest'
    else:
        mode = 'bilinear'

    # Rotating mumpy image along horizontal axis
    rotation = monai.transforms.Affine(rotate_params=(rad_angle, 0, 0),
                                       image_only=True,
                                       padding_mode='zeros',
                                       mode=mode)
    np_img_rot = rotation(np_img)

    if modality == 'seg' or modality == 'hguo':
        np_img_rot.astype('int8')

    # Computing maximum along axes
    np_mip = np.amax(np_img_rot, axis=1)

    if return_inds:
        # Compute index of maximum intensity pixel along axis
        np_inds = np.argmax(np_img_rot, axis=1)

        # Rotating 2d mip
        np_mip = np.rot90(np_mip, k=-1)
        np_inds = np.rot90(np_inds, k=-1)

        return np_mip, np_inds
    else:
        # Rotating 2d mip
        np_mip = np.rot90(np_mip, k=-1)

        return np_mip