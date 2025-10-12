
# from pet_ct.main_utils.connected_components import cc
# import cc3d
# import monai
# import numpy as np
# import torch
# from tqdm import tqdm
#
#
# def create_screened_mips(suv: torch.Tensor,
#                          seg: torch.Tensor,
#                          hguo: torch.Tensor = None,
#                          threshold: float = -1,
#                          horizontal_rot_angles: np.ndarray = np.empty(shape=0),
#                          num_of_mips: int = 16,
#                          volume_threshold=None,
#                          split_tumors=False,
#                          filter_split_tumors_by_contrast=False,
#                          filter_split_tumors_by_gradient=True):
#
#     """Creates 2D SUV, 'screened' SEG, (HGUO) MIP images from 3D image data
#
#         Orientation of 3D image data should be: [Height, Width, Depth]
#
#     Args:
#         - suv: MetaTensor / torch.Tensor
#         - seg: MetaTensor / torch.Tensor
#         - hguo: MetaTensor / torch.Tensor
#         - threshold: Threshold to use when erasing screened segmented tumors
#         - horizontal_rot_angles: The MIP angles required
#         - num_of_mips: Number of MIPs to create from 3D SUV
#         - volume_threshold: Any tumor with smaller volume than this will be removed from 3D SUV
#
#     Returns:
#         - suv_mips, suv_inds_mips, seg_mips, hguo_mips (if not None)
#     """
#
#     threshold /= 100
#     spacing = suv.pixdim
#
#     # Calculating Radians from angles
#     if horizontal_rot_angles is []:
#         raise ValueError('List of rotations angles given to function cannot be empty!')
#     else:
#         for i, ang in enumerate(horizontal_rot_angles):
#             horizontal_rot_angles[i] = (2 * np.pi) * (ang / 360)
#
#     HEIGHT = suv.shape[0]
#     WIDTH = suv.shape[1]
#
#     # Pre-allocate MIPs
#     suv_mips = torch.zeros(size=(HEIGHT, WIDTH, len(horizontal_rot_angles)))
#     suv_inds_mips = torch.zeros(size=(HEIGHT, WIDTH, len(horizontal_rot_angles)))
#     seg_mips = torch.zeros(size=(HEIGHT, WIDTH, len(horizontal_rot_angles)))
#     if hguo is not None:
#         hguo_mips = torch.zeros(size=(HEIGHT, WIDTH, len(horizontal_rot_angles)))
#
#     # Loop on the angles I want to create MIPs from
#     for i, ang in enumerate(horizontal_rot_angles):
#         rotation_bilinear = monai.transforms.Affine(rotate_params=(ang, 0, 0),
#                                            image_only=True,
#                                            padding_mode='zeros',
#                                            mode=3)
#
#         rotation_nearest = monai.transforms.Affine(rotate_params=(ang, 0, 0),
#                                            image_only=True,
#                                            padding_mode='zeros',
#                                            mode='nearest')
#
#         suv_rot = rotation_bilinear(suv)
#         seg_rot = rotation_nearest(seg)
#         if hguo is not None:
#             hguo_rot = rotation_nearest(hguo)
#
#         # Calculating MIPs
#         suv_mip, suv_inds_mip = torch.max(suv_rot, dim=2)
#         suv_mips[:, :, i], suv_inds_mips[:, :, i] = suv_mip, suv_inds_mip
#         if hguo is not None:
#             hguo_mips[:, :, i], _ = torch.max((hguo_rot > 0) * 1, dim=2)
#
#         if not seg.any():  # If patient is NEGATIVE
#             seg_mips[:, :, i] = torch.zeros(size=(HEIGHT, WIDTH))
#         else:
#             # Create CC for 3D segmentation
#             CC3D = cc(seg_rot)
#             CC3D_volumes = torch.from_numpy(cc3d.statistics(CC3D)['voxel_counts'].astype(np.int64)) * torch.prod(spacing)
#
#             for label, image in tqdm(cc3d.each(CC3D, binary=True, in_place=True)):
#                 # MIP of one tumor
#                 CC_mip = torch.Tensor(np.max(image, axis=2))
#
#                 # Filter out very small tumors
#                 if CC3D_volumes[label] < volume_threshold:
#                     continue
#
#                 # Find the first occurrence along the third axis
#                 first_occurrences = torch.Tensor(np.argmax(image == 1, axis=2))
#                 first_occurrences[first_occurrences == 0] = torch.inf
#
#                 # Find the last occurrence along the third axis
#                 temp = np.argmax(image[:, :, ::-1] == 1, axis=2)
#                 temp[temp == 0] = image.shape[2] - 1
#                 last_occurrences = torch.Tensor(image.shape[2] - 1 - temp)
#                 last_occurrences[last_occurrences == 0] = -1
#
#                 suv_inds = suv_inds_mips[:, :, i]
#                 suv_inds_ver = 1 * ((first_occurrences <= suv_inds) & (suv_inds <= last_occurrences))
#
#                 ver_ratio = torch.sum(suv_inds_ver) / torch.sum(CC_mip)
#                 if ver_ratio >= threshold:
#                     seg_mips[:, :, i] += CC_mip
#                 elif split_tumors:
#                     if (not filter_split_tumors_by_contrast) & (not filter_split_tumors_by_gradient):
#                         seg_mips[:, :, i] += suv_inds_ver
#
#                     # Connected-components to the parts of the tumor that do come from the tumor
#                     CC_rest = cc(suv_inds_ver)
#                     stats = cc3d.statistics(CC_rest)
#
#                     for part_label, part_image in cc3d.each(CC_rest, binary=True, in_place=True):
#                         if (part_image.sum() * torch.prod(spacing)) < volume_threshold:
#                             pass
#                         elif filter_split_tumors_by_contrast:
#                             bounding_boxes = stats['bounding_boxes']
#                             x1, x2 = bounding_boxes[part_label][0].start, bounding_boxes[part_label][0].stop
#                             y1, y2 = bounding_boxes[part_label][1].start, bounding_boxes[part_label][1].stop
#
#                             foreground = suv_mip[torch.from_numpy(part_image)]
#                             mask = ~torch.from_numpy(part_image)[x1-1:x2+1, y1-1:y2+1]
#                             background = suv_mip[x1-1:x2+1, y1-1:y2+1]
#                             background = background[mask]
#
#                             # Check contrast
#                             if is_in_contrast(foreground=foreground, background=background):
#                                 seg_mips[:, :, i] += part_image
#
#                         elif filter_split_tumors_by_gradient:
#                             #TODO: Take each splitted tumor, find it's contour (optional)
#                             #TODO: Calculate gradient using sobel.
#                             # from skimage import ndimage
#                             # TODO: Take contour or tumor with a large background.
#                             # sx = ndimage.sobel(suv_mip, axis=1)
#                             # sy = ndimage.sobel(suv_mip, axis=0)
#                             # Magnitude = np.sqrt(sx**2 + sy**2)
#                             # Check gradients in tumor, if large enough keep, else get rid of.
#                             # TODO: Don't forget to shift gradients back one step in X and Y axis!!
#                             pass
#
#
#     # All tumor segmentation is 1
#     seg_mips[seg_mips > 0] = 1
#
#     if hguo is not None:
#         return suv_mips, suv_inds_mips, seg_mips, hguo_mips
#     else:
#         return suv_mips, suv_inds_mips, seg_mips
#
#
# def is_in_contrast(foreground, background):
#     ### Original analysis ###
#     # med = np.median(foreground)
#     # back_mean = np.mean(background)
#     # back_std = np.std(background)
#     #
#     # z_score = (med - back_mean)/back_std
#     # if np.abs(z_score) > 2:
#     #     return True
#     # else:
#     #     return False
#
#     background_median = torch.median(background)
#     low_background = background[background < torch.median(background)]
#     low_background_mean = torch.mean(low_background)
#     foreground_mean = torch.mean(foreground)
#     contrast = foreground_mean - low_background_mean
#     visual_snr = contrast * np.sqrt(torch.numel(foreground))
#
#     if visual_snr > 8:
#         return True
#     else:
#         return False
#
#
# def high_gradient(foregorund, background):
#     pass

from pet_ct.main_utils.connected_components import cc
import cc3d
import monai
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_gradient_magnitude(tensor):
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [B,C,H,W]
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(tensor, sobel_x, padding=1)
    grad_y = F.conv2d(tensor, sobel_y, padding=1)
    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return grad.squeeze()


def is_in_contrast(foreground, background):
    fg_mean = torch.mean(foreground)
    bg_sorted = torch.sort(background).values
    bg_low = bg_sorted[:len(bg_sorted)//2]
    bg_mean = torch.mean(bg_low)
    snr = (fg_mean - bg_mean) * torch.sqrt(torch.tensor(len(foreground), device=foreground.device))
    return snr > 8


def is_bright_enough(foreground, background):
    fg_mean = torch.mean(foreground)
    bg_mean = torch.mean(background)
    return fg_mean > bg_mean


@torch.no_grad()
def create_screened_mips(suv: torch.Tensor,
                         seg: torch.Tensor,
                         hguo: torch.Tensor = None,
                         threshold: float = -1,
                         horizontal_rot_angles: np.ndarray = np.empty(shape=0),
                         num_of_mips: int = 16,
                         volume_threshold=None,
                         split_tumors=False,
                         filter_split_tumors_by_contrast=False,
                         filter_split_tumors_by_gradient=True):

    threshold /= 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suv, seg = suv.to(device), seg.to(device)
    if hguo is not None:
        hguo = hguo.to(device)

    spacing = torch.as_tensor(suv.pixdim).to(device)

    if horizontal_rot_angles.size == 0:
        raise ValueError('List of rotation angles cannot be empty!')
    horizontal_rot_angles = np.radians(horizontal_rot_angles)

    HEIGHT, WIDTH = suv.shape[0], suv.shape[1]
    suv_mips = torch.zeros((HEIGHT, WIDTH, len(horizontal_rot_angles)), device=device)
    suv_inds_mips = torch.zeros_like(suv_mips)
    seg_mips = torch.zeros_like(suv_mips)
    if hguo is not None:
        hguo_mips = torch.zeros_like(suv_mips)

    for i, ang in enumerate(horizontal_rot_angles):
        rot_bilinear = monai.transforms.Affine(rotate_params=(ang, 0, 0), image_only=True, padding_mode='zeros', mode=3)
        rot_nearest = monai.transforms.Affine(rotate_params=(ang, 0, 0), image_only=True, padding_mode='zeros', mode='nearest')

        suv_rot = rot_bilinear(suv.cpu()).to(device)
        seg_rot = rot_nearest(seg.cpu()).to(device)
        hguo_rot = rot_nearest(hguo.cpu()).to(device) if hguo is not None else None

        suv_mip, suv_inds_mip = torch.max(suv_rot, dim=2)
        suv_mips[:, :, i], suv_inds_mips[:, :, i] = suv_mip, suv_inds_mip
        if hguo is not None:
            hguo_mips[:, :, i], _ = torch.max((hguo_rot > 0) * 1, dim=2)

        if not seg.any():
            seg_mips[:, :, i] = 0
            continue

        CC3D = cc(seg_rot.cpu())
        CC3D_volumes = torch.from_numpy(cc3d.statistics(CC3D)['voxel_counts']).to(device) * torch.prod(spacing)

        for label, image in tqdm(cc3d.each(CC3D, binary=True, in_place=True)):
            CC_mip = torch.Tensor(np.max(image, axis=2)).to(device)
            if CC3D_volumes[label] < volume_threshold:
                continue

            first_occ = torch.Tensor(np.argmax(image == 1, axis=2)).to(device)
            first_occ[first_occ == 0] = torch.inf

            temp = np.argmax(image[:, :, ::-1] == 1, axis=2)
            temp[temp == 0] = image.shape[2] - 1
            last_occ = torch.Tensor(image.shape[2] - 1 - temp).to(device)
            last_occ[last_occ == 0] = -1

            suv_inds = suv_inds_mips[:, :, i]
            suv_inds_ver = ((first_occ <= suv_inds) & (suv_inds <= last_occ)).float()

            ver_ratio = torch.sum(suv_inds_ver) / torch.sum(CC_mip)
            if ver_ratio >= threshold:
                seg_mips[:, :, i] += CC_mip
            elif split_tumors:
                if not filter_split_tumors_by_contrast and not filter_split_tumors_by_gradient:
                    seg_mips[:, :, i] += suv_inds_ver
                else:
                    CC_rest = cc(suv_inds_ver.cpu())
                    stats = cc3d.statistics(CC_rest)

                    for part_label, part_image in cc3d.each(CC_rest, binary=True, in_place=True):
                        part_tensor = torch.from_numpy(part_image).to(device)
                        if (part_tensor.sum() * torch.prod(spacing)) < volume_threshold:
                            continue

                        if filter_split_tumors_by_contrast:
                            x1, x2 = stats['bounding_boxes'][part_label][0].start, stats['bounding_boxes'][part_label][0].stop
                            y1, y2 = stats['bounding_boxes'][part_label][1].start, stats['bounding_boxes'][part_label][1].stop

                            fg = suv_mip[part_tensor]
                            mask = ~part_tensor[x1-1:x2+1, y1-1:y2+1]
                            bg = suv_mip[x1-1:x2+1, y1-1:y2+1][mask]

                            if is_in_contrast(fg, bg):
                                seg_mips[:, :, i] += part_tensor

                        elif filter_split_tumors_by_gradient:
                            grad = compute_gradient_magnitude(suv_mip)
                            if grad[part_tensor].mean() > 0.15:
                                seg_mips[:, :, i] += part_tensor

                        # Additional check: brightness
                        fg = suv_mip[part_tensor]
                        x1, x2 = stats['bounding_boxes'][part_label][0].start, stats['bounding_boxes'][part_label][0].stop
                        y1, y2 = stats['bounding_boxes'][part_label][1].start, stats['bounding_boxes'][part_label][1].stop
                        bg_mask = ~part_tensor[x1-1:x2+1, y1-1:y2+1]
                        bg = suv_mip[x1-1:x2+1, y1-1:y2+1][bg_mask]
                        if not is_bright_enough(fg, bg):
                            continue

    seg_mips[seg_mips > 0] = 1
    return (suv_mips, suv_inds_mips, seg_mips, hguo_mips) if hguo is not None else (suv_mips, suv_inds_mips, seg_mips)

