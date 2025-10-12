from pet_ct.main_utils.connected_components import cc
import cc3d
import monai
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.sparse import coo_matrix

def compute_overlap_matrix(rotated_labels: np.ndarray,
                           rotated_original_labels: np.ndarray):
    """
    Overlap matrix: rows = rotated CCs, cols = original CCs (after rotation).
    Entry [i, j] = # voxels in rotated CC i from original CC j.
    """
    r = rotated_labels.flatten()
    o = rotated_original_labels.flatten()

    M = r.max() + 1  # number of rotated labels
    N = o.max() + 1  # number of original labels

    mask = (r > 0) & (o > 0)
    r = r[mask]
    o = o[mask]

    data = np.ones_like(r, dtype=np.int32)
    # M = r.max() + 1  # number of rotated labels
    # N = o.max() + 1  # number of original labels

    matrix = coo_matrix((data, (r, o)), shape=(M, N)).toarray()

    return matrix

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

    spacing = torch.as_tensor(suv.pixdim).to(device=device, dtype=torch.float32)

    if horizontal_rot_angles.size == 0:
        raise ValueError('List of rotation angles cannot be empty!')
    horizontal_rot_angles = np.radians(horizontal_rot_angles)

    HEIGHT, WIDTH = suv.shape[0], suv.shape[1]
    suv_mips = torch.zeros((HEIGHT, WIDTH, len(horizontal_rot_angles)), device=device)
    suv_inds_mips = torch.zeros_like(suv_mips)
    seg_mips = torch.zeros_like(suv_mips)
    if hguo is not None:
        hguo_mips = torch.zeros_like(suv_mips)

    tumor_visibility = {}  # Key: 3D label, Value: set of angles (indices) where it is visible

    # Compute 3D connected components once for mapping
    if not seg.any():
        return {
            'suv_mips': suv_mips,
            'suv_inds_mips': suv_inds_mips,
            'seg_mips': seg_mips,
            'hguo_mips': hguo_mips if hguo is not None else None,
            'tumor_visibility': None,
            'tumor_volumes': [],
            'missed_tumors': [],
        }

    original_CC3D = cc(seg.cpu(), dust=True, threshold_dust=float(volume_threshold/spacing.prod())).astype(np.int32)
    for label in np.unique(original_CC3D)[1:]:
        tumor_visibility[label] = set()

    for i, ang in enumerate(horizontal_rot_angles):
        rot_bilinear = monai.transforms.Affine(rotate_params=(ang, 0, 0), image_only=True, padding_mode='zeros', mode=3, device=device)
        rot_nearest = monai.transforms.Affine(rotate_params=(ang, 0, 0), image_only=True, padding_mode='zeros', mode='nearest', device=device)

        suv_rot = rot_bilinear(suv.cpu()).to(device)
        seg_rot = rot_nearest(seg.cpu()).to(device)
        hguo_rot = rot_nearest(hguo.cpu()).to(device) if hguo is not None else None

        suv_mip, suv_inds_mip = torch.max(suv_rot, dim=2)
        suv_mips[:, :, i], suv_inds_mips[:, :, i] = suv_mip, suv_inds_mip
        if hguo is not None:
            hguo_mips[:, :, i], _ = torch.max((hguo_rot > 0) * 1, dim=2)

        CC3D = cc(seg_rot.cpu(), dust=True, threshold_dust=float(volume_threshold/spacing.prod()))
        CC3D_volumes = (torch.from_numpy(cc3d.statistics(CC3D)['voxel_counts']).to(device) * torch.prod(spacing)).to(dtype=torch.float32)

        original_CC3D_rotated = rot_nearest(original_CC3D).astype(np.int32)

        overlap = compute_overlap_matrix(original_CC3D_rotated, CC3D)
        # Use overlap to track tumor contributions and build tumor_visibility[label].add(i)
        correspondence = dict()
        for j in range(1, overlap.shape[1]):
            correspondence[j] = list(np.where(overlap[:,j] > 0)[0])

        for label, image in tqdm(cc3d.each(CC3D, binary=True, in_place=True), desc=f'Angle {i}'):
            CC_mip = torch.Tensor(np.max(image, axis=2)).to(device)

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
                for part_label in correspondence[label]:
                    tumor_visibility[part_label].add(i)

            elif split_tumors:
                CC_rest = cc(suv_inds_ver.cpu())
                stats = cc3d.statistics(CC_rest)

                for part_label, part_image in cc3d.each(CC_rest, binary=True, in_place=True):
                    part_tensor = torch.from_numpy(part_image).to(device)
                    if (part_tensor.sum() * torch.prod(spacing)) < volume_threshold:
                        continue

                    include = True
                    if filter_split_tumors_by_contrast:
                        x1, x2 = stats['bounding_boxes'][part_label][0].start, stats['bounding_boxes'][part_label][0].stop
                        y1, y2 = stats['bounding_boxes'][part_label][1].start, stats['bounding_boxes'][part_label][1].stop
                        fg = suv_mip[part_tensor]
                        mask = ~part_tensor[x1-1:x2+1, y1-1:y2+1]
                        bg = suv_mip[x1-1:x2+1, y1-1:y2+1][mask]
                        if not is_in_contrast(fg, bg):
                            include = False
                    if filter_split_tumors_by_gradient and include:
                        grad = compute_gradient_magnitude(suv_mip)
                        if grad[part_tensor].mean() <= 0.15:
                            include = False

                    if include:
                        seg_mips[:, :, i] += part_tensor
                        # Map partial tumor back to original label (approximate)
                        for part_label in correspondence[label]:
                            tumor_visibility[part_label].add(i)

    seg_mips[seg_mips > 0] = 1
    missed_tumors = [label for label, vis in tumor_visibility.items() if len(vis) == 0]

    return {
        'suv_mips': suv_mips,
        'suv_inds_mips': suv_inds_mips,
        'seg_mips': seg_mips,
        'hguo_mips': hguo_mips if hguo is not None else None,
        'tumor_visibility': tumor_visibility,
        'tumor_volumes': CC3D_volumes[1:],
        'missed_tumors': missed_tumors,
    }

