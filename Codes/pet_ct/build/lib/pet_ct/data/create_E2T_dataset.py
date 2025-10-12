import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import monai
from monai.transforms import Compose, LoadImage, Orientation
import torch
import torch.nn.functional as F
from monai.networks.nets import AttentionUnet
from monai.transforms import Compose, EnsureChannelFirst, DivisiblePad, ToTensor
from monai.transforms.utils import map_binary_to_indices
import numpy as np
import matplotlib.pyplot as plt
import cc3d
import pandas as pd

MIN_COMPONENTS = 2
MIN_AXIAL_SLICES = 200  # e.g. ~8cm with 3mm spacing

incomplete_cases = []
lost_cancer_cases = []

# Configs
model_weights_path = "/mnt/sda1/PET/Codes/weights/PET41_Padded16angs_HGUO_NoHeart_AttentionUnet.pt"  # UPDATE THIS

# Define model
model = AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=[16, 32, 64, 128, 256],
    strides=[2, 2, 2, 2]
)

# Load weights
checkpoint = torch.load(model_weights_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

# Optional: Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing for MIP
preprocess = Compose([
    DivisiblePad(k=16, method='end'),
    ToTensor()
])

# Define MONAI loader
load = Compose([
    LoadImage(image_only=True, ensure_channel_first=True),
    Orientation(axcodes='ILP')  # Reorients to LPS + Inferior-Left-Posterior
])

def remove_small_components_cc3d(seg_array, min_voxels=3):
    """
    Remove 3D connected components smaller than `min_voxels` using cc3d.
    Keeps only large connected regions.
    """
    labels_out = cc3d.connected_components(seg_array.astype(np.uint8), connectivity=26)
    cleaned = seg_array.copy()
    for label in range(1, labels_out.max() + 1):
        if np.sum(labels_out == label) < min_voxels:
            cleaned[labels_out == label] = 0
    return cleaned


def createMIP(np_img: np.array, return_inds: bool = False, horizontal_angle: int = 0, modality: str = 'suv') -> np.array:

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

# Your actual functions
def create_mips(volume):
    number_of_MIP_directions = 32
    mips = np.zeros([number_of_MIP_directions, volume.shape[1], volume.shape[0]])
    all_angles = np.linspace(0, 360 - 360 / number_of_MIP_directions, num=number_of_MIP_directions)
    for i in range(0, number_of_MIP_directions):
        angle = all_angles[i]
        mip = createMIP(volume, return_inds=False, horizontal_angle=angle, modality='suv')
        mips[i, :, :] = mip

    return mips

def segment_mips(mip_2d):
    """
    Takes a 2D MIP (numpy array) and returns a segmentation mask of same size.
    """
    orig_shape = mip_2d.shape  # (num_of_mips, H_orig, W_orig)
    orig_h, orig_w = orig_shape[-1], orig_shape[-2]
    # Expand to 3D and ensure dtype is correct
    mip_vol = np.expand_dims(mip_2d.astype(np.float32), axis=0)  # (1, H, W)

    # Preprocess (pad to divisible by 32, convert to tensor)
    mip_tensor = preprocess(mip_vol)  # (1, H_pad, W_pad)
    mip_tensor = torch.unsqueeze(mip_tensor, dim=0).to(device)
    pad_h, pad_w = mip_tensor.shape[-1], mip_tensor.shape[-2]

    # Inference
    with torch.no_grad():
        logits = model(mip_tensor)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)  # Shape: (1, 1, H, W)
        x = F.one_hot(preds, num_classes=2)
        x = torch.movedim(x,-1,1)

    pred_mask = x.cpu().detach().numpy()

    # Reverse padding
    h_crop = pad_h - orig_h
    w_crop = pad_w - orig_w
    if w_crop > 0:
        pred_mask = pred_mask[:,:,:,:-w_crop, :]
    if h_crop > 0:
        pred_mask = pred_mask[:,:,:,:,:-h_crop]

    pred_mask = pred_mask[0,1,:,:,:]  # Remove batch and channel dimensions
    assert pred_mask.shape == orig_shape, f"Expected shape {orig_shape}, got {pred_mask.shape}"

    # ✅ Flip height axis to match volumetric Z-axis
    pred_mask = np.flip(pred_mask, axis=2)  # shape: (32, W, H)

    return pred_mask

def find_extreme_slices(label_mask, value):
    indices = np.where(label_mask == value)
    if len(indices[0]) == 0:
        return None
    return np.min(indices[0]), np.max(indices[0])

def crop_and_fix_affine(orig_nib, crop_start, crop_end):
    """Crop the NIfTI volume and fix its affine for the new slice range."""
    data = orig_nib.get_fdata()
    num_of_axial_slices = data.shape[2]
    crop_start, crop_end = num_of_axial_slices - crop_end, num_of_axial_slices - crop_start
    cropped = data[:, :, crop_start:crop_end]
    new_affine = orig_nib.affine.copy()
    new_affine[:3, 3] += crop_start * orig_nib.affine[:3, 2]
    return nib.Nifti1Image(cropped, new_affine, header=orig_nib.header)

def load_nib(path):
    return nib.load(path)

def process_case(case_path, base_output_dir):
    modalities = ['SUV', 'SEG', 'CTres', 'HGUO', 'PET']
    loaded_data = {}
    nib_data = {}

    for mod in modalities:
        path = os.path.join(case_path, f"{mod}.nii.gz")
        if os.path.exists(path):
            loaded_data[mod] = load(path)[0]
            nib_data[mod] = load_nib(path)

    if 'SUV' not in loaded_data or 'SEG' not in loaded_data:
        raise RuntimeError("SUV and SEG are required for processing")

    suv_vol = loaded_data['SUV']
    seg_orig_vol = nib_data['SEG'].get_fdata()
    spacing = nib_data['SUV'].header.get_zooms()[2]
    total_slices = suv_vol.shape[0]

    patient_id = os.path.basename(os.path.dirname(case_path))
    case_id = os.path.basename(case_path)

    # Segment MIPs
    mips = create_mips(suv_vol)
    seg_stack = segment_mips(mips)

    # Check connected components in MIP segmentation
    seg_cc_labels = cc3d.connected_components((seg_stack > 0).astype(np.uint8), connectivity=26)
    num_mip_components = seg_cc_labels.max()

    # Check presence in height dimension
    presence = (seg_stack == 1).any(axis=0).any(axis=0)
    if not presence.any():
        raise RuntimeError("No foreground in MIP segmentation")

    brain_index = np.where(presence)[0].min()
    bladder_index = np.where(presence)[0].max()

    slices_above_brain = int(round(50 / spacing))
    slices_below_bladder = int(round(130 / spacing))

    crop_start = max(0, brain_index - slices_above_brain)
    crop_end = min(total_slices, bladder_index + slices_below_bladder + 1)

    if crop_start == 0:
        crop_start += 1
    if crop_end == total_slices:
        crop_end -= 1
    if crop_end <= crop_start:
        raise RuntimeError(f"Invalid crop range: [{crop_start}, {crop_end}]")

    num_slices = crop_end - crop_start
    is_flagged = False

    if num_mip_components < MIN_COMPONENTS:
        incomplete_cases.append({
            "patient": patient_id,
            "case": case_id,
            "reason": f"Only {num_mip_components} component(s) in MIP segmentation"
        })
        is_flagged = True

    if num_slices < MIN_AXIAL_SLICES:
        incomplete_cases.append({
            "patient": patient_id,
            "case": case_id,
            "reason": f"Only {num_slices} axial slices"
        })
        is_flagged = True

    had_cancer = np.any(seg_orig_vol > 0)
    cancer_removed = False

    cropped = {}
    for mod, nib_img in nib_data.items():
        cropped_img = crop_and_fix_affine(nib_img, crop_start, crop_end)

        if mod == 'SEG':
            seg_data = cropped_img.get_fdata()
            cleaned_seg = remove_small_components_cc3d(seg_data, min_voxels=3)

            if had_cancer and not np.any(cleaned_seg > 0):
                lost_cancer_cases.append({
                    "patient": patient_id,
                    "case": case_id,
                    "reason": "Had cancer before, none left after cropping"
                })

            cropped_img = nib.Nifti1Image(cleaned_seg, cropped_img.affine, header=cropped_img.header)

        cropped[mod] = cropped_img

    # Decide output path
    subfolder = "flagged" if is_flagged else ""
    output_case_path = os.path.join(base_output_dir, subfolder, patient_id, case_id)
    os.makedirs(output_case_path, exist_ok=True)

    # Save NIfTIs
    for mod, cropped_img in cropped.items():
        out_path = os.path.join(output_case_path, f"{mod}.nii.gz")
        nib.save(cropped_img, out_path)

    return {
        "patient": patient_id,
        "case": case_id,
        "crop_start": crop_start,
        "crop_end": crop_end,
        "num_slices": num_slices,
        "flag": is_flagged
    }


def process_all(base_input_dir, base_output_dir):
    processed_log = []

    for patient in tqdm(sorted(os.listdir(base_input_dir))):
        patient_path = os.path.join(base_input_dir, patient)
        if not os.path.isdir(patient_path):
            continue

        for case in sorted(os.listdir(patient_path)):
            case_path = os.path.join(patient_path, case)
            if not os.path.isdir(case_path):
                continue

            try:
                result = process_case(case_path, base_output_dir)

                if result is None:
                    continue  # skipped due to missing SUV/SEG or invalid crop range

                # Add to processed case log
                processed_log.append(result)

                print(f"{result['patient']}/{result['case']} done | "
                      f"Axial crop: [{result['crop_start']}, {result['crop_end']}] | "
                      f"Slices: {result['num_slices']} | Flag: {result['flag']}")

            except Exception as e:
                print(f"❌ Failed on {patient}/{case}: {e}")

    # Save logs to CSV
    os.makedirs(base_output_dir, exist_ok=True)

    # Save processed cases
    if processed_log:
        processed_df = pd.DataFrame(processed_log)
        processed_df.to_csv(os.path.join(base_output_dir, "processed_cases.csv"), index=False)
        print(f"\n✅ Processed case summary saved to: processed_cases.csv")
    else:
        print("\n⚠️ No cases were processed.")

    # Save incomplete cases
    if incomplete_cases:
        pd.DataFrame(incomplete_cases).to_csv(
            os.path.join(base_output_dir, "skipped_incomplete_cases.csv"), index=False
        )
        print(f"⚠️ Incomplete cases logged to: skipped_incomplete_cases.csv")

    # Save cancer-removed cases
    if lost_cancer_cases:
        pd.DataFrame(lost_cancer_cases).to_csv(
            os.path.join(base_output_dir, "lost_cancer_cases.csv"), index=False
        )
        print(f"⚠️ Cancer-lost cases logged to: lost_cancer_cases.csv")

if __name__ == "__main__":
    # Run the pipeline
    process_all(
        base_input_dir="/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis",
        base_output_dir="/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D"
    )

    print("\n⚠️ Cases that had cancer but none remained after cropping and cleaning:")
    for case in lost_cancer_cases:
        print("  -", case)

    print("\n⚠️ Skipped incomplete or too-small cases:")
    for case in incomplete_cases:
        print("  -", case)
