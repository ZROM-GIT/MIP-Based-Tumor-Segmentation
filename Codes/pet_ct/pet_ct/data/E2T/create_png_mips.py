import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

# CONFIG
base_dir = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D"
output_mip_dir = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_png"
modality = "SUV"

os.makedirs(output_mip_dir, exist_ok=True)


# Contrast enhancement using ITK-SNAP-style window/level
def apply_window_level(image, mask=None):
    if mask is None:
        mask = image > 0  # mask out background
    nonzero = image[mask]
    if nonzero.size == 0:
        return np.zeros_like(image)

    level = np.percentile(nonzero, 50)  # median intensity
    window = np.percentile(nonzero, 98) - np.percentile(nonzero, 2)  # spread

    lower = level - window / 2
    upper = level + window / 2
    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower + 1e-8)  # normalize to [0, 1]
    return image


# Create MIP along specified axis
def create_mip(volume, axis=1):
    mip = np.max(volume, axis=axis)
    mip = np.rot90(mip)
    return mip


# Save MIP as high-contrast PNG
def save_mip_as_png(mip, out_path):
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(mip, cmap='gray', vmin=0, vmax=1)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# Walk through all patient/case folders
for root, dirs, files in os.walk(base_dir):
    if f"{modality}.nii.gz" in files:
        path = os.path.join(root, f"{modality}.nii.gz")
        try:
            vol = nib.load(path).get_fdata()
            mip = create_mip(vol)
            mip_contrast = apply_window_level(mip)

            patient = os.path.basename(os.path.dirname(root))
            case = os.path.basename(root)
            out_name = f"{patient}_{case}.png"
            out_path = os.path.join(output_mip_dir, out_name)

            save_mip_as_png(mip_contrast, out_path)
        except Exception as e:
            print(f"❌ Failed on {path}: {e}")
