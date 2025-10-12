import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# CONFIG
nifti_path = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis/PETCT_1f65acff65/05-06-2007-NA-PET-CT Ganzkoerper nativ u. mit KM-95034/SUV.nii.gz"
output_mip_dir = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/"
modality = "SUV"

# Output name
patient = "PETCT_06a46414eb"
case = "03-12-2006-NA-PET-CT Ganzkoerper  primaer mit KM-38502"
out_name = f"{patient}_{case}.png"
out_path = os.path.join(output_mip_dir, out_name)

# Create output dir if missing
os.makedirs(output_mip_dir, exist_ok=True)


# Contrast adjustment (ITK-SNAP style) with background exclusion
def apply_window_level(image, mask=None):
    if mask is None:
        mask = image > 0  # exclude zero background
    nonzero_vals = image[mask]
    if nonzero_vals.size == 0:
        return np.zeros_like(image)
    level = np.percentile(nonzero_vals, 50)
    window = np.percentile(nonzero_vals, 98) - np.percentile(nonzero_vals, 2)
    lower = level - window / 2
    upper = level + window / 2
    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower + 1e-8)  # prevent divide by 0
    return image


# MIP generation
def create_mip(volume, axis=1):
    mip = np.max(volume, axis=axis)
    mip = np.rot90(mip)
    return mip


# Save as PNG
def save_mip_as_png(mip, out_path):
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(mip, cmap='gray', vmin=0, vmax=1)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# Main
try:
    vol = nib.load(nifti_path).get_fdata()
    mip = create_mip(vol)

    # Apply contrast only on foreground
    mip_windowed = apply_window_level(mip)

    save_mip_as_png(mip_windowed, out_path)
    print(f"✅ Saved: {out_path}")
except Exception as e:
    print(f"❌ Failed on {nifti_path}: {e}")
