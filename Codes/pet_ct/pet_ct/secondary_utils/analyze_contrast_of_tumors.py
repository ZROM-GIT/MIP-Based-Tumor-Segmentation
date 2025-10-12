import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import monai
import cc3d
from tqdm import tqdm
from pathlib import Path
from pet_ct.main_utils.connected_components import cc
from create_screened_mips import compute_gradient_magnitude

# --- Configuration ---
dataset_root = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D')  # e.g., /mnt/sda1/.../E2T_MIPs/MIPs32
output_plot_dir = Path('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_tumor_analysis')
output_plot_dir.mkdir(parents=True, exist_ok=True)

load_nifti = monai.transforms.Compose([
    monai.transforms.LoadImage(image_only=True, ensure_channel_first=True),
    monai.transforms.Orientation(axcodes='ILP')
])

# --- Storage ---
contrast_scores = []
gradient_means = []
brightness_ratios = []

print("Analyzing tumors for contrast/gradient/brightness...")
for trial_path in tqdm(sorted(dataset_root.glob('*/*'))):
    suv = load_nifti(trial_path / "SUV.nii.gz")[0].squeeze()
    seg = load_nifti(trial_path / "SEG.nii.gz")[0].squeeze()

    for i in range(suv.shape[-1]):  # Over MIPs
        suv_mip = suv[..., i]
        seg_mip = seg[..., i]

        if seg_mip.sum() == 0:
            continue

        CC = cc(seg_mip.cpu())
        stats = cc3d.statistics(CC)

        for label, mask in cc3d.each(CC, binary=True, in_place=True):
            mask = mask.copy()  # avoid torch warning
            mask_t = torch.from_numpy(mask).bool()

            if mask_t.sum() == 0:
                continue

            x1, x2 = stats['bounding_boxes'][label][0].start, stats['bounding_boxes'][label][0].stop
            y1, y2 = stats['bounding_boxes'][label][1].start, stats['bounding_boxes'][label][1].stop

            fg = suv_mip[mask_t]
            bg_mask = ~mask_t[x1-1:x2+1, y1-1:y2+1]
            bg = suv_mip[x1-1:x2+1, y1-1:y2+1][bg_mask]

            if fg.numel() == 0 or bg.numel() == 0:
                continue

            # --- Contrast ---
            bg_sorted = torch.sort(bg).values
            bg_low = bg_sorted[:len(bg_sorted) // 2]
            snr = (fg.mean() - bg_low.mean()) * torch.sqrt(torch.tensor(len(fg)))
            contrast_scores.append(snr.item())

            # --- Brightness ---
            brightness_ratios.append((fg.mean() / (bg.mean() + 1e-6)).item())

            # --- Gradient ---
            grad = compute_gradient_magnitude(suv_mip)
            gradient_means.append(grad[mask_t].mean().item())

# --- Plotting ---
def save_hist(data, title, xlabel, filename):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot_dir / filename)
    plt.close()

save_hist(contrast_scores, "SNR (Tumor vs Background)", "SNR", "snr_histogram.png")
save_hist(gradient_means, "Mean Gradient Magnitude", "Gradient", "gradient_histogram.png")
save_hist(brightness_ratios, "Brightness Ratio (fg / bg)", "Brightness Ratio", "brightness_histogram.png")

print(f"Saved histograms to {output_plot_dir}")
