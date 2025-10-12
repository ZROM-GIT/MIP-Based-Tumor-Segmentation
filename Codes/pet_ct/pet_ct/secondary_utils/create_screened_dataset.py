# import numpy as np
# import nibabel as nib
# import monai
# from tqdm import tqdm
# from pathlib2 import Path
# from pet_ct.secondary_utils.AffineMatrices import affine_matrices
# from create_screened_mips import create_screened_mips
#
# # ------------------------
# # Configuration
# # ------------------------
# volume_threshold = 25
# threshold = 75
# split_tumors = True
# filter_by_contrast = False
# filter_by_gradient = True
# num_of_mips = 3
# start_angle = 0
# end_angle = 180
#
# input_path = Path('/mnt/sda1/Research/EMA4MICCAI-2025-MIP-Based-Tumor-Segmentation/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis')
# output_path = Path(f'/mnt/sda1/Research/EMA4MICCAI-2025-MIP-Based-Tumor-Segmentation/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs/MIPs{num_of_mips}_{threshold}th_{volume_threshold}vth_{start_angle}_{end_angle}')
#
# affine = affine_matrices['SRA']
# angles = np.linspace(start=start_angle, stop=end_angle - end_angle / num_of_mips, num=num_of_mips)
#
# load_nifti = monai.transforms.Compose([
#     monai.transforms.LoadImage(image_only=True, ensure_channel_first=True),
#     monai.transforms.Orientation(axcodes='ILP')
# ])
#
# # ------------------------
# # Main Loop
# # ------------------------
# trials = sorted(input_path.glob('*/*'))
# print(f"Loaded {len(trials)} patient paths...")
#
# total_tumors = 0
# total_missed_tumors = 0
# total_tumors_volume = 0
# total_missed_tumors_volume = 0
#
# stats_file = output_path / 'tumor_statistics.txt'
# stats_file.parent.mkdir(parents=True, exist_ok=True)
#
# with open(stats_file, 'w') as f:
#     f.write("Tumor Statistics\n")
#     f.write("================\n")
#
# for trial_path in tqdm(trials, desc='Processing cases'):
#     paths = {
#         'SUV': trial_path / 'SUV.nii.gz',
#         'SEG': trial_path / 'SEG.nii.gz',
#         'HGUO': trial_path / 'HGUO.nii.gz'
#     }
#
#     # Load required volumes
#     SUV = load_nifti(paths['SUV'])[0]
#     SEG = load_nifti(paths['SEG'])[0]
#     HGUO = load_nifti(paths['HGUO'])[0] if paths['HGUO'].exists() else None
#
#     # Run MIP creation
#     mips_dict = create_screened_mips(
#         suv=SUV,
#         seg=SEG,
#         hguo=HGUO,
#         horizontal_rot_angles=angles,
#         threshold=threshold,
#         volume_threshold=volume_threshold,
#         split_tumors=split_tumors,
#         filter_split_tumors_by_contrast=filter_by_contrast,
#         filter_split_tumors_by_gradient=filter_by_gradient
#     )
#
#     suv_mips = mips_dict['suv_mips']
#     suv_inds_mips = mips_dict['suv_inds_mips']
#     seg_mips = mips_dict['seg_mips']
#     hguo_mips = mips_dict.get('hguo_mips', None)
#     tumor_visibility = mips_dict['tumor_visibility']
#     missed_tumors = mips_dict['missed_tumors']
#     tumor_volumes = mips_dict['tumor_volumes']
#
#     # Update statistics
#     total_tumors += len(tumor_volumes)
#     total_missed_tumors += len(missed_tumors)
#     if len(tumor_volumes) > 0:
#         total_tumors_volume += float(sum(tumor_volumes.to('cpu')))
#         total_missed_tumors_volume += float(sum(tumor_volumes[list(map(lambda x: x - 1, missed_tumors))]))
#
#     # Output folder
#     out_dir = output_path / trial_path.parent.name / trial_path.name
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     # Save MIPs
#     suv_mips_nif = nib.Nifti1Image(suv_mips.cpu().numpy(), affine)
#     suv_inds_nif = nib.Nifti1Image(suv_inds_mips.cpu().numpy(), affine)
#     seg_mips_nif = nib.Nifti1Image(seg_mips.cpu().numpy(), affine)
#
#     nib.save(suv_mips_nif, out_dir / 'SUV.nii.gz')
#     nib.save(suv_inds_nif, out_dir / 'SUV_inds.nii.gz')
#     nib.save(seg_mips_nif, out_dir / 'SEG.nii.gz')
#
#     if hguo_mips is not None:
#         hguo_mips_nif = nib.Nifti1Image(hguo_mips.cpu().numpy(), affine)
#         nib.save(hguo_mips_nif, out_dir / 'HGUO.nii.gz')
#
#     # Calculate percentages
#     missed_tumors_percent = (total_missed_tumors / total_tumors * 100) if total_tumors > 0 else 0
#     missed_volume_percent = (total_missed_tumors_volume / total_tumors_volume * 100) if total_tumors_volume > 0 else 0
#
#     # Write per-case statistics to file
#     with open(stats_file, 'a') as f:
#         f.write(f"Case: {trial_path.name}\n")
#         f.write(f"  Total Tumors: {total_tumors}\n")
#         f.write(f"  Missed Tumors: {total_missed_tumors} ({missed_tumors_percent:.2f}%)\n")
#         f.write(f"  Total Tumor Volume: {total_tumors_volume:.2f} mm^3\n")
#         f.write(f"  Missed Tumor Volume: {total_missed_tumors_volume:.2f} mm^3 ({missed_volume_percent:.2f}%)\n")
#         f.write("\n")
#
# # Calculate final percentages
# final_missed_tumors_percent = (total_missed_tumors / total_tumors * 100) if total_tumors > 0 else 0
# final_missed_volume_percent = (total_missed_tumors_volume / total_tumors_volume * 100) if total_tumors_volume > 0 else 0
#
# # Print and save overall statistics
# overall_stats = (
#     f"Overall Statistics\n"
#     f"==================\n"
#     f"Total Tumors: {total_tumors}\n"
#     f"Missed Tumors: {total_missed_tumors} ({final_missed_tumors_percent:.2f}%)\n"
#     f"Total Tumor Volume: {total_tumors_volume:.2f} mm^3\n"
#     f"Missed Tumor Volume: {total_missed_tumors_volume:.2f} mm^3 ({final_missed_volume_percent:.2f}%)\n"
# )
#
# print(overall_stats)
# with open(stats_file, 'a') as f:
#     f.write(overall_stats)
/mnt/sda1/Research/EMA4MICCAI-2025-MIP-Based-Tumor-Segmentation/Codes/create_dataset
import os
import argparse
import numpy as np
import nibabel as nib
import monai
from tqdm import tqdm
from pathlib import Path
from pet_ct.secondary_utils.AffineMatrices import affine_matrices
from create_screened_mips import create_screened_mips


def main(volume_threshold: float, threshold: float, split_tumors: bool,
         filter_by_contrast: bool, filter_by_gradient: bool, num_of_mips: int,
         start_angle: float, end_angle: float, input_path: str, output_path: str):

    # Compute project_dir (go up 3 levels like in the other script)
    project_dir = Path(__file__).resolve().parents[2]

    input_path = Path(input_path)
    output_path = Path(output_path) / f"MIPs{num_of_mips}_{threshold}th_{volume_threshold}vth_{start_angle}_{end_angle}"

    affine = affine_matrices['SRA']
    angles = np.linspace(start=start_angle, stop=end_angle - end_angle / num_of_mips, num=num_of_mips)

    load_nifti = monai.transforms.Compose([
        monai.transforms.LoadImage(image_only=True, ensure_channel_first=True),
        monai.transforms.Orientation(axcodes='ILP')
    ])

    trials = sorted(input_path.glob('*/*'))
    print(f"Loaded {len(trials)} patient paths from {input_path}...")

    total_tumors = 0
    total_missed_tumors = 0
    total_tumors_volume = 0
    total_missed_tumors_volume = 0

    stats_file = output_path / 'tumor_statistics.txt'
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_file, 'w') as f:
        f.write("Tumor Statistics\n")
        f.write("================\n")

    for trial_path in tqdm(trials, desc='Processing cases'):
        paths = {
            'SUV': trial_path / 'SUV.nii.gz',
            'SEG': trial_path / 'SEG.nii.gz',
            'HGUO': trial_path / 'HGUO.nii.gz'
        }

        # Load volumes
        SUV = load_nifti(paths['SUV'])[0]
        SEG = load_nifti(paths['SEG'])[0]
        HGUO = load_nifti(paths['HGUO'])[0] if paths['HGUO'].exists() else None

        mips_dict = create_screened_mips(
            suv=SUV,
            seg=SEG,
            hguo=HGUO,
            horizontal_rot_angles=angles,
            threshold=threshold,
            volume_threshold=volume_threshold,
            split_tumors=split_tumors,
            filter_split_tumors_by_contrast=filter_by_contrast,
            filter_split_tumors_by_gradient=filter_by_gradient
        )

        suv_mips = mips_dict['suv_mips']
        suv_inds_mips = mips_dict['suv_inds_mips']
        seg_mips = mips_dict['seg_mips']
        hguo_mips = mips_dict.get('hguo_mips', None)
        missed_tumors = mips_dict['missed_tumors']
        tumor_volumes = mips_dict['tumor_volumes']

        total_tumors += len(tumor_volumes)
        total_missed_tumors += len(missed_tumors)
        if len(tumor_volumes) > 0:
            total_tumors_volume += float(sum(tumor_volumes.to('cpu')))
            total_missed_tumors_volume += float(sum(tumor_volumes[list(map(lambda x: x - 1, missed_tumors))]))

        # Output folder per case
        out_dir = output_path / trial_path.parent.name / trial_path.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        nib.save(nib.Nifti1Image(suv_mips.cpu().numpy(), affine), out_dir / 'SUV.nii.gz')
        nib.save(nib.Nifti1Image(suv_inds_mips.cpu().numpy(), affine), out_dir / 'SUV_inds.nii.gz')
        nib.save(nib.Nifti1Image(seg_mips.cpu().numpy(), affine), out_dir / 'SEG.nii.gz')

        if hguo_mips is not None:
            nib.save(nib.Nifti1Image(hguo_mips.cpu().numpy(), affine), out_dir / 'HGUO.nii.gz')

        missed_tumors_percent = (total_missed_tumors / total_tumors * 100) if total_tumors > 0 else 0
        missed_volume_percent = (total_missed_tumors_volume / total_tumors_volume * 100) if total_tumors_volume > 0 else 0

        with open(stats_file, 'a') as f:
            f.write(f"Case: {trial_path.name}\n")
            f.write(f"  Total Tumors: {total_tumors}\n")
            f.write(f"  Missed Tumors: {total_missed_tumors} ({missed_tumors_percent:.2f}%)\n")
            f.write(f"  Total Tumor Volume: {total_tumors_volume:.2f} mm^3\n")
            f.write(f"  Missed Tumor Volume: {total_missed_tumors_volume:.2f} mm^3 ({missed_volume_percent:.2f}%)\n\n")

    final_missed_tumors_percent = (total_missed_tumors / total_tumors * 100) if total_tumors > 0 else 0
    final_missed_volume_percent = (total_missed_tumors_volume / total_tumors_volume * 100) if total_tumors_volume > 0 else 0

    overall_stats = (
        f"Overall Statistics\n"
        f"==================\n"
        f"Total Tumors: {total_tumors}\n"
        f"Missed Tumors: {total_missed_tumors} ({final_missed_tumors_percent:.2f}%)\n"
        f"Total Tumor Volume: {total_tumors_volume:.2f} mm^3\n"
        f"Missed Tumor Volume: {total_missed_tumors_volume:.2f} mm^3 ({final_missed_volume_percent:.2f}%)\n"
    )

    print(overall_stats)
    with open(stats_file, 'a') as f:
        f.write(overall_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate screened MIPs from SUV/SEG data and compute tumor statistics")
    parser.add_argument("--volume_threshold", type=float, default=25, help="Minimum tumor volume to consider (mm^3)")
    parser.add_argument("--threshold", type=float, default=75, help="SUV threshold for tumor screening")
    parser.add_argument("--split_tumors", action="store_true", help="Split tumors into separate connected components")
    parser.add_argument("--filter_by_contrast", action="store_true", help="Filter split tumors by contrast")
    parser.add_argument("--filter_by_gradient", action="store_true", help="Filter split tumors by gradient")
    parser.add_argument("--num_of_mips", type=int, default=16, help="Number of MIPs to generate")
    parser.add_argument("--start_angle", type=float, default=0, help="Start angle for MIP generation")
    parser.add_argument("--end_angle", type=float, default=180, help="End angle for MIP generation")
    parser.add_argument("--input_path", type=str, required=True, help="Input nifti dataset path")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for saving MIPs and stats")

    args = parser.parse_args()

    main(
        volume_threshold=args.volume_threshold,
        threshold=args.threshold,
        split_tumors=args.split_tumors,
        filter_by_contrast=args.filter_by_contrast,
        filter_by_gradient=args.filter_by_gradient,
        num_of_mips=args.num_of_mips,
        start_angle=args.start_angle,
        end_angle=args.end_angle,
        input_path=args.input_path,
        output_path=args.output_path
    )