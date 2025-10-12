from create_E2T_dataset import process_case
import os

# CONFIG
case_path = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis/PETCT_06a46414eb/03-12-2006-NA-PET-CT Ganzkoerper  primaer mit KM-38502"
base_output_dir = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D"

# Run just this case
result = process_case(case_path, base_output_dir)

if result:
    print(f"✅ Reprocessed: {result['patient']}/{result['case']}")
else:
    print("❌ Failed or skipped.")
