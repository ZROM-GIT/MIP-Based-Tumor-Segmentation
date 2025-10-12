import os
import nibabel as nib
import json
from tqdm import tqdm

# CONFIG
output_base = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D"
original_dict_path = "/mnt/sda1/PET/json_datasets/patients_PET_data.json"  # Optional
output_json = "E2T_dict.json"

# Load original dictionary if exists
if os.path.exists(original_dict_path):
    with open(original_dict_path, "r") as f:
        original_dict = json.load(f)
else:
    original_dict = {}

new_dict = {}

# Traverse new output structure
for patient in tqdm(os.listdir(output_base)):
    patient_path = os.path.join(output_base, patient)
    if not os.path.isdir(patient_path):
        continue

    for case in os.listdir(patient_path):
        case_path = os.path.join(patient_path, case)
        suv_path = os.path.join(case_path, "SUV.nii.gz")

        if not os.path.exists(suv_path):
            continue

        try:
            suv = nib.load(suv_path)
            dims = list(suv.shape)
            num_of_slices = dims[2]
        except Exception as e:
            print(f"❌ Failed to read {suv_path}: {e}")
            continue

        full_key = f"{patient}/{case}"
        entry = {
            "full_path": case_path,
            "Subject_ID": patient,
            "trial_name": case,
            "dim": dims,
            "num_of_slices": num_of_slices,
        }

        # Try to keep additional info from original dict
        if full_key in original_dict:
            for field in ["diagnosis", "date"]:
                if field in original_dict[full_key]:
                    entry[field] = original_dict[full_key][field]

        new_dict[full_key] = entry

# Save updated dictionary
with open(output_json, "w") as f:
    json.dump(new_dict, f, indent=4)

print(f"\n✅ Saved updated dictionary with {len(new_dict)} cases to {output_json}")
