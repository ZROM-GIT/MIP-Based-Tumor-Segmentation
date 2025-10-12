import os
import nibabel as nib
import json
from tqdm import tqdm

# CONFIG
original_dict_path = "/mnt/sda1/PET/json_datasets/patients_PET_data.json"
e2t_base = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D"
output_dict_path = "E2T_dictionary.json"

# Load original dictionary
with open(original_dict_path, "r") as f:
    original_dict = json.load(f)

new_dict = {}
removed_cases = []

for case_key, info in tqdm(original_dict.items()):
    subj_id = info["Subject_ID"]
    trial = info["trial_name"]
    diagnosis = info.get("diagnosis", "").upper()
    seg_path = os.path.join(e2t_base, subj_id, trial, "SEG.nii.gz")
    suv_path = os.path.join(e2t_base, subj_id, trial, "SUV.nii.gz")

    # Skip if essential file is missing
    if not os.path.exists(suv_path):
        continue

    try:
        seg_lost = False
        if diagnosis != "NEGATIVE" and os.path.exists(seg_path):
            seg = nib.load(seg_path).get_fdata().astype(int)
            if not (seg == 1).any():
                seg_lost = True

        if seg_lost:
            removed_cases.append(case_key)
            continue

        # Load SUV to determine number of slices
        suv = nib.load(suv_path).get_fdata()
        num_slices = suv.shape[2]  # Z-axis

        # Update dictionary entry
        new_dict[case_key] = dict(info)
        new_dict[case_key]["num_of_slices"] = num_slices

    except Exception as e:
        print(f"❌ Failed for {case_key}: {e}")

# Save new dictionary
with open(output_dict_path, "w") as f:
    json.dump(new_dict, f, indent=2)

print(f"✅ New E2T dictionary saved to: {output_dict_path}")
print(f"🧹 Removed {len(removed_cases)} cases due to lost diagnosis")