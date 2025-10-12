import os
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import json

# CONFIG
e2t_base = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D"
dict_path = "/mnt/sda1/PET/json_datasets/patients_PET_data.json"  # Your original dictionary path
output_csv = "lost_diagnosis_cases.csv"

# Load the original dictionary
with open(dict_path, "r") as f:
    case_dict = json.load(f)

lost_cases = []

# Go through all cases in the dictionary
for case_key, info in tqdm(case_dict.items()):
    diagnosis = info.get("diagnosis", "").upper()
    if diagnosis == "NEGATIVE":
        continue  # We only care about cases that *had* a diagnosis

    subj_id = info["Subject_ID"]
    trial = info["trial_name"]
    seg_path = os.path.join(e2t_base, subj_id, trial, "SEG.nii.gz")

    if os.path.exists(seg_path):
        try:
            seg = nib.load(seg_path).get_fdata().astype(int)
            if not (seg == 1).any():  # SEG is completely empty
                lost_cases.append({
                    "case_id": case_key,
                    "Subject_ID": subj_id,
                    "trial_name": trial,
                    "diagnosis": diagnosis,
                    "reason": "Original diagnosis lost after cropping (SEG empty)"
                })
        except Exception as e:
            print(f"❌ Failed to read {seg_path}: {e}")

# Save result
if lost_cases:
    df = pd.DataFrame(lost_cases)
    df.to_csv(output_csv, index=False)
    print(f"\n⚠️ {len(df)} cases lost their diagnosis. Saved to: {output_csv}")
else:
    print("✅ All diagnosed cases retained their segmentation.")

# Remove lost cases from original dictionary
filtered_dict = {
    k: v for k, v in case_dict.items()
    if k not in {case['case_id'] for case in lost_cases}
}

# Diagnosis statistics
diagnoses = [v.get("diagnosis", "UNKNOWN").strip().upper() for v in filtered_dict.values()]
diagnosis_counts = pd.Series(diagnoses).value_counts()
print("\n📊 Final Diagnosis Statistics (after removing lost diagnosis cases):")
print(diagnosis_counts)

# Optionally save the new filtered dictionary
with open("updated_patient_dict.json", "w") as f:
    json.dump(filtered_dict, f, indent=2)
