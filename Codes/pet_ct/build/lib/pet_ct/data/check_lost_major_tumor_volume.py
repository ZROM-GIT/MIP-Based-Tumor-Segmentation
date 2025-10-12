import json
import os
import nibabel as nib
import numpy as np
import pandas as pd

def load_seg_volume(seg_path):
    """Load a NIfTI segmentation and return the number of non-zero voxels."""
    if not os.path.exists(seg_path):
        return None
    seg = nib.load(seg_path)
    data = seg.get_fdata()
    return np.count_nonzero(data)

def compare_seg_volumes(case_dict, old_root, new_root, seg_filename="SEG.nii.gz", threshold=0.5):
    flagged_cases = []

    for case_id, case_data in case_dict.items():
        if case_data["diagnosis"].upper() == "NEGATIVE":
            continue  # skip NEGATIVE cases

        rel_path = case_data["full_path"].split("niftis/")[-1]
        old_seg = os.path.join(old_root, rel_path, seg_filename)
        new_seg = os.path.join(new_root, rel_path, seg_filename)

        old_volume = load_seg_volume(old_seg)
        new_volume = load_seg_volume(new_seg)

        if old_volume is None or new_volume is None:
            print(f"Missing SEG file for: {case_id}")
            continue

        if old_volume == 0:
            print(f"Original SEG volume is 0 in: {case_id}")
            continue

        ratio = new_volume / old_volume

        if ratio <= threshold:
            flagged_cases.append({
                "Case_ID": case_id,
                "Subject_ID": case_data["Subject_ID"],
                "Trial_Name": case_data["trial_name"],
                "Diagnosis": case_data["diagnosis"],
                "Old_Volume": old_volume,
                "New_Volume": new_volume,
                "Retained_Ratio": round(ratio, 3)
            })

    df_flagged = pd.DataFrame(flagged_cases)
    return df_flagged

if __name__ == "__main__":
    json_path = "./E2T_dictionary.json"  # 👈 replace with your path
    old_root = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis"  # 👈 replace with real old root
    new_root = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D"   # 👈 replace with real new root

    with open(json_path, "r") as f:
        case_dict = json.load(f)

    flagged_df = compare_seg_volumes(case_dict, old_root, new_root)

    # Save results
    if not flagged_df.empty:
        flagged_df.to_csv("flagged_cases_seg_volume_drop.csv", index=False)
        print(f"\nFlagged {len(flagged_df)} case(s). Saved to 'flagged_cases_seg_volume_drop.csv'")
    else:
        print("\nNo cases had SEG volume drop below threshold.")
