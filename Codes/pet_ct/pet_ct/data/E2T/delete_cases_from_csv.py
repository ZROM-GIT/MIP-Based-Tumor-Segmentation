import os
import json
import csv
import shutil

def load_removal_list(csv_path):
    """Returns a set of (Subject_ID, Case_ID) to remove from the dictionary and filesystem."""
    removal_set = set()
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            subject = row["Subject_ID"].strip()
            case = row["Case_ID"].strip()
            removal_set.add((subject, case))
    return removal_set

def remove_cases_from_dict(cases_dict, removal_set):
    """Removes specified case entries from the dictionary."""
    to_remove = [
        cid for cid, data in cases_dict.items()
        if (data["Subject_ID"], cid) in removal_set
    ]
    for cid in to_remove:
        del cases_dict[cid]
    return to_remove

def delete_case_folders(base_data_dir, removed_case_ids):
    """Removes folders from disk for each removed case ID."""
    for case_id in removed_case_ids:
        case_path = os.path.join(base_data_dir, case_id)
        if os.path.exists(case_path):
            shutil.rmtree(case_path)
            print(f"🗑️ Deleted folder: {case_path}")
        else:
            print(f"⚠️ Not found (skipped): {case_path}")

def save_updated_dict(cases_dict, output_json_path):
    with open(output_json_path, "w") as f:
        json.dump(cases_dict, f, indent=2)
    print(f"✅ Saved updated dictionary to: {output_json_path}")

# ============================
#        MAIN SCRIPT
# ============================
if __name__ == "__main__":
    # === CONFIGURATION ===
    original_json_path = "E2T_dictionary.json"
    csv_to_remove_path = "flagged_cases_seg_volume_drop.csv"
    data_folder_path = "/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D"
    updated_json_path = "E2T_dictionary.json"

    # === PROCESSING ===
    with open(original_json_path, "r") as f:
        cases_dict = json.load(f)

    removal_set = load_removal_list(csv_to_remove_path)
    removed_case_ids = remove_cases_from_dict(cases_dict, removal_set)
    delete_case_folders(data_folder_path, removed_case_ids)
    save_updated_dict(cases_dict, updated_json_path)

    print(f"\n🧹 Removed {len(removed_case_ids)} cases from dictionary and disk.")
