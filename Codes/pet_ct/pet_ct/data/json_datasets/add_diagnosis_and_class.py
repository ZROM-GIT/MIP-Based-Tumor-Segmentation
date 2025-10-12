import json
import pickle
from pathlib import Path
from collections import OrderedDict

###############################################################################
# CONFIGURATION — set these paths for your local setup
###############################################################################
dataset_json_path = Path("/mnt/sda1/Research/PET_CT_TS/Configs/json_datasets/num_of_mips_comparison/MIPs80/MIPs80_75th_25vth_IncSplit_0_180_fold5.json")
lookup_dict_path = Path("/mnt/sda1/Research/PET_CT_TS/Configs/json_datasets/patients_PET_data.json")
output_path = dataset_json_path          # overwrite in-place; change if you prefer

PATH_KEY = "SUV_mips"                    # where to find any path to parse

LABEL_MAP = {                            # fixed class mapping
    "NEGATIVE": 0,
    "MELANOMA": 1,
    "LYMPHOMA": 2,
    "LUNG_CANCER": 3,
}
###############################################################################

def load_lookup(path: Path) -> dict:
    """Return dict {'patient/case': {'diagnosis': str}}."""
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            return pickle.load(f)
    with path.open() as f:
        return json.load(f)

def extract_patient_case(path_str: str) -> str:
    """
    Extract '<patient>/<case>' from a path like
    .../<patient>/<case>/some_file.nii.gz
    """
    parts = Path(path_str).parts
    if len(parts) < 3:
        raise ValueError(f"Cannot parse patient/case from: {path_str}")
    return f"{parts[-3]}/{parts[-2]}"

def patch_split(split_items, lookup, label_map):
    for item in split_items:
        source_path = item.get(PATH_KEY)
        if source_path is None:
            raise KeyError(f"Missing '{PATH_KEY}' key in:\n{json.dumps(item, indent=2)}")

        full_case = extract_patient_case(source_path)           # ← NEW
        if full_case not in lookup:
            raise KeyError(f"'{full_case}' not in diagnosis lookup.")

        diagnosis = lookup[full_case]["diagnosis"]
        item["diagnosis"] = diagnosis
        item["class_number"] = label_map[diagnosis]
        item["full_case_name"] = full_case                      # ← NEW

# === Main logic ===
lookup = load_lookup(lookup_dict_path)

with dataset_json_path.open() as f:
    dataset = json.load(f, object_pairs_hook=OrderedDict)

for split_name, items in dataset.items():
    if isinstance(items, list):        # skip any non-sample blocks
        patch_split(items, lookup, LABEL_MAP)

with output_path.open("w") as f:
    json.dump(dataset, f, indent=4)

print(f"✓ Wrote updated dataset to {output_path}")
