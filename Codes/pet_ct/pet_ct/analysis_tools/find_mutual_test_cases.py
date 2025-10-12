import json
from collections import Counter
import re

# === Manually set your paths here ===
json1_path = "/mnt/sda1/Research/PET_CT_TS/Configs/json_datasets/num_of_mips_comparison/MIPs48/MIPs48_75th_25vth_IncSplit_0_180_fold1.json"
json2_path = "/mnt/sda1/Research/PET_CT_TS/Configs/json_datasets/E2T/3D/fold_1.json"
diagnosis_dict_path = "/mnt/sda1/PET/json_datasets/patients_PET_data.json"

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_patient_case_ids(json_data):
    """
    Extract 'PETCT_<alphanumeric>/CASE_ID' from any value in each item of the 'test' list.
    """
    patient_case_ids = set()
    for item in json_data['test']:
        found = False
        for value in item.values():
            if isinstance(value, str):
                match = re.search(r'(PETCT_[A-Za-z0-9]+/[^/\\]+)', value)
                if match:
                    patient_case_ids.add(match.group(1))
                    found = True
                    break
        if not found:
            print(f"[Warning] Could not extract patient/case from: {item}")
    return patient_case_ids

def load_diagnosis_dict(dict_path):
    with open(dict_path, 'r') as f:
        return json.load(f)

# === Main logic ===

# Load both split JSONs
json1 = load_json(json1_path)
json2 = load_json(json2_path)

# Extract test sets
test_set_1 = extract_patient_case_ids(json1)
test_set_2 = extract_patient_case_ids(json2)

# Find shared instances
shared_instances = sorted(test_set_1.intersection(test_set_2))
print(f"\n✅ Found {len(shared_instances)} shared test instances.\n")

# Load diagnosis dictionary
diagnosis_dict = load_diagnosis_dict(diagnosis_dict_path)

# Collect diagnosis statistics
diagnoses = [diagnosis_dict[instance]['diagnosis'] for instance in shared_instances if instance in diagnosis_dict]
missing = [instance for instance in shared_instances if instance not in diagnosis_dict]
diagnosis_counts = Counter(diagnoses)

# Print results
print("📋 Shared test instances:")
for inst in shared_instances:
    print(f"  {inst}")

print("\n📊 Diagnosis statistics:")
for diagnosis, count in diagnosis_counts.items():
    print(f"  {diagnosis}: {count}")

if missing:
    print(f"\n⚠️ {len(missing)} instance(s) were not found in the diagnosis dictionary:")
    for m in missing:
        print(f"  {m}")
