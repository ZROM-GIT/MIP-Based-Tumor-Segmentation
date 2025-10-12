import os
import json
import random
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def split_dataset(cases_dict, test_ratio=0.15, seed=42):
    rng = random.Random(seed)

    # Step 1: Group cases by patient
    patients = defaultdict(list)
    for case_id in sorted(cases_dict):  # Sorted for determinism
        case_data = cases_dict[case_id]
        subject_id = case_data["Subject_ID"]
        diagnosis = case_data["diagnosis"]
        patients[subject_id].append((case_id, diagnosis))

    test_cases = set()
    train_val_cases = set()
    eligible_patient_groups = []  # (subject_id, [case_ids], diagnosis)
    backup_cases = set()

    for subject_id in sorted(patients):  # Deterministic order
        cases = patients[subject_id]
        case_ids = [cid for cid, _ in cases]
        diagnoses = {diag for _, diag in cases}

        if len(diagnoses) == 1:
            eligible_patient_groups.append((subject_id, case_ids, next(iter(diagnoses))))
        else:
            neg_cases = [cid for cid, diag in cases if diag == "NEGATIVE"]
            pos_cases = [cid for cid, diag in cases if diag != "NEGATIVE"]

            if rng.random() < 0.5:
                test_cases.update(pos_cases)
                train_val_cases.update(neg_cases)
            else:
                test_cases.update(neg_cases)
                train_val_cases.update(pos_cases)

            for cid in neg_cases + pos_cases:
                subj_cases = set(neg_cases + pos_cases)
                if cid in test_cases and (subj_cases - {cid}) & train_val_cases:
                    backup_cases.add(cid)

    eligible_patient_groups = sorted(eligible_patient_groups, key=lambda x: x[0])

    if eligible_patient_groups:
        indices = list(range(len(eligible_patient_groups)))
        group_labels = [diag for _, _, diag in eligible_patient_groups]

        total_test_needed = int(len(cases_dict) * test_ratio)
        test_group_count = max(min(total_test_needed - len(test_cases), len(indices)), 0)

        if test_group_count > 0:
            train_idx, test_idx = train_test_split(
                indices, test_size=test_group_count, stratify=group_labels, random_state=seed
            )
            for i in sorted(train_idx):
                train_val_cases.update(eligible_patient_groups[i][1])
            for i in sorted(test_idx):
                test_cases.update(eligible_patient_groups[i][1])
        else:
            for _, cids, _ in eligible_patient_groups:
                train_val_cases.update(cids)

    test_cases -= train_val_cases
    train_val_cases -= test_cases

    test_set = {cid: cases_dict[cid] for cid in sorted(test_cases)}
    train_val_set = {cid: cases_dict[cid] for cid in sorted(train_val_cases)}
    backup_set = {cid: cases_dict[cid] for cid in sorted(backup_cases)}

    return train_val_set, test_set, backup_set

def print_diagnosis_stats(name, dataset):
    diag_counter = Counter(case_data["diagnosis"] for case_data in dataset.values())
    print(f"\nDiagnosis breakdown for {name}:")
    for diag, count in sorted(diag_counter.items(), key=lambda x: -x[1]):
        print(f"  {diag:20} : {count}")
    print(f"  Total: {sum(diag_counter.values())}")

def validate_backup_in_test(backup_set, test_set):
    missing = set(backup_set) - set(test_set)
    if missing:
        print(f"\u274c ERROR: {len(missing)} backup cases are not in the test set!")
    else:
        print("\u2705 All backup cases are also present in the test set.")

def print_split_patients_info(backup_set, train_val_set):
    train_val_subjects = {case["Subject_ID"] for case in train_val_set.values()}
    split_patients = {
        case["Subject_ID"] for case in backup_set.values() if case["Subject_ID"] in train_val_subjects
    }
    print(f"\U0001f50d Patients with cases split across test/train: {len(split_patients)}")

def cross_validate(train_val_set, n_splits=5, seed=42):
    print(f"\n===== Performing {n_splits}-fold Cross-Validation on Train/Val Set =====")
    case_ids = sorted(train_val_set.keys())  # Deterministic order
    diagnoses = [train_val_set[cid]["diagnosis"] for cid in case_ids]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for i, (train_idx, val_idx) in enumerate(skf.split(case_ids, diagnoses), 1):
        train_ids = [case_ids[j] for j in sorted(train_idx)]
        val_ids = [case_ids[j] for j in sorted(val_idx)]

        train_fold = {cid: train_val_set[cid] for cid in train_ids}
        val_fold = {cid: train_val_set[cid] for cid in val_ids}
        folds.append((train_fold, val_fold))

        train_counter = Counter(train_val_set[cid]["diagnosis"] for cid in train_ids)
        val_counter = Counter(train_val_set[cid]["diagnosis"] for cid in val_ids)
        all_diags = sorted(set(train_counter) | set(val_counter), key=lambda x: -train_counter[x] - val_counter[x])

        print(f"\n\U0001f4c2 Fold {i}")
        print(f"  Train size: {len(train_fold)}")
        print(f"  Val size:   {len(val_fold)}")
        print(f"  Diagnosis breakdown:")
        for diag in all_diags:
            print(f"    {diag:20} | Train = {train_counter.get(diag, 0):4d}  Val = {val_counter.get(diag, 0):4d}")

    return folds

def create_monai_json(
    output_dir: str,
    folds: list,
    test_set: dict,
    backup_set: dict,
    key_configs: dict,
    base_path: str,
):
    os.makedirs(output_dir, exist_ok=True)

    def convert_case(case_id):
        case_data = cases_dict[case_id]
        patient_id = case_data.get("Subject_ID", "UNKNOWN")
        return {
            **{
                key: os.path.join(base_path, case_id, suffix)
                for key, suffix in sorted(key_configs.items())
            },
            "CaseID": case_id,
        }

    test_list = [convert_case(cid) for cid in sorted(test_set)]
    backup_list = [convert_case(cid) for cid in sorted(backup_set)]

    with open(os.path.join(output_dir, "backup.json"), "w") as f:
        json.dump({"test": backup_list}, f, indent=2)

    for i, (train, val) in enumerate(folds, 1):
        train_list = [convert_case(cid) for cid in sorted(train)]
        val_list = [convert_case(cid) for cid in sorted(val)]

        json_dict = {
            "training": train_list,
            "validation": val_list,
            "test": test_list,
        }

        with open(os.path.join(output_dir, f"fold_{i}.json"), "w") as f:
            json.dump(json_dict, f, indent=2)

    print(f"\n\U0001f4e6 Saved {len(folds)} folds and backup JSON to: {output_dir}")

if __name__ == "__main__":
    input_json_path = "./E2T_dictionary.json"

    with open(input_json_path, "r") as f:
        cases_dict = json.load(f)

    train_val_set, test_set, backup_set = split_dataset(cases_dict, test_ratio=0.15, seed=42)

    print("\n===== SPLIT SUMMARY =====")
    print(f"Total cases:        {len(cases_dict)}")
    print(f"Train/Val cases:    {len(train_val_set)}")
    print(f"Test cases:         {len(test_set)}")
    print(f"Backup (conflicts): {len(backup_set)}")

    print_diagnosis_stats("Train/Val Set", train_val_set)
    print_diagnosis_stats("Test Set", test_set)
    print_diagnosis_stats("Backup Set", backup_set)

    validate_backup_in_test(backup_set, test_set)
    print_split_patients_info(backup_set, train_val_set)

    cv_folds = cross_validate(train_val_set, n_splits=5, seed=42)

    output_dir = "/mnt/sda1/Research/PET_CT_TS/Configs/json_datasets/E2T"
    base_path = "Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D"

    key_configs = {
        "SUV_3D": "SUV.nii.gz",
        "PET_3D": "PET.nii.gz",
        "CT_3D": "CTres.nii.gz",
        "SEG_3D": "SEG.nii.gz",
        "HGUO_3D": "HGUO.nii.gz",
    }

    create_monai_json(
        output_dir=output_dir,
        folds=cv_folds,
        test_set=test_set,
        backup_set=backup_set,
        key_configs=key_configs,
        base_path=base_path,
    )
