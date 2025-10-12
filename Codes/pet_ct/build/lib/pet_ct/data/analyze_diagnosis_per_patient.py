import json
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

def analyze_cases(data, save_csv=True, output_dir="."):
    subject_case_counts = defaultdict(int)
    subject_diagnoses = defaultdict(set)

    # Populate data structures
    for entry in data.values():
        subject_id = entry["Subject_ID"]
        diagnosis = entry["diagnosis"]

        subject_case_counts[subject_id] += 1
        subject_diagnoses[subject_id].add(diagnosis)

    # Convert to DataFrame
    df = pd.DataFrame({
        "Subject_ID": list(subject_case_counts.keys()),
        "Num_Cases": list(subject_case_counts.values()),
        "Diagnoses": [list(subject_diagnoses[s]) for s in subject_case_counts.keys()],
        "Num_Diagnoses": [len(subject_diagnoses[s]) for s in subject_case_counts.keys()]
    })

    # Patients with >1 case
    multi_case_df = df[df["Num_Cases"] > 1].copy()

    # Patients with multiple diagnoses
    multi_diag_df = df[df["Num_Diagnoses"] > 1].copy()

    print(f"Total unique patients: {len(df)}")
    print(f"Patients with >1 case: {len(multi_case_df)}")
    print(f"Patients with >1 diagnosis: {len(multi_diag_df)}")

    # Print number of patients per case count (e.g., how many have 1, 2, 3 cases, etc.)
    print("\nNumber of patients by number of cases:")
    case_distribution = df["Num_Cases"].value_counts().sort_index()
    for num_cases, num_patients in case_distribution.items():
        print(f"{num_cases} case(s): {num_patients} patient(s)")

    # Count diagnosis occurrences *only for cases* belonging to patients with 2+ cases
    print("\nDiagnosis occurrence counts for patients with 2+ cases:")

    diagnosis_counter = defaultdict(int)

    # Build a list of all entries (i.e., cases) from patients with 2+ cases
    for key, entry in data.items():
        subject_id = entry["Subject_ID"]
        if subject_case_counts[subject_id] >= 2:
            diagnosis = entry["diagnosis"]
            diagnosis_counter[diagnosis] += 1

    # Print sorted by frequency
    for diagnosis, count in sorted(diagnosis_counter.items(), key=lambda x: -x[1]):
        print(f"{diagnosis}: {count} case(s)")

    # Detailed diagnosis occurrence by number of cases per patient
    print("\nDiagnosis breakdown for each patient group (by number of cases):")

    # First, group Subject_IDs by their Num_Cases
    subject_ids_by_case_count = defaultdict(set)
    for subject_id, num_cases in subject_case_counts.items():
        subject_ids_by_case_count[num_cases].add(subject_id)

    # For each group, count diagnosis occurrences
    for case_count in sorted(subject_ids_by_case_count.keys()):
        diagnosis_counter = defaultdict(int)
        subject_ids = subject_ids_by_case_count[case_count]

        for entry in data.values():
            if entry["Subject_ID"] in subject_ids:
                diagnosis_counter[entry["diagnosis"]] += 1

        print(f"\n--- Diagnosis counts for patients with {case_count} case(s) ---")
        for diagnosis, count in sorted(diagnosis_counter.items(), key=lambda x: -x[1]):
            print(f"{diagnosis}: {count} case(s)")


    # Plot histogram of number of cases
    plt.figure(figsize=(8, 5))
    df["Num_Cases"].hist(bins=range(1, df["Num_Cases"].max() + 2))
    plt.xlabel("Number of Cases per Subject")
    plt.ylabel("Number of Subjects")
    plt.title("Distribution of Cases per Subject")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cases_histogram.png")
    plt.show()

    # Bar chart: Number of subjects per number of unique diagnoses
    plt.figure(figsize=(6, 4))
    diagnosis_counts = df["Num_Diagnoses"].value_counts().sort_index()
    plt.bar(diagnosis_counts.index, diagnosis_counts.values)

    plt.xlabel("Number of Unique Diagnoses")
    plt.ylabel("Number of Subjects")
    plt.title("Distribution of Diagnoses per Subject")
    plt.xticks(diagnosis_counts.index)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/diagnosis_bar_chart.png")
    plt.show()

    # Optionally save CSVs
    if save_csv:
        multi_case_df.to_csv(f"{output_dir}/subjects_with_multiple_cases.csv", index=False)
        multi_diag_df.to_csv(f"{output_dir}/subjects_with_multiple_diagnoses.csv", index=False)

    return df, multi_case_df, multi_diag_df


if __name__ == "__main__":
    json_path = './E2T_dictionary.json'  # 👈 Replace with your JSON file path
    output_dir = "."  # 👈 Replace with your desired output folder if needed

    with open(json_path, "r") as f:
        data = json.load(f)

    analyze_cases(data, save_csv=True, output_dir=output_dir)

