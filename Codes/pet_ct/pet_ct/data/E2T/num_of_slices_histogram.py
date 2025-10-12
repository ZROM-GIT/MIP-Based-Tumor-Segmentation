import json
import matplotlib.pyplot as plt
import numpy as np

def plot_num_of_slices_hist(data, output_path="num_of_slices_histogram.png", bin_width=25, histogram_bins=50):
    # Extract slice counts
    num_slices = [entry["num_of_slices"] for entry in data.values()]
    min_slices, max_slices = min(num_slices), max(num_slices)

    # Print group counts in steps of 25
    print("\nNumber of cases in slice count groups (width = 25):")
    group_edges = range(min_slices // 25 * 25, max_slices + 25, 25)
    group_counts = {f"{start}-{start + 24}": 0 for start in group_edges}

    for n in num_slices:
        bucket_start = (n // 25) * 25
        bucket_label = f"{bucket_start}-{bucket_start + 24}"
        if bucket_label in group_counts:
            group_counts[bucket_label] += 1

    for label, count in group_counts.items():
        print(f"{label}: {count} case(s)")

    # Plot histogram with more bins
    plt.figure(figsize=(10, 6))
    plt.hist(num_slices, bins=histogram_bins, edgecolor='black')
    plt.xlabel("Number of Slices")
    plt.ylabel("Number of Cases")
    plt.title("Distribution of Number of Slices per Case")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    import pandas as pd  # Make sure to import this at the top if not already

    # Save cases with < 275 slices to CSV
    low_slice_cases = [
        {
            "Subject_ID": entry["Subject_ID"],
            "Trial_Name": entry["trial_name"],
            "Num_Slices": entry["num_of_slices"],
            "Diagnosis": entry["diagnosis"]
        }
        for entry in data.values() if entry["num_of_slices"] < 275
    ]

    df_low = pd.DataFrame(low_slice_cases)
    csv_output_path = output_path.replace(".png", "_low_slice_cases.csv")
    df_low.to_csv(csv_output_path, index=False)
    print(f"\nSaved {len(df_low)} cases with < 275 slices to: {csv_output_path}")


if __name__ == "__main__":
    json_path = 'E2T_dictionary.json'  # 👈 Replace with your actual JSON path

    with open(json_path, "r") as f:
        data = json.load(f)

    plot_num_of_slices_hist(data)
