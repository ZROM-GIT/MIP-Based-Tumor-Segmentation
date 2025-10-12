import csv

# === User input ===
csv_file_path = "gpu_power_usage_ASmips_fold5.csv"
epoch_duration_minutes = float(input("Enter epoch duration in minutes: "))

# === Read data ===
power_watts = []

with open(csv_file_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            mw = int(row["power_usage_mW"])
            power_watts.append(mw / 1000.0)  # convert to watts
        except:
            continue  # skip malformed lines

# === Compute average power ===
if not power_watts:
    print("No valid power readings found.")
    exit()

mean_power_w = sum(power_watts) / len(power_watts)

# === Compute energy per epoch ===
energy_per_epoch_watt_min = mean_power_w * epoch_duration_minutes
energy_per_epoch_watt_hr = energy_per_epoch_watt_min / 60.0
energy_per_epoch_kwh = energy_per_epoch_watt_hr / 1000.0

# === Output ===
print(f"Average Power Usage: {mean_power_w:.2f} Watts")
print(f"Energy per Epoch: {energy_per_epoch_watt_min:.2f} Watt-minutes")
print(f"Energy per Epoch: {energy_per_epoch_watt_hr:.4f} Watt-hours")
print(f"Energy per Epoch: {energy_per_epoch_kwh:.6f} kilowatt-hours (kWh)")
