import pandas as pd

# Replace with your CSV file path
csv_path = "/mnt/sda1/Research/PET_CT_TS/system_performance/classification/volumetric_fold1/wandb_export_2025-07-02T04_46_20.105+03_00.csv"
average_epoch_time_seconds = 1564.492 # <-- Change this to your actual average epoch time

# Load the CSV
df = pd.read_csv(csv_path)

# Check columns
columns = df.columns.tolist()
print("Columns in CSV:", columns)

# Find the GPU power and time columns — adjust names if needed
# Example column names you might have:
power_col = 'PET1_classifying_volumes_fold1 - system/gpu.0.powerWatts'  # exact column name from your CSV
time_col_candidates = ['Relative Time (Process)'] # common guesses

# Pick a time column that exists
time_col = None
for candidate in time_col_candidates:
    if candidate in df.columns:
        time_col = candidate
        break

if time_col is None:
    raise ValueError("No suitable time column found in CSV!")

print(f"Using time column: {time_col}")
print(f"Using power column: {power_col}")

# Convert time to seconds if needed
# If it's a datetime, convert to seconds elapsed from start
if pd.api.types.is_datetime64_any_dtype(df[time_col]):
    df[time_col] = pd.to_datetime(df[time_col])
    df[time_col] = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds()
else:
    # Try to guess if time is in minutes or seconds
    max_time = df[time_col].max()
    if max_time < 10000:  # Probably seconds or minutes
        # if minutes, convert to seconds
        if max_time < 1000:
            df[time_col] = df[time_col] * 60

# Sort by time just in case
df = df.sort_values(time_col)

# Compute delta time between samples
df['delta_t'] = df[time_col].diff().fillna(0)

# Compute energy = power * delta time (W * s = Joules)
df['energy_J'] = df[power_col] * df['delta_t']

total_energy_joules = df['energy_J'].sum()
total_energy_kwh = total_energy_joules / (1000 * 3600)
total_time = df[time_col].iloc[-1] - df[time_col].iloc[0]
average_power = total_energy_joules / total_time if total_time > 0 else 0

print("\nSanity check — delta_t stats:")
print(df['delta_t'].describe())

print(f"\nGPU Energy Consumption Summary:")
print(f"Total duration    : {total_time:.2f} seconds")
print(f"Total energy      : {total_energy_joules:.2f} Joules")
print(f"                  : {total_energy_kwh:.6f} kWh")
print(f"Average GPU power : {average_power:.2f} W")

# ------------------------------------------
# ⚡ Energy per epoch calculation (add here)
# ------------------------------------------

# User input: average epoch time in seconds

# Energy spent per epoch
energy_per_epoch_joules = average_power * average_epoch_time_seconds
energy_per_epoch_kwh = energy_per_epoch_joules / (1000 * 3600)

print(f"\n⚡ Energy per epoch:")
print(f"{energy_per_epoch_joules:.2f} J ({energy_per_epoch_kwh:.6f} kWh) per epoch")
