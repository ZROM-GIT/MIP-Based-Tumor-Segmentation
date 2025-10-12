import os
import shutil
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib

# Force interactive backend for GUI usability with PyQt5
matplotlib.use('Qt5Agg')

# CONFIG
csv_path = "flagged_cases_new.csv"  # CSV with PETCT_ID/caseID in column 'case_id'
input_base = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis"
output_base = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_3D"
modalities = ['SUV', 'SEG', 'CTres', 'HGUO', 'PET']

corrected = []
flagged = []

# Load case list
case_df = pd.read_csv(csv_path)
if 'case_id' not in case_df.columns:
    raise ValueError("CSV must contain a column named 'case_id'")
cases = case_df['case_id'].tolist()

current_idx = 0
crop_start, crop_end = 0, 0

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
mip_img = None
start_line = None
end_line = None

# Sliders
ax_start = plt.axes([0.15, 0.1, 0.7, 0.03])
ax_end = plt.axes([0.15, 0.05, 0.7, 0.03])
slider_start = Slider(ax_start, 'Start', 0, 500, valinit=0, valstep=1)
slider_end = Slider(ax_end, 'End', 0, 500, valinit=0, valstep=1)

# Buttons
ax_flag = plt.axes([0.81, 0.9, 0.1, 0.04])
btn_flag = Button(ax_flag, 'Flag')
ax_next = plt.axes([0.81, 0.85, 0.1, 0.04])
btn_next = Button(ax_next, 'Save & Next')

flag_current = False


def load_volume(pet_path):
    return nib.load(pet_path).get_fdata()

def create_mip(volume):
    mip = np.max(volume, axis=1)  # coronal
    # Improve contrast using 2nd and 98th percentiles
    p2, p98 = np.percentile(mip, (2, 98))
    mip = np.clip((mip - p2) / (p98 - p2 + 1e-6), 0, 1)
    return np.rot90(mip)

def update_display():
    global mip_img, start_line, end_line

    stam = cases[current_idx].split('_', 2)
    patient_id, case_id = stam[0] + '_' + stam[1], stam[2]
    case_path = os.path.join(input_base, patient_id, case_id)
    suv_path = os.path.join(case_path, "SUV.nii.gz")
    if not os.path.exists(suv_path):
        print(f"❌ Missing SUV: {suv_path}")
        return

    vol = load_volume(suv_path)
    mip = create_mip(vol)

    ax.clear()
    mip_img = ax.imshow(mip, cmap='gray')
    ax.set_title(cases[current_idx])

    slider_start.valmax = mip.shape[0] - 1
    slider_end.valmax = mip.shape[0] - 1
    slider_start.set_val(0)
    slider_end.set_val(mip.shape[0] - 1)
    slider_start.ax.set_xlim(slider_start.valmin, slider_start.valmax)
    slider_end.ax.set_xlim(slider_end.valmin, slider_end.valmax)

    start_line = ax.axhline(0, color='red', label='Start')
    end_line = ax.axhline(mip.shape[0] - 1, color='red', label='End')
    fig.canvas.draw_idle()

def update_lines(val):
    if start_line and end_line:
        start_line.set_ydata([slider_start.val, slider_start.val])
        end_line.set_ydata([slider_end.val, slider_end.val])
        fig.canvas.draw_idle()

slider_start.on_changed(update_lines)
slider_end.on_changed(update_lines)

def on_flag(event):
    global flag_current
    flag_current = True
    print(f"⚠️ Flagged: {cases[current_idx]}")
    # Also remove from output folder if it exists
    stam = cases[current_idx].split('_', 2)
    patient_id, case_id = stam[0] + '_' + stam[1], stam[2]
    out_case_path = os.path.join(output_base, patient_id, case_id)
    if os.path.exists(out_case_path):
        shutil.rmtree(out_case_path)
        print(f"🗑️ Removed folder: {out_case_path}")
    on_next(None)  # Proceed to next case immediately

def crop_and_save(case_id_str, start, end):
    stam = case_id_str.split('_', 2)
    patient_id, case_id = stam[0] + '_' + stam[1], stam[2]
    case_path = os.path.join(input_base, patient_id, case_id)
    out_case_path = os.path.join(output_base, patient_id, case_id)
    os.makedirs(out_case_path, exist_ok=True)

    for mod in modalities:
        in_path = os.path.join(case_path, f"{mod}.nii.gz")
        if not os.path.exists(in_path):
            continue
        nii = nib.load(in_path)
        data = nii.get_fdata()
        num_of_axial_slices = data.shape[2]
        crop_start, crop_end = num_of_axial_slices - end, num_of_axial_slices - start
        cropped = data[:, :, crop_start:crop_end]
        new_affine = nii.affine.copy()
        new_affine[:3, 3] += crop_start * nii.affine[:3, 2]
        cropped_img = nib.Nifti1Image(cropped, new_affine, header=nii.header)

        # Clean SEG
        if mod == 'SEG':
            import cc3d
            labels = cc3d.connected_components((cropped > 0).astype(np.uint8))
            for l in range(1, labels.max() + 1):
                if np.sum(labels == l) < 3:
                    cropped[labels == l] = 0
            cropped_img = nib.Nifti1Image(cropped, new_affine, header=nii.header)

        nib.save(cropped_img, os.path.join(out_case_path, f"{mod}.nii.gz"))

def on_next(event):
    global current_idx, flag_current
    cid = cases[current_idx]

    if flag_current:
        flagged.append(cid)
    else:
        s, e = int(slider_start.val), int(slider_end.val)
        if s >= e:
            print(f"⚠️ Invalid crop for {cid}: start >= end")
            return
        crop_and_save(cid, s, e)
        corrected.append({"case": cid, "crop_start": s, "crop_end": e, "num_slices": e - s})

    flag_current = False
    current_idx += 1
    if current_idx >= len(cases):
        finish()
    else:
        update_display()

def finish():
    print("✅ All cases reviewed. Saving logs...")
    pd.DataFrame(corrected).to_csv("corrected_crops.csv", index=False)
    pd.DataFrame(flagged, columns=["case"]).to_csv("flagged_cases.csv", index=False)
    plt.close()

btn_flag.on_clicked(on_flag)
btn_next.on_clicked(on_next)

update_display()
plt.show()
