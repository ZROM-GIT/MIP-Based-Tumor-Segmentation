import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')

# CONFIG
png_dir = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_png"  # <-- update this path
output_csv = "flagged_cases_new.csv"

# Load PNGs
images = sorted([f for f in os.listdir(png_dir) if f.endswith(".png")])
if not images:
    print("No PNG images found.")
    exit()

# State
current_index = 0
flagged = set()

# GUI Setup
fig, ax = plt.subplots(figsize=(8, 8))
fig.canvas.manager.set_window_title("MIP QA Tool")

def update_image():
    ax.clear()
    img_path = os.path.join(png_dir, images[current_index])
    img = mpimg.imread(img_path)
    ax.imshow(img, cmap='gray')
    ax.axis('off')

    is_flagged = images[current_index] in flagged
    flag_text = "⚠️ FLAGGED" if is_flagged else "✓ OK"
    flagged_count = len(flagged)

    ax.set_title(f"[{current_index + 1}/{len(images)}] {images[current_index]} - {flag_text} | {flagged_count} flagged")
    fig.canvas.draw_idle()

def on_key(event):
    global current_index
    if event.key == 'right':
        if current_index < len(images) - 1:
            current_index += 1
            update_image()
    elif event.key == 'left':
        if current_index > 0:
            current_index -= 1
            update_image()
    elif event.key == ' ':
        filename = images[current_index]
        if filename in flagged:
            flagged.remove(filename)
        else:
            flagged.add(filename)
        update_image()
    elif event.key == 'q':
        print("Saving flagged cases...")
        case_ids = [f.replace(".png", "") for f in flagged]
        df = pd.DataFrame(case_ids, columns=["case_id"])
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved {len(case_ids)} flagged cases to {output_csv}")
        plt.close(fig)

# Key bindings
fig.canvas.mpl_connect('key_press_event', on_key)

# Start
update_image()
plt.show()