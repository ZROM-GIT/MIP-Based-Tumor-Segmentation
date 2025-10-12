import os
import shutil

# === Manually set paths ===
shared_instances_txt = "shared_instances.txt"  # List of patient-case IDs, one per line
source_root = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/E2T_MIPs/MIPs48"         # Where to copy from
destination_root = "/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/Liran_Tests/E2T" # Where to copy to

def read_shared_instances(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def copy_instance_folders(instances, src_root, dst_root):
    missing = []
    for instance in instances:
        src_path = os.path.join(src_root, instance)
        dst_path = os.path.join(dst_root, instance)

        if not os.path.exists(src_path):
            missing.append(instance)
            print(f"[Missing] Source not found: {src_path}")
            continue

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        try:
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            print(f"[Copied] {instance}")
        except Exception as e:
            print(f"[Error] Could not copy {instance}: {e}")

    print(f"\n✅ Done copying. {len(instances) - len(missing)} copied, {len(missing)} missing.")
    if missing:
        print("Missing instances:")
        for m in missing:
            print(f"  {m}")

# === Main ===
if __name__ == "__main__":
    shared_instances = read_shared_instances(shared_instances_txt)
    copy_instance_folders(shared_instances, source_root, destination_root)
