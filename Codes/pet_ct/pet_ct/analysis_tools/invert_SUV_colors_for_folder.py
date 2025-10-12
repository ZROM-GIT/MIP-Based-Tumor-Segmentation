import os
import SimpleITK as sitk


def invert_image_intensity(input_path):
    """Load, invert, and return a SimpleITK image."""
    image = sitk.ReadImage(input_path)
    array = sitk.GetArrayFromImage(image)
    inverted_array = array.max() - array
    inverted_image = sitk.GetImageFromArray(inverted_array)
    inverted_image.CopyInformation(image)
    return inverted_image


def process_folder(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'SUV.nii.gz':
                full_path = os.path.join(dirpath, filename)
                print(f"Inverting: {full_path}")

                inverted_image = invert_image_intensity(full_path)

                output_path = os.path.join(dirpath, 'SUV_inv.nii.gz')
                sitk.WriteImage(inverted_image, output_path)
                print(f"Saved inverted image to: {output_path}")


# Example usage
if __name__ == "__main__":
    root_directory = '/mnt/sda1/Research/PET_CT_TS/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/Liran_Tests/Original'  # Change this to your actual path
    process_folder(root_directory)
