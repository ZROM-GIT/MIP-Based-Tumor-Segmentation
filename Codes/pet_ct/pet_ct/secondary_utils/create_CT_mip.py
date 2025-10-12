import numpy as np
import nibabel as nib
import monai
import matplotlib.pyplot as plt


def create_ct_mips(ct_data, suv_inds, starting_angle=0, ending_angle=180, device='cpu'):
    """
    Creates a 2D CT image from 3D CT data using depth indices from SUV_inds.

    Parameters:
    ct_data (numpy.ndarray): 3D array of CT data with dimensions (height, width, depth).
    suv_inds (numpy.ndarray): 2D array of depth indices corresponding to the MIP x number of MIPs.

    Returns:
    numpy.ndarray: 2D CT image x number of MIPs
    """

    height, width, num_of_mips = suv_inds.shape
    ct_mips = np.zeros((height, width, num_of_mips))
    all_angles = np.linspace(start=starting_angle, stop=(ending_angle - ending_angle / num_of_mips), num=num_of_mips)

    for i, angle in enumerate(all_angles):
        rad_angle = (2 * np.pi) * (angle / 360)

        rotation = monai.transforms.Affine(rotate_params=(rad_angle, 0, 0),
                                          image_only=True,
                                          padding_mode='zeros',
                                          mode='bilinear')

        ct_data_rot = rotation(ct_data)

        rows, cols = np.indices((height, width))
        ct_mips[:, :, i] = ct_data_rot[rows, cols, suv_inds[:, :, i]]

    return ct_mips


def load_nifti_image(file_path):
    """
    Loads a NIfTI image file and returns the image data as a numpy array.

    Parameters:
    file_path (str): Path to the NIfTI file.

    Returns:
    numpy.ndarray: The image data as a numpy array.
    """
    load = monai.transforms.Compose([monai.transforms.LoadImage(image_only=True, ensure_channel_first=True),
                                     monai.transforms.Orientation(axcodes='ILP')])

    img = load(file_path)
    return img


def save_nifti_image(data, file_path, reference_img):
    """
    Saves a numpy array as a NIfTI image file.

    Parameters:
    data (numpy.ndarray): The image data to save.
    file_path (str): The file path to save the image.
    reference_img (nib.Nifti1Image): Reference NIfTI image to copy the affine and header.
    """
    new_img = nib.Nifti1Image(data, affine=reference_img.affine, header=reference_img.header)
    nib.save(new_img, file_path)


if __name__ == "__main__":
    # Define the file paths
    ct_data_path = "/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/niftis/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/CTres.nii.gz"
    suv_inds_path = "/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs16_75th_25vth_IncSplit_0_180/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/SUV_inds.nii.gz"
    suv_mips_path = "/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs_new/MIPs16_75th_25vth_IncSplit_0_180/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/SUV.nii.gz"
    output_ct_image_path = "/mnt/sda1/PET/CT_mips/CT_mip.nii.gz"

    # Load the 3D CT data and the SUV_inds data
    ct_data = load_nifti_image(ct_data_path)[0]
    suv_inds = load_nifti_image(suv_inds_path)[0]
    suv_mips = load_nifti_image(suv_mips_path)[0]

    # Ensure suv_inds is an integer type for indexing
    suv_inds = suv_inds.astype(int)

    # Create the 2D CT image
    ct_image = create_ct_mips(ct_data, suv_inds, starting_angle=0, ending_angle=180, device='cpu')

    # Save the resulting 2D CT image
    reference_img = nib.load(ct_data_path)  # Use the original CT data as reference for affine/header
    save_nifti_image(ct_image, output_ct_image_path, reference_img)

    print(f"2D CT image saved to {output_ct_image_path}")
