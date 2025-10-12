import numpy as np
from pathlib2 import Path
import subprocess

from ct.dicom.my_dicom import myDicom

def convert_from_pacs(path):
    """
    # Michael helper function for dcm2nii
    :param path: dicom path dir
    """

    dcm = myDicom()
    convertFromPACS = not len([x for x in Path(path).iterdir()
                               if x.suffix == '.dcm'])
    if convertFromPACS:
        dcm.convertFromPACS(path)


def dcm2nifti(path):
    """
    # applying dcm2nii tool as a subprocess
    :param path: dicom path dir
    :return:
    """
    print("hi")
    convert_from_pacs(path)
    dicom2nifti = '/mnt/sda1/PET/tools/dcm2niix_lnx/dcm2niix'
    cmd = dicom2nifti + f' -p y -f {path.name} -z y -o {path.parent} {path}'
    process = subprocess.run(cmd,
                             check=True,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)

    print(f'output dcm2nii - {process.stdout}')
    print(f'output dcm2nii - {process.stderr}')


def apply_dcm2nii(full_path):
    """
    # wraper function for dcm2nifti
    :param full_path: dicom path dir
    """

    path = Path(full_path)
    dcm2nifti(path)


def save_as_dcm(final_pred, path=None):
    """
    save np.ndarray as dicom dir
    :param final_pred: np.ndarray
    :param path: output path for dicom dir
    :return:
    """

    # todo: input paramater path has added due to an error, i'm not really sure if needed #
    # Binary dcm

    to_dcm = np.clip(np.sum(final_pred, axis=3), 0, 1)
    to_dcm = np.transpose(to_dcm, (2, 1, 0))
    to_dcm = to_dcm[:, ::-1, :]
    segOut = Path(path).joinpath('Fiber_segmentation')
    segOutRGB = Path(path).joinpath('Fiber_segmentation_RGB')
    sourcePath = Path(path).joinpath('T1')
    dcmWriter = myDicom()
    dcmWriter.writeDicomSeries(I=to_dcm,
                               source_path=sourcePath.as_posix(),
                               output_path=segOut.as_posix(),
                               description='output')

    # RGB dcm
    temp = final_pred
    temp[:, :, :, 1] = 2 * temp[:, :, :, 1]
    temp[:, :, :, 2] = 3 * temp[:, :, :, 2]
    temp[:, :, :, 3] = 4 * temp[:, :, :, 3]
    number_coding = np.sum(temp, axis=3)
    i, j, k, _ = final_pred.shape
    to_dcm_RGB = np.zeros((i, j, k, 3))
    to_dcm_RGB[number_coding == 1, 0] = 255  # OR_left is Red
    to_dcm_RGB[number_coding == 2, 1] = 255  # OR_right is Green
    to_dcm_RGB[number_coding == 3, 2] = 255  # CST_left is Blue
    to_dcm_RGB[number_coding == 4, 0] = 255  # CST_left is Yellow
    to_dcm_RGB[number_coding == 4, 1] = 255  # CST_left is Yellow
    to_dcm_RGB = np.transpose(to_dcm_RGB, (2, 1, 0, 3))
    to_dcm_RGB = to_dcm_RGB[:, ::-1, ...]

    dcmWriter.writeDicomSeries(I=to_dcm_RGB,
                               source_path=sourcePath.as_posix(),
                               output_path=segOutRGB.as_posix(),
                               description='output')


def convert_T1_and_DWI_dcm2nii(path):
    """
    converte all DWI & T1 dicom dirs to nifti images
    :param path: path dir of multiple dicom subdirs
    """

    path = Path(path)
    for temp in path.iterdir():
        try:
            if Path.is_dir(temp) and "T1" in temp.name:
                dcm2nifti(path=temp)
            if Path.is_dir(temp) and "DWI" in temp.name:
                dcm2nifti(path=temp)
        except Exception as E:
            print(f"dcm2nii exeption at {str(temp)}")
            print(f"{E}\n")

if __name__ == '__main__':
    dcm_path = '/mnt/sda1/PET/Datasets/PACS/42749259/X1WYBG5X/XK3EITSY'
    dcm2nifti(path=Path(dcm_path))