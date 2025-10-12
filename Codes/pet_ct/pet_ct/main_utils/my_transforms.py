"""
This file contains any self-made data transforms used.
"""

import datetime as dt
import monai
from monai.data import MetaTensor
from monai.transforms import MapTransform, InvertibleTransform
import numpy as np
import os
from pathlib2 import Path
import pydicom
import re
import shutil
import torch
import torch.nn.functional as F
from sympy.codegen.ast import continue_
from tqdm import tqdm

from pet_ct.secondary_utils.createMIP import createMIP
from pet_ct.secondary_utils.createMIP_new import create_mip_new

class LoadDicomPETasSUV(MapTransform):
    def __init__(self, keys='seg', allow_missing_keys=False, convertFromPacs=True, ensure_channel_first=True):
        super().__init__(keys, allow_missing_keys)
        self.convertFromPacs = convertFromPacs
        self.ensure_channel_first = ensure_channel_first

    def __call__(self, data):
        data = dict(data)
        for key in self.keys:
            data[key], meta = self.readDicomSeriesWithMeta(path=data[key])
            if self.ensure_channel_first:
                data[key] = np.expand_dims(data[key], axis=0)
            affine = self.getAffineFromDicomMeta(meta)
            data[key] = MetaTensor(data[key])
        return data

    def readDicomSeriesWithMeta(self, path,
                                insertInstanceNumber=False,
                                calc_suv=True,
                                return_hu=False,
                                only_first_dcm=False,
                                dcm_idx=0,
                                useForceRead=False,
                                printProcess=True):

        singleDicom = False
        if not os.path.isdir(path):
            dicomFiles = [path]
            singleDicom = True
        else:
            self.convertFromPACS(path)
            dicomFiles = os.listdir(path)

        self.sliceNumsFilenames = []

        dicomFiles = dicomFiles if not only_first_dcm else dicomFiles[dcm_idx: dcm_idx + 1]
        numFiles = len(dicomFiles)
        meta = [None] * numFiles

        if not singleDicom:
            sameSliceNumber = self.check_duplicate_instance_numbers(path)
        else:
            sameSliceNumber = False

        if printProcess:
            loader = tqdm(dicomFiles, position=0, leave=True)
        else:
            loader = dicomFiles
        I = None

        for i, file in enumerate(loader):

            dataset = pydicom.read_file(os.path.join(path, file), force=useForceRead)
            if not (0x7FE0, 0x10) in dataset:
                continue

            if not insertInstanceNumber and dataset[0x8, 0x60].value == 'MG':
                insertInstanceNumber = True

            if (sameSliceNumber and not ['0x7A1', '0x103E'] in dataset):
                insertInstanceNumber = True

            if (sameSliceNumber and ['0x7A1', '0x103E'] in dataset):
                slice_number = int(dataset['0x7A1', '0x103E'].value) - 1
            else:
                if (not ['20', '13'] in dataset):
                    continue
                if (not insertInstanceNumber):
                    slice_number = int(
                        dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python
                else:
                    slice_number = i

                if only_first_dcm or singleDicom:
                    slice_number = 0

            if (i == 0):
                RGB = True if (0x28, 0x2) in dataset and dataset[0x28, 0x2].value == 3 else False
                if (RGB):
                    I = np.zeros((numFiles, dataset.pixel_array.shape[0], dataset.pixel_array.shape[1],
                                  dataset.pixel_array.shape[2]), dtype=dataset.pixel_array.dtype)
                else:
                    I = np.zeros((numFiles, dataset.pixel_array.shape[0], dataset.pixel_array.shape[1]),
                                 dtype=dataset.pixel_array.dtype)
                if [0x28, 0x1052] in dataset:
                    self.airVal = float(dataset['0x28', '0x1052'].value)
                else:
                    self.airVal = 0
            else:
                if not isinstance(I, list) and I.shape[1:] != dataset.pixel_array.shape:
                    I_ = I
                    I = [None] * I_.shape[0]
                    for j in range(i):
                        I[j] = I_[j, ...]

            if slice_number < len(meta):
                meta[slice_number] = dataset
            tmpIm = dataset.pixel_array
            tmpIm[tmpIm < self.airVal] = 0

            if dataset.Modality == 'PT':

                if calc_suv:
                    tmpIm = tmpIm * float(dataset[0x28, 0x1053].value) + float(dataset[0x28, 0x1052].value)
                    tmpIm = self.calc_suv_func(tmpIm, dataset)
                    I = I.astype('float32')

                currentMaxDtype = self.find_max_dtype(I.dtype)
                if tmpIm.max() > currentMaxDtype:
                    match = re.match(r"([a-z]+)([0-9]+)", str(dataset.pixel_array.dtype), re.I)
                    split = match.groups()
                    newDtype = split[0] + str(int(split[1]) * 2)
                    I = I.astype(newDtype)

            _slice = None
            if isinstance(I, list):
                I[i] = tmpIm
                _slice = i
            else:
                if (slice_number < I.shape[0]):
                    I[slice_number, ...] = tmpIm
                    _slice = slice_number

            if not _slice is None:
                self.sliceNumsFilenames.append({
                    'slice': _slice,
                    'filename': Path(path, file)})

        instances = [x for x in range(len(meta)) if not meta[x] is None]

        if len(instances) != numFiles:
            meta = [meta[x] for x in instances]
            I = I[instances]

        if return_hu:
            slope = float(meta[0].RescaleSlope)
            intercept = int(meta[0].RescaleIntercept)
            I = ((I + intercept) * slope).astype(I.dtype)

        return I, meta

    def check_duplicate_instance_numbers(self, path):

        if 'dcm' in path:
            return False

        dicomFiles = os.listdir(path)
        hasDuplicates = False
        sliceNumbers = []

        for i, file in enumerate(dicomFiles):

            dataset = pydicom.read_file(os.path.join(path, file), force=True)

            if not hasattr(dataset, 'InstanceNumber'):
                continue

            sliceNumber = int(dataset.InstanceNumber) - 1  # InstanceNumber starts from 1 and not zero as in python
            sliceNumbers.append(sliceNumber)
            slices = [x for x in sliceNumbers if x == sliceNumber]
            if (len(slices) > 2):
                hasDuplicates = True
                break
            if (i > 3):
                break

        return hasDuplicates

    def calc_suv_func(self, Ac, ds):

        startTime = ds[0x8, 0x32].value
        startTime = dt.datetime.strptime(startTime, '%H%M%S')
        weight = float(ds[0x10, 0x1030].value) * 1e3
        doseInfo = ds[0x54, 0x16][0]
        totalDose = float(doseInfo[0x18, 0x1074].value)
        injectionTime = doseInfo[0x18, 0x1072].value
        injectionTime = dt.datetime.strptime(injectionTime, '%H%M%S')
        halfLife = float(doseInfo[0x18, 0x1075].value)
        deltaT = (startTime - injectionTime).seconds

        num = weight * Ac
        denom = totalDose * (2 ** (-deltaT / halfLife))
        SUV = num / denom

        return SUV

    def find_max_dtype(self, dtype):

        if (isinstance(dtype, str) and 'float' in dtype) or (isinstance(dtype, np.dtype) and 'float' in dtype.name):
            m = np.finfo(dtype).max
        else:
            m = np.iinfo(dtype).max

        return m

    def convertFromPACS(self, directory):
        files = os.listdir(directory)
        if len(files) == 0:
            return
        # TODO: FIX this to work without the .dcm ext
        # dicom_files = [f for f in files if 'dcm' in f]
        dicom_files = files
        if len(dicom_files) == 0 and self.convertFromPacs:
            print('converting the following directory: ' + directory)
            [os.remove(os.path.join(directory, f)) for f in files if
             (not os.path.isdir(os.path.join(directory, f)) and '.' not in f)]
            for folder, subs, files in os.walk(directory):
                if 'VERSION' in files and len(subs) == 0:
                    os.remove(os.path.join(folder, 'VERSION'))
                    for f in files:
                        if 'VERSION' in f:
                            continue
                        new_file = os.path.join(folder, f + '.dcm')
                        os.rename(os.path.join(folder, f), new_file)
                        shutil.copy2(new_file, directory)
                    break
            files = os.listdir(directory)
            [shutil.rmtree(os.path.join(directory, f))
             for f in files if os.path.isdir(os.path.join(directory, f))]
        elif len(dicom_files) == 0:
            print('no dicom files in {}, maybe convert from PACS? '
                  'add convertFromPacs = True for conversion'.format(directory))

    def getAffineFromDicomMeta(self, meta, multi_slice=True):
        if not multi_slice:
            (Sx, Sy, Sz) = meta.ImagePositionPatient
            F1 = (F11, F21, F31) = meta.ImageOrientationPatient[3:]
            F2 = (F12, F22, F32) = meta.ImageOrientationPatient[0:3]
            dr, dc = meta.PixelSpacing
            affine = torch.as_tensor([[F11*dr, F12*dc, 0, Sx],
                                      [F21*dr, F22*dc, 0, Sy],
                                      [F31*dr, F32*dc, 0, Sz],
                                      [   0  ,    0  , 0,  1]])

        else:
            meta_1 = meta[0]
            meta_n = meta[-1]
            (Sx1, Sy1, Sz1) = meta_1.ImagePositionPatient
            (Sxn, Syn, Szn) = meta_n.ImagePositionPatient
            F1 = (F11, F21, F31) = meta_1.ImageOrientationPatient[3:]
            F2 = (F12, F22, F32) = meta_1.ImageOrientationPatient[0:3]
            dr, dc = meta_1.PixelSpacing
            ds = abs(meta_1.ImagePositionPatient[-1] - meta_n.ImagePositionPatient[-1])/(len(meta) - 1)
            n = (n1, n2, n3) = np.cross(F1, F2)
            affine = torch.as_tensor([[F11*dr, F12*dc, (Sxn-Sx1)/(len(meta)-1), (-1)*Sx1],
                                      [F21*dr, F22*dc, (Syn-Sy1)/(len(meta)-1), (-1)*Sy1],
                                      [F31*dr, F32*dc, (Szn-Sz1)/(len(meta)-1), Sz1],
                                      [   0  ,    0  ,           0            ,  1 ]], dtype=torch.float32)
            M = torch.tensor([[0, -1, 0, 0],
                                   [-1, 0, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=torch.float32)
            affine_new = torch.matmul(affine, M)
        return affine_new


class onehot_vector(MapTransform):
    """
    Transforms a tensor into a n-dimensional one-hot vector
    """
    def __init__(self, keys='seg', allow_missing_keys=False, multi_class=False, ground_truth='seg', dim=-1):
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.multi_class = multi_class
        self.ground_truth = ground_truth
        self.dim = dim

    def __call__(self, data):
        data = dict(data)
        if self.multi_class:
            d = self._FilterTensors(data)
            d[self.ground_truth] = self._onehot(d[self.ground_truth])
            return d
        else:
            data[self.ground_truth] = self._onehot(data[self.keys[0]])
            return data

    def _FilterTensors(self, data):
        M = max(data[self.ground_truth].max(), 1)
        for key in self.key_iterator(data):
            if key == self.ground_truth:
                continue
            value = data[key]
            value[(value != 0) & (data[self.ground_truth] != 0)] = 0
            value[value != 0] += M
            data[self.ground_truth] += value
            data[key] = value
        return data

    def _onehot(self, data):
        if not isinstance(data, torch.Tensor):
            raise ValueError()
        if (data == 0).all():  # If tensor is all zeros
            new_data = F.one_hot(data, 2)
        else:
            new_data = F.one_hot(data, int(data.max()) + 1)
        new_data = new_data.squeeze()
        new_data = new_data.movedim(-1, self.dim)
        return new_data


class BinaryThreshold(MapTransform):
    def __init__(self, keys='seg', allow_missing_keys=False, threshold=0.5):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.combine_threshold_intensity_functions(d[key])
        return d

    def combine_threshold_intensity_functions(self, data):
        transform = monai.transforms.Compose([
            monai.transforms.ThresholdIntensity(threshold=self.threshold, above=True, cval=0),
            monai.transforms.ThresholdIntensity(threshold=self.threshold, above=False, cval=1)
        ])
        data = transform(data)
        return data

# class onehot_bin(MapTransform):
#     """
#     Transforms a tensor into an n-dimensional one-hot vector
#     """
#     def __init__(self, keys='seg', allow_missing_keys=False, dim=-1, dtype=None):
#         super().__init__(keys, allow_missing_keys)
#         self.keys = keys
#         self.allow_missing_keys = allow_missing_keys
#         self.dim = dim
#
#     def __invert__(self):
#         pass
#
#     def __call__(self, data):
#         data = dict(data)
#
#         for key in self.keys:
#             self.dtype = data[key].dtype  # Save original dtype
#             data[key] = data[key].to(dtype=torch.long)  # Dtype long (int64) for one hot
#             data[key] = self._onehot(data[key])  # Execute one hot
#             data[key] = data[key].to(dtype=self.dtype)  # Return to original dtype
#
#         return data
#
#
#     def _onehot(self, data):
#         if not isinstance(data, torch.Tensor):
#             raise ValueError()
#         data[data != 0] = 1
#         if not data.any():  # If tensor is all zeros
#             new_data = F.one_hot(data, 2)
#         else:
#             new_data = F.one_hot(data, -1)
#         new_data = new_data.squeeze()
#         new_data = new_data.movedim(-1, self.dim)
#         return new_data


import torch
from monai.transforms import MapTransform, InvertibleTransform


class onehot_bin(MapTransform): #, InvertibleTransform):
    """
    Transforms a tensor into an n-dimensional one-hot vector and supports inverting the transformation.
    """

    def __init__(self, keys='seg', allow_missing_keys=False, dim=-1, dtype=None):
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.dim = dim  # One-hot dimension
        self.dtype = dtype  # Optional: store the data type

    def __call__(self, data):
        data = dict(data)

        for key in self.keys:
            img = data[key]

            # Save original dtype and shape for inversion
            orig_dtype = img.dtype
            orig_shape = img.shape

            # Convert to long (int64) and apply one-hot encoding
            img = img.to(dtype=torch.long)
            img = self._onehot(img, dim=self.dim)

            # Cast back to original dtype if necessary
            img = img.to(dtype=orig_dtype)

            # # Push this transformation to the metadata (applied_operations) using push_transform
            # self.push_transform(
            #     img,
            #     orig_size=orig_shape,
            # )

            data[key] = img

        return data

    # def inverse(self, data):
    #     """
    #     Reverse the one-hot transformation using stored metadata in applied_operations.
    #     """
    #
    #     for key in self.keys:
    #         img = data[key]
    #
    #         # Pop the most recent transform
    #         transform = self.pop_transform(img)
    #
    #         # Undo the one-hot encoding by converting back to class indices
    #         img = torch.argmax(img, dim=transform['dim'])
    #
    #         # Reshape the tensor to its original shape
    #         img = img.view(transform['orig_size'])
    #
    #         # Cast back to the original dtype if necessary
    #         img = img.to(dtype=transform['orig_dtype'])
    #
    #         data[key] = img
    #
    #     return data

    def _onehot(self, tensor, dim=-1):
        # Example one-hot encoding function for demonstration
        n_classes = max(torch.max(tensor) + 1, 2)  # Calculate number of classes from tensor
        tensor = torch.nn.functional.one_hot(tensor, num_classes=n_classes).moveaxis(-1, dim)
        tensor = tensor.squeeze()
        return tensor

class UniteLabels(MapTransform):
    def __init__(self, keys='seg', allow_missing_keys=False, value=1):
        super().__init__(keys, allow_missing_keys)
        self.value = value

    def __call__(self, data):
        data = dict(data)
        for key in self.keys:
            data[key] = self._UniteLabels(data[key])
        return data

    def _UniteLabels(self, data):
        data[data > 0] = self.value
        return data


class create_mips(MapTransform):
    """
    Transforms a tensor into an n-dimensional one-hot vector
    """
    def __init__(self, keys='seg',
                 allow_missing_keys=False,
                 spatial_dims=3,
                 num_of_mips=16,
                 mip_dim=2,
                 ver='new',
                 start_ang=0,
                 end_ang=180,
                 pixel_size_threshold=0,
                 visual_snr_threshold=0
                 ):
        super().__init__(keys, allow_missing_keys)
        self.spatial_dims = spatial_dims
        self.num_of_mips = num_of_mips
        self.mip_dim = mip_dim
        self.ver = ver
        self.start_ang = start_ang
        self.end_ang = end_ang
        self.pixel_size_threshold = pixel_size_threshold
        self.visual_snr_threshold = visual_snr_threshold

    def __call__(self, data):
        data = dict(data)
        for key in self.keys:
            data[key] = self.create_mips(data[key])
        return data

    def create_mips(self, data):
        dims = len(data.shape)
        spatial_dims = data.shape[-self.spatial_dims:]
        match self.ver:
            case 'old':
                # TODO: Change to old dims
                HEIGHT, WIDTH, DEPTH = spatial_dims[0], spatial_dims[1], spatial_dims[2]
                new_data = torch.zeros(size=(1, self.num_of_mips, WIDTH, HEIGHT))
            case 'new':
                HEIGHT, WIDTH, DEPTH = spatial_dims[0], spatial_dims[1], spatial_dims[2]
                new_data = torch.zeros(size=(1, HEIGHT, WIDTH, self.num_of_mips))
        data = torch.squeeze(data)
        angles = np.linspace(start=self.start_ang, stop=(self.end_ang - (self.end_ang/self.num_of_mips)), num=self.num_of_mips)
        for i, angle in enumerate(angles):
            # TODO: FIX 'create_mip_new' function to compute on torch tensor
            match self.ver:
                case 'old':
                    mip = torch.tensor(createMIP(np_img=data, horizontal_angle=angle, modality='suv').copy())
                    new_data[0, i, :, :] = mip
                case 'new':
                    mip = torch.tensor(create_mip_new(img=data, horizontal_angle=angle, modality='suv', device='cpu'))
                    new_data[0, :, :, i] = mip
        return new_data


class TumorLabelToInt(MapTransform):
    """
    Transforms a tensor into an n-dimensional one-hot vector
    """
    def __init__(self, keys='seg', allow_missing_keys=False, multi=False):
        super().__init__(keys, allow_missing_keys)
        self.multi = multi

    def __call__(self, data):
        data = dict(data)
        for key in self.keys:
            data[key] = self._morph_name_to_int(data[key])
        return data

    def _morph_name_to_int(self, data):
        if self.multi:
            NameToInt = {'NEGATIVE': 0, 'LUNG_CANCER': 1, 'LYMPHOMA': 2, 'MELANOMA': 3}
            data = NameToInt[data]
        else:
            data = 0 if data == 'NEGATIVE' else 1
        return data




