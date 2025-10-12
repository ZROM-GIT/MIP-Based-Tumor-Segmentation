import nibabel
import torch
import monai

import hiera
from hiera.hiera import Hiera

# image_model = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")
# data = nibabel.load('/mnt/sda1/PET/Datasets/FDG-PET-CT-Lesions/manifest-1654187277763/MIPs/MIPs16/PETCT_0af7ffe12a/08-12-2005-NA-PET-CT Ganzkoerper  primaer mit KM-96698/0_SUV.nii.gz')
# array = data.get_fdata()
# array = torch.from_numpy(array.astype('float32'))

# video_model = torch.hub.load("facebookresearch/hiera", model="hiera_base_16x224", pretrained=True, checkpoint="mae_k400_ft_k400")
model = hiera.hiera_base_16x224()

video_model = Hiera(
        num_classes=10,  # K400 has 400 classes
        input_size=(16, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_pos_embed=True
    )
video_model(torch.rand(size=(1,3,16,208,216)))
print()