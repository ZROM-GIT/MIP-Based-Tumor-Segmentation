"""
This file is meant to connect between the model names and the model classes
"""

from .Unet.unet3d import UNet3D, UNet2D, ResidualUNet3D
from .my_models.rec import rec2d3d
from .my_models.conv_rec import Conv_rec
from .my_models.trial import TinyModel
from .my_models.MIPUNETR.mip_unetr import MipUNETR
# from .my_models.segmamba import SegMamba
# from .my_models.LightMUNet import LightMUNet
from .my_models.VMUNet.vmunet import VMUNet
from .my_models.DUCKNet.duck_net_3d import DuckNet3D
from .my_models.AR.AR_UNETR import ARUNETR
models = {
    'UNET3D': UNet3D,
    'UNET2D': UNet2D,
    'resUNET3D': ResidualUNet3D,
    'rec': rec2d3d,
    'conv_rec': Conv_rec,
    'TinyModel': TinyModel,
    'MipUNETR': MipUNETR,
    # 'SegMamba': SegMamba,
    # 'LightMUNet': LightMUNet,
    'VMUNet': VMUNet,
    'DuckNet': DuckNet3D,
    'ARUNETR': ARUNETR,
}
