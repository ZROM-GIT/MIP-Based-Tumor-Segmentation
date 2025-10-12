import warnings
from collections.abc import Sequence
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection

class UNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        use_fusion: bool = False,
        fusion_channels: int = None,
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.use_fusion = use_fusion

        if use_fusion and fusion_channels is not None:
            self.fusion_proj = nn.Conv2d(fusion_channels, channels[-1], kernel_size=1)
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)  # output shape: [B, 64, 1, 1]
            )
        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            c = channels[0]
            s = strides[0]
            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)
                upc = c * 2
            else:
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)
            up = self._get_up_layer(upc, outc, s, is_top)

            block = self._get_connection_block(down, up, subblock)
            return block

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions, in_channels, out_channels, strides=strides,
                kernel_size=self.kernel_size, subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
        return Convolution(
            self.dimensions, in_channels, out_channels, strides=strides,
            kernel_size=self.kernel_size, act=self.act, norm=self.norm,
            dropout=self.dropout, bias=self.bias, adn_ordering=self.adn_ordering,
        )

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv = Convolution(
            self.dimensions, in_channels, out_channels, strides=strides,
            kernel_size=self.up_kernel_size, act=self.act, norm=self.norm,
            dropout=self.dropout, bias=self.bias, conv_only=is_top and self.num_res_units == 0,
            is_transposed=True, adn_ordering=self.adn_ordering,
        )
        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions, out_channels, out_channels, strides=1,
                kernel_size=self.kernel_size, subunits=1, act=self.act, norm=self.norm,
                dropout=self.dropout, bias=self.bias, last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            return nn.Sequential(conv, ru)
        return conv


    def forward(self, suv_2d: torch.Tensor, suv_mip, seg_mip) -> torch.Tensor:
        down_path, skip_conn, up_path = self.model
        down_out = down_path(suv_2d)
        skip_out = skip_conn(down_out)

        features = self.feature_extractor(torch.concat([suv_mip, suv_2d], dim=1))  # shape: [B, 2, H, W] → [B, 64, 1, 1]

        fusion_proj = self.fusion_proj(features)  # shape: [B, C, 1, 1] → project
        fusion_proj = fusion_proj.expand_as(skip_out)
        skip_out = skip_out + fusion_proj

        x = up_path(skip_out)
        return x

if __name__ == '__main__':
    # Example usage
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=[16, 32, 64, 128, 256],
        strides=[2, 2, 2, 2],
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=2,
        act="relu",
        norm="batch",
        dropout=0.1,
        bias=True,
        adn_ordering="NDA",
        fusion_channels=64,
        use_fusion=True,
    )

    # Create a random input tensor
    suv_2d = torch.randn(1, 1, 400, 400)
    suv_mip = torch.randn(1, 1, 48, 400)
    seg_mip = torch.round(torch.randn(1, 1, 48, 400))

    # Forward pass
    output = model(suv_2d, suv_mip, seg_mip)
    print()
