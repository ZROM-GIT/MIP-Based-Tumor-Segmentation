from __future__ import annotations

import torch.nn as nn

from monai.networks.blocks.mlp import MLPBlock
# from monai.networks.blocks.selfattention import SABlock
from .mip_sablock import MipSABlock
from triton.language import block_type

from monai.utils import optional_import
from einops import rearrange

class MipTransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        n_patches=(24, 25, 16),
        block_type: str = "self"  # "self" block / "cross" block
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            mlp_dim (int): dimension of feedforward layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): apply bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.hidden_size = hidden_size
        self.block_type = block_type
        self.n_patches = n_patches

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = MipSABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn, include_mip_dim=True)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        n_self_tokens = self.n_patches[0] * self.n_patches[1]
        h, w = self.n_patches[0], self.n_patches[1]
        n_mips = self.n_patches[-1]

        if self.block_type == "self":
            x = self.norm1(x)
            x = rearrange(x, 'b (hw m) c -> m b hw c', m=n_mips)

            z = self.attn(x)
            x = x + z

            x = rearrange(x, 'm b hw c -> b (hw m) c')
            x = x + self.mlp(self.norm2(x))

        if self.block_type == "cross":
            x = self.norm1(x)
            x = rearrange(x, 'b (h wd) c -> h b wd c', h=h)

            z = self.attn(x)
            x = x + z

            x = rearrange(x, 'h b wd c -> b (h wd) c')
            x = x + self.mlp(self.norm2(x))
        return x
