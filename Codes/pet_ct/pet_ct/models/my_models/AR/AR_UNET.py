import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims=2):
        super().__init__()
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        norm = nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d
        self.double_conv = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
            nn.ReLU(inplace=True),
            conv(out_channels, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims=2):
        super().__init__()
        pool = nn.MaxPool2d if spatial_dims == 2 else nn.MaxPool3d
        self.maxpool_conv = nn.Sequential(
            pool(2),
            DoubleConv(in_channels, out_channels, spatial_dims)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims=2, bilinear=True):
        super().__init__()
        self.spatial_dims = spatial_dims
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear' if spatial_dims == 2 else 'trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, spatial_dims)
        else:
            conv_transpose = nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d
            self.up = conv_transpose(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, spatial_dims)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[-2] - x1.size()[-2]
        diffX = x2.size()[-1] - x1.size()[-1]
        if self.spatial_dims == 3:
            diffZ = x2.size()[-3] - x1.size()[-3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2,
                            diffZ // 2, diffZ - diffZ // 2])
        else:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims=2):
        super().__init__()
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.conv = conv(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, chunk_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.scale = (hidden_size // num_heads) ** -0.5

    def forward(self, x):
        batch, channels, *spatial_dims = x.size()
        x = x.view(batch, channels, -1).permute(0, 2, 1)  # [B, H*W, C]

        # Process in chunks to save memory
        chunks = x.chunk(max(x.shape[1] // self.chunk_size, 1), dim=1)
        output_chunks = []

        for chunk in chunks:
            q, k, v = self.qkv(chunk).chunk(3, dim=-1)

            q = q.view(batch, -1, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)
            k = k.view(batch, -1, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)
            v = v.view(batch, -1, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)

            # Compute attention scores and apply them to values
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            chunk_out = (attn @ v).transpose(1, 2).contiguous()
            chunk_out = chunk_out.view(batch, -1, self.hidden_size)
            output_chunks.append(chunk_out)

        out = torch.cat(output_chunks, dim=1)
        out = self.proj(out).permute(0, 2, 1).view(batch, channels, *spatial_dims)
        return out

class TokenAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers=3):
        super().__init__()
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # Pool along the MIP channel

    def forward(self, x):
        # Reshape x for multi-head attention: (B, C, H, W) -> (B, H*W, C)
        batch, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)

        # Apply multi-head self-attention 3 times
        for attention in self.attention_layers:
            x, _ = attention(x, x, x)

        # Reshape back to (B, C, H, W)
        x = x.view(batch, height, width, channels).permute(0, 3, 1, 2)

        # Apply average pooling along the MIP channel (dimension 2)
        x = self.avg_pool(x)  # Shape: (B, C, 1, W)
        return x

class MipEmbedding(nn.Module):
    def __init__(self, in_channels, embed_size, num_of_mips, width=400, spatial_dims=2, mode="conv", kernel_size=3, stride=1, padding=1, use_rope=False, token_size=10, num_heads=4):
        super().__init__()
        self.mode = mode
        self.num_of_mips = num_of_mips
        self.use_rope = use_rope
        self.embed_size = embed_size
        self.token_size = token_size  # Token size for downsampling
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        norm = nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d

        if mode == "conv":
            self.embedding = nn.Sequential(
                conv(in_channels, embed_size, kernel_size, stride, padding),
                nn.ReLU(inplace=True),
                norm(embed_size)
            )
        elif mode == "mlp":
            self.embedding = nn.Sequential(
                nn.Linear(400, embed_size),
                nn.ReLU(inplace=True),
                nn.Linear(embed_size, embed_size)
            )
        elif mode == "sinusoidal":
            self.embedding = None
        elif mode == "angular_spatial":
            self.conv1 = nn.Conv2d(2, embed_size, kernel_size=(1, 3), padding=(0, 1))
            self.conv2 = nn.Conv2d(embed_size, embed_size, kernel_size=(1, 3), padding=(0, 1))
            self.conv3 = nn.Conv2d(embed_size, embed_size, kernel_size=(1, 3), padding=(0, 1))
            self.relu = nn.ReLU()

            # Positional embeddings
            self.spatial_pos_embed = nn.Parameter(torch.zeros(1, embed_size, 1, width))
            self.angular_pos_embed = nn.Parameter(torch.zeros(1, embed_size, num_of_mips, 1))

            self._init_positional_embeddings()

            self.tokenizer = nn.Conv2d(
            in_channels=embed_size,  # Number of input channels
            out_channels=embed_size,  # Keep the same number of channels
            kernel_size=(1, self.token_size),  # Tokenize along the width
            stride=(1, self.token_size),  # Stride equal to token size
            padding=0,  # No padding
            groups=embed_size  # Depthwise convolution
        )
            self.token_attention = TokenAttention(embed_size=16, num_heads=4)

        elif mode == "rope":
            self.embedding = nn.Sequential(
                conv(in_channels, embed_size, kernel_size, stride, padding),
                nn.ReLU(inplace=True),
                norm(embed_size)
            )

    def forward(self, x):
        if self.mode == "conv" or self.mode == "rope":
            x = self.embedding(x)
            if self.use_rope:
                x = self.apply_rope(x)
        elif self.mode == "mlp":
            batch_size, _, num_of_mips = x.size()
            x = x.view(batch_size * num_of_mips, -1)
            x = self.embedding(x)
            x = x.view(batch_size, num_of_mips, -1).permute(0, 2, 1).unsqueeze(2)
        elif self.mode == "sinusoidal":
            x = self.apply_sinusoidal(x)
        elif self.mode == "angular_spatial":
            x = self.apply_angular_spatial(x)
            x = self.tokenizer(x)
            x = self.token_attention(x)
            x = self.concatenate_features(x)

        return x

    def concatenate_features(self, x):
        return rearrange(x, 'b c h w -> b (c h w)')

    def _init_positional_embeddings(self):
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.angular_pos_embed, std=0.02)

    def _generate_angular_embedding(self, num_of_mips, embed_size, device):
        angles = torch.linspace(0, 180, steps=num_of_mips, device=device)  # [48]
        angles = torch.deg2rad(angles)  # Convert to radians

        # Now build [sin(θ), cos(θ), sin(2θ), cos(2θ), sin(3θ), cos(3θ), ...]
        embeddings = []
        for i in range(embed_size // 2):
            embeddings.append(torch.sin((i + 1) * angles))  # sin((i+1)θ)
            embeddings.append(torch.cos((i + 1) * angles))  # cos((i+1)θ)

        angular_pos = torch.stack(embeddings, dim=0)  # Shape: [embed_size, num_of_mips]
        angular_pos = angular_pos.unsqueeze(0).unsqueeze(-1)  # [1, embed_size, num_of_mips, 1]
        return angular_pos  # Ready for addition

    def apply_rope(self, x):
        batch, channels, *spatial_dims = x.size()
        pos = torch.arange(spatial_dims[-1], dtype=torch.float32, device=x.device)
        angle_rates = 1 / (10000 ** (torch.arange(0, channels, 2, device=x.device) / channels))
        angles = pos[:, None] * angle_rates[None, :]
        sin, cos = angles.sin(), angles.cos()
        rope = torch.cat([sin, cos], dim=-1).unsqueeze(0).unsqueeze(2)
        x = x * rope + torch.roll(x, shifts=1, dims=1) * rope
        return x

    def apply_sinusoidal(self, x):
        batch_size, _, num_of_mips = x.size()
        pos = torch.arange(400, dtype=torch.float32, device=x.device).unsqueeze(0)
        angle_rates = 1 / (10000 ** (torch.arange(0, self.embed_size, 2, device=x.device) / self.embed_size))
        angles = pos * angle_rates
        sin, cos = angles.sin(), angles.cos()
        sinusoidal = torch.cat([sin, cos], dim=-1).unsqueeze(0).repeat(batch_size, num_of_mips, 1, 1)
        return sinusoidal

    def apply_angular_spatial(self, x):
        batch_size, C, num_of_mips, width = x.size()

        # 3 Conv layers + ReLU
        x = self.relu(self.conv1(x))  # (batch_size, embed_size, num_of_mips, width)
        x = self.relu(self.conv2(x))  # (batch_size, embed_size, num_of_mips, width)
        x = self.relu(self.conv3(x))  # (batch_size, embed_size, num_of_mips, width)

        # Add angular positional embedding
        angular_pos = self._generate_angular_embedding(num_of_mips, self.embed_size, x.device)  # Shape (1, embed_size, num_of_mips, 1)
        x = x + angular_pos

        # Add spatial positional embedding
        x = x + self.spatial_pos_embed

        return x  # (batch_size, embed_size, num_of_mips, width)


class MipFeatures(nn.Module):
    def __init__(self, in_channels, channels, strides, num_heads=4, spatial_dims=2, width=400, embedding_mode="conv", num_of_mips=48, token_size=10):
        """
        Args:
            in_channels: Number of input channels.
            channels: List of feature sizes for each depth.
            strides: List of strides for depth transitions.
            num_heads: Number of attention heads.
            spatial_dims: 2D or 3D input data.
            embedding_mode: Mode for the MipEmbedding class (e.g., "conv", "mlp", etc.).
            num_of_mips: Number of MIPs to process.
        """
        super().__init__()
        self.depths = len(strides)
        self.spatial_dims = spatial_dims
        self.attention_blocks = nn.ModuleList()
        self.transition_blocks = nn.ModuleList()

        # Initialize MipEmbedding with num_of_mips
        self.embedding = MipEmbedding(
            in_channels=in_channels,
            embed_size=channels[0],
            num_of_mips=num_of_mips,
            width=width,
            spatial_dims=spatial_dims,
            mode=embedding_mode,
            num_heads=num_heads,
            token_size=token_size,
        )

        # # Define self-attention and transition blocks for each depth
        # for i in range(self.depths):
        #     # Self-attention block (3 times per depth)
        #     self.attention_blocks.append(
        #         nn.Sequential(*[SelfAttention(channels[i], num_heads) for _ in range(3)])
        #     )
        #
        #     # Transition block (max pooling + convolution)
        #     if i < self.depths - 1:
        #         pool = nn.MaxPool2d if spatial_dims == 2 else nn.MaxPool3d
        #         conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        #         self.transition_blocks.append(
        #             nn.Sequential(
        #                 pool(kernel_size=strides[i]),
        #                 conv(channels[i], channels[i + 1], kernel_size=3, padding=1),
        #                 nn.ReLU(inplace=True)
        #             )
        #         )

        # Fully connected layers for feature reduction
        self.fc_layers = nn.ModuleList()
        input_size = 768
        for out_size in channels[::-1]:  # Reverse the channels list
            self.fc_layers.append(nn.Linear(input_size, out_size))
            input_size = out_size

    def forward(self, suv_mips, seg_mips):
        # Concatenate SUV MIPs and SEG MIPs along the channel dimension
        x = torch.cat((suv_mips, seg_mips), dim=1)

        # Apply MipEmbedding
        x = self.embedding(x)

        # # Collect features from all depths
        # features = []
        # for i in range(self.depths):
        #     # Apply self-attention block
        #     x = self.attention_blocks[i](x)
        #     features.append(x)
        #
        #     # Apply transition block if not the last depth
        #     if i < self.depths - 1:
        #         x = self.transition_blocks[i](x)

        # Apply fully connected layers
        fc_features = []
        x = x.view(x.size(0), -1)  # Flatten the tensor
        for fc in self.fc_layers:
            x = fc(x)
            fc_features.append(x)

        return fc_features[::-1]


class ARUNET(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 channels,
                 strides,
                 num_heads=4,
                 spatial_dims=2,
                 embedding_mode="conv",
                 num_of_mips=48,
                 width=400,
                 token_size=10):
        super().__init__()
        self.spatial_dims = spatial_dims

        # Initial convolution
        self.inc = DoubleConv(in_channels, channels[0], spatial_dims)

        # Create encoder blocks
        self.encoders = nn.ModuleList()
        for i in range(1, len(channels)):
            self.encoders.append(Down(channels[i - 1], channels[i], spatial_dims))

        # Create decoder blocks
        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoders.append(Up(channels[i] * 2, channels[i - 1], spatial_dims))

        # Add an additional decoder for the first encoder
        self.decoders.append(Up(channels[0] * 2, channels[0], spatial_dims))

        # Final output layer
        self.outc = OutConv(channels[0], out_channels, spatial_dims)

        # Feature extractor for MIP features
        self.feature_extractor = MipFeatures(
            in_channels=in_channels * 2,  # Concatenated SUV and SEG MIPs
            channels=channels,
            strides=strides,
            num_heads=num_heads,
            spatial_dims=spatial_dims,
            embedding_mode=embedding_mode,
            num_of_mips=num_of_mips,
            width=width,
            token_size=token_size
        )

    def forward(self, x, suv_mips, seg_mips):
        # Extract MIP features
        fc_features = self.feature_extractor(suv_mips, seg_mips)

        # Encoder path
        enc_features = []
        x1 = self.inc(x)

        # Add fc_features to all "pixels" for the first encoder
        if 0 < len(fc_features):
            fc_feature = fc_features[0].unsqueeze(-1).unsqueeze(-1)  # Reshape to (B, C, 1, 1)
            x1 = x1 + fc_feature  # Broadcast addition

        enc_features.append(x1)

        for i, encoder in enumerate(self.encoders):
            x1 = encoder(x1)

            # Add fc_features to all "pixels"
            if i + 1 < len(fc_features):
                fc_feature = fc_features[i+1].unsqueeze(-1).unsqueeze(-1)  # Reshape to (B, C, 1, 1)
                x1 = x1 + fc_feature  # Broadcast addition

            enc_features.append(x1)

        # Decoder path
        for decoder, enc_feat in zip(self.decoders, enc_features[::-1]):
            x1 = decoder(x1, enc_feat)

        # Output layer
        return self.outc(x1)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example usage
    model = ARUNET(
        in_channels=1,
        out_channels=2,
        channels=[16, 32, 64, 128],
        strides=[2, 2, 2],
        spatial_dims=2,
        num_heads=4,
        embedding_mode="angular_spatial",
        num_of_mips=48,
        width=400,
        token_size=10
    ).to(device)

    suv_2d = torch.randn(1, 1, 400, 400).to(device)  # Example input
    suv_mip = torch.randn(1, 1, 48, 400).to(device)  # Example SUV MIP
    seg_mip = torch.round(torch.randn(1, 1, 48, 400)).to(device)

    output = model(suv_2d, suv_mip, seg_mip)
    print(output.shape)

    # model = ARUNET(
    #     spatial_dims=2,
    #     in_channels=1,
    #     out_channels=2,
    #     channels=[16, 32, 64, 128, 256],
    #     strides=[2, 2, 2, 2],
    #     kernel_size=3,
    #     up_kernel_size=3,
    #     num_res_units=2,
    #     act="relu",
    #     norm="batch",
    #     dropout=0.1,
    #     bias=True,
    #     adn_ordering="NDA",
    #     fusion_channels=64,
    #     use_fusion=True,
    # )

    # Create a random input tensor
    suv_2d = torch.randn(1, 1, 400, 400)
    suv_mip = torch.randn(1, 1, 48, 400)
    seg_mip = torch.round(torch.randn(1, 1, 48, 400))

    # Forward pass
    output = model(suv_2d, suv_mip, seg_mip)
    print()
