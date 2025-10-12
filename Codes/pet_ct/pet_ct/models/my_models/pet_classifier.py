import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionPool3D(nn.Module):
    """
    True 3D Attention Pooling using learnable queries to attend to spatial locations
    """

    def __init__(self, in_channels, out_features=512, num_queries=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.num_queries = num_queries

        # Learnable query vectors - these determine what to attend to
        self.queries = nn.Parameter(torch.randn(num_queries, in_channels))

        # Key and Value projections for attention
        self.key_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # Scale factor for attention
        self.scale = in_channels ** -0.5

        # Final projection to desired output features
        self.output_proj = nn.Linear(num_queries * in_channels, out_features)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch_size, channels, depth, height, width)
        batch_size, channels, d, h, w = x.shape
        spatial_size = d * h * w

        # Generate keys and values from input features
        keys = self.key_proj(x).view(batch_size, channels, spatial_size)  # (B, C, D*H*W)
        values = self.value_proj(x).view(batch_size, channels, spatial_size)  # (B, C, D*H*W)

        # Transpose for matrix multiplication
        keys = keys.transpose(1, 2)  # (B, D*H*W, C)
        values = values.transpose(1, 2)  # (B, D*H*W, C)

        # Expand queries for batch processing
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_queries, C)

        # Compute attention scores: Q @ K^T
        attention_scores = torch.matmul(queries, keys.transpose(1, 2)) * self.scale  # (B, num_queries, D*H*W)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, num_queries, D*H*W)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values: Attention @ V
        attended_features = torch.matmul(attention_weights, values)  # (B, num_queries, C)

        # Flatten and project to output dimension
        attended_features = attended_features.view(batch_size, -1)  # (B, num_queries * C)
        output = self.output_proj(attended_features)  # (B, out_features)

        return output


class MultiHeadAttentionPool3D(nn.Module):
    """
    Multi-head self-attention pooling with learnable queries
    """

    def __init__(self, in_channels, out_features=512, num_heads=8, num_queries=4):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.num_queries = num_queries
        self.head_dim = out_features // num_heads

        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"

        # Learnable queries (what we want to extract)
        self.queries = nn.Parameter(torch.randn(num_queries, out_features))

        # Key and Value projections
        self.key_proj = nn.Conv3d(in_channels, out_features, kernel_size=1)
        self.value_proj = nn.Conv3d(in_channels, out_features, kernel_size=1)

        # Multi-head processing
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(0.1)

        # Final projection and normalization
        self.output_proj = nn.Linear(num_queries * out_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x):
        batch_size, channels, d, h, w = x.shape
        spatial_size = d * h * w

        # Generate keys and values from spatial features
        keys = self.key_proj(x).view(batch_size, self.out_features, spatial_size)  # (B, out_features, spatial)
        values = self.value_proj(x).view(batch_size, self.out_features, spatial_size)  # (B, out_features, spatial)

        # Reshape for multi-head attention
        keys = keys.view(batch_size, self.num_heads, self.head_dim, spatial_size).transpose(-2,
                                                                                            -1)  # (B, heads, spatial, head_dim)
        values = values.view(batch_size, self.num_heads, self.head_dim, spatial_size).transpose(-2,
                                                                                                -1)  # (B, heads, spatial, head_dim)

        # Expand queries for batch and multi-head processing
        queries = self.queries.view(self.num_queries, self.num_heads, self.head_dim)  # (num_queries, heads, head_dim)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, num_queries, heads, head_dim)

        # Process each head separately
        attended_heads = []
        for head in range(self.num_heads):
            # Get Q, K, V for this head
            q_head = queries[:, :, head, :]  # (B, num_queries, head_dim)
            k_head = keys[:, head, :, :]  # (B, spatial, head_dim)
            v_head = values[:, head, :, :]  # (B, spatial, head_dim)

            # Compute attention scores
            attn_scores = torch.matmul(q_head, k_head.transpose(-2, -1)) * self.scale  # (B, num_queries, spatial)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            attended = torch.matmul(attn_weights, v_head)  # (B, num_queries, head_dim)
            attended_heads.append(attended)

        # Concatenate all heads
        multi_head_output = torch.cat(attended_heads, dim=-1)  # (B, num_queries, out_features)

        # Flatten queries dimension
        flattened = multi_head_output.view(batch_size, -1)  # (B, num_queries * out_features)

        # Final projection
        output = self.output_proj(flattened)  # (B, out_features)
        output = self.layer_norm(output)

        return output


class MLP(nn.Module):
    def __init__(self, hidden_sizes, dropout_rate=0.3):
        super().__init__()
        layers = []
        self.linears = []  # Store Linear layers to apply custom init later
        for i in range(len(hidden_sizes) - 1):
            linear = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            self.linears.append(linear)
            layers.extend([
                linear,
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for linear in self.linears:
            # Example: Xavier (Glorot) initialization
            init.xavier_uniform_(linear.weight)
            init.zeros_(linear.bias)

    def forward(self, x):
        return self.mlp(x)


class PETClassifier(nn.Module):
    def __init__(self,
                 in_channels=1,
                 encoder_features=[64, 128, 256, 512],
                 attention_out_features=512,
                 hidden_sizes=[512, 256, 128, 64],
                 num_classes=4,
                 dropout=0.3,
                 attention_type='simple',  # 'simple', 'multihead', 'hierarchical', 'multiscale'
                 num_attention_heads=8,
                    ):
        super().__init__()

        # Encoder
        self.encoder = nn.ModuleList()
        curr_channels = in_channels
        for feat in encoder_features:
            self.encoder.append(ConvBlock(curr_channels, feat))
            curr_channels = feat

        # Attention Pooling
        if attention_type == 'multihead':
            self.attention_pool = MultiHeadAttentionPool3D(
                encoder_features[-1],
                attention_out_features,
                num_attention_heads
            )
        else:  # simple
            self.attention_pool = AttentionPool3D(
                encoder_features[-1],
                attention_out_features,
                num_queries=8  # Increased from 4 for richer representation
            )

        # Classifier
        classifier_sizes = [attention_out_features] + hidden_sizes + [num_classes]
        self.classifier = MLP(classifier_sizes, dropout)

    def forward(self, x):
        # Encoder
        for conv_block in self.encoder:
            x = conv_block(x)

        # Attention Pooling
        x = self.attention_pool(x)

        # Classification
        x = self.classifier(x)

        return x

    def predict_class(self, x):
        return torch.argmax(x, dim=1, keepdim=False)


# Example usage and testing
if __name__ == "__main__":
    # Test different attention types
    attention_types = ['simple', 'multihead']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for attention_type in attention_types:
        print(f"\n=== Testing {attention_type.upper()} Attention ===")

        model = PETClassifier(
            attention_type='multihead',  # Change to 'simple' or 'multihead' to test
            attention_out_features=768, # Larger hidden size
            encoder_features=[32, 64, 128, 256],
            hidden_sizes=[768, 512, 256, 64, 16, 8],  # More hidden layers
            num_classes=2,  # Number of output classes
        ).to(device)

        # Test with input
        test_input = torch.randn(1, 1, 670, 250, 250).to(device)
        output = model(test_input)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Input: {test_input.shape} -> Output: {output.shape}")

        # Test attention module separately to see intermediate sizes
        encoder_out = test_input
        for conv_block in model.encoder:
            encoder_out = conv_block(encoder_out)

        attention_out = model.attention_pool(encoder_out)
        print(f"Encoder output: {encoder_out.shape}")
        print(f"Attention output: {attention_out.shape}")

        break  # Just test one for brevity