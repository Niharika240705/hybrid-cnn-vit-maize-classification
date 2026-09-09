import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Channel Attention Module of CBAM.
    Applies both average and max pooling along spatial dimensions, projects via a shared MLP,
    and returns channel weights.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module of CBAM.
    Applies average and max pooling along the channel dimension, concatenates them,
    and applies a 7x7 convolution to generate spatial attention maps.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_concat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines Channel Attention and Spatial Attention sequentially.
    """
    def __init__(self, gate_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(gate_channels, ratio=reduction_ratio)
        self.sa = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out

class CrossAttention(nn.Module):
    """
    Multi-head Cross-Attention block.
    Computes attention between query tensor `x` and key/value tensor `y`.
    Input shapes:
        x (query): [B, L_q, D]
        y (key/value): [B, L_kv, D]
    Output shape:
        fused: [B, L_q, D]
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, y):
        """
        Args:
            x (Tensor): Query tensor [B, L_q, D]
            y (Tensor): Key/Value tensor [B, L_kv, D]
        """
        # Cross-Attention: queries from x, keys and values from y
        attn_out, _ = self.mha(query=x, key=y, value=y)
        x = self.norm(x + self.dropout(attn_out))
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
