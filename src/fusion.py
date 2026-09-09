import torch
import torch.nn as nn
from src.attention import CrossAttention

class MultiScaleCrossAttentionFusion(nn.Module):
    """
    Multi-Scale Cross-Attention Fusion module.
    Fuses intermediate CNN features at two scales (Stage 2 and Stage 4) with ViT tokens.
    """
    def __init__(self, cnn_stage2_channels, cnn_stage4_channels, vit_dim, 
                 embed_dim=256, num_heads=8, dropout=0.1):
        super(MultiScaleCrossAttentionFusion, self).__init__()
        
        # 1x1 convolutions to project CNN feature maps to embed_dim
        self.proj_cnn_stage2 = nn.Conv2d(cnn_stage2_channels, embed_dim, kernel_size=1)
        self.proj_cnn_stage4 = nn.Conv2d(cnn_stage4_channels, embed_dim, kernel_size=1)
        
        # Linear layer to project ViT tokens to embed_dim
        self.proj_vit = nn.Linear(vit_dim, embed_dim)
        
        # Cross-Attention module (ViT tokens query CNN features)
        self.cross_attn = CrossAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, feat_stage2, feat_stage4, feat_vit):
        """
        Args:
            feat_stage2: mid-scale CNN features [B, C_stage2, H1, W1] (typically 28x28)
            feat_stage4: high-scale CNN features [B, C_stage4, H2, W2] (typically 7x7)
            feat_vit: ViT patch sequence tokens [B, N_vit, D_vit]
        """
        B = feat_stage2.size(0)
        
        # 1. Project and flatten CNN features
        # Stage 2: [B, embed_dim, H1, W1] -> [B, H1*W1, embed_dim]
        proj_s2 = self.proj_cnn_stage2(feat_stage2)
        proj_s2 = proj_s2.flatten(2).transpose(1, 2)
        
        # Stage 4: [B, embed_dim, H2, W2] -> [B, H2*W2, embed_dim]
        proj_s4 = self.proj_cnn_stage4(feat_stage4)
        proj_s4 = proj_s4.flatten(2).transpose(1, 2)
        
        # Combine multi-scale CNN tokens: [B, (H1*W1 + H2*W2), embed_dim]
        feat_cnn_combined = torch.cat([proj_s2, proj_s4], dim=1)
        
        # 2. Project ViT tokens to embed_dim: [B, N_vit, embed_dim]
        proj_vit = self.proj_vit(feat_vit)
        
        # 3. Cross-Attention: ViT tokens (query) attend to combined CNN features (key/value)
        fused_seq = self.cross_attn(proj_vit, feat_cnn_combined)
        
        # 4. Global average pooling to obtain a single fused feature vector [B, embed_dim]
        fused_vector = torch.mean(fused_seq, dim=1)
        
        return fused_vector
