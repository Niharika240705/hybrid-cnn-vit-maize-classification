import torch
import torch.nn as nn
import timm

class ViTBranch(nn.Module):
    """
    Vision Transformer (ViT) Branch of the Hybrid model.
    Wraps a timm ViT/Swin model and returns sequence tokens.
    """
    def __init__(self, backbone_name="vit_small_patch16_224", pretrained=True):
        super(ViTBranch, self).__init__()
        
        # Load the ViT/Swin backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained
        )
        
        # Extract features dimension
        self.num_features = self.backbone.num_features

    def forward(self, x):
        # Extract features (patch tokens + class token if available)
        features = self.backbone.forward_features(x)
        
        # Standardize features output shape to [B, L, D_vit]
        if len(features.shape) == 4:
            # Hierarchical ViT (e.g. Swin) returns [B, H, W, D_vit] -> flatten spatial
            b, h, w, d = features.shape
            features = features.view(b, h * w, d)
            
        # Standard ViT returns [B, N, D_vit] directly
        return features
