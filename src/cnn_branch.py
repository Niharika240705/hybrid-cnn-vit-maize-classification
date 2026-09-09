import torch
import torch.nn as nn
import timm
from src.attention import CBAM

class CNNBranch(nn.Module):
    """
    CNN Branch of the Hybrid model.
    Wraps a timm CNN model, extracts feature maps at Stage 2 and Stage 4,
    and applies CBAM attention.
    """
    def __init__(self, backbone_name="resnet50", pretrained=True):
        super(CNNBranch, self).__init__()
        
        # Load the feature extractor
        self.backbone = timm.create_model(
            backbone_name, 
            features_only=True, 
            pretrained=pretrained
        )
        
        # Get channel dimensions for each stage
        channels = self.backbone.feature_info.channels()
        self.stage2_channels = channels[2]
        self.stage4_channels = channels[4]
        
        # Instantiate CBAM for Stage 2 and Stage 4
        self.cbam_stage2 = CBAM(gate_channels=self.stage2_channels)
        self.cbam_stage4 = CBAM(gate_channels=self.stage4_channels)

    def forward(self, x):
        # Pass input through backbone to get intermediate features
        features = self.backbone(x)
        
        # Extract features for Stage 2 and Stage 4
        feat_stage2 = features[2]  # typically [B, C_stage2, 28, 28]
        feat_stage4 = features[4]  # typically [B, C_stage4, 7, 7]
        
        # Apply CBAM attention
        feat_stage2 = self.cbam_stage2(feat_stage2)
        feat_stage4 = self.cbam_stage4(feat_stage4)
        
        return feat_stage2, feat_stage4
