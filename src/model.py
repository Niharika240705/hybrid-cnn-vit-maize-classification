import torch
import torch.nn as nn
from src.cnn_branch import CNNBranch
from src.vit_branch import ViTBranch
from src.fusion import MultiScaleCrossAttentionFusion
from src.heads import MaizeClassifierHead

class HybridCNNViTModel(nn.Module):
    """
    Hybrid CNN-Vision Transformer model for Maize classification.
    Processes input image through CNN and ViT branches, fuses features using
    Multi-scale Cross-Attention, and outputs class logits.
    """
    def __init__(self, cnn_backbone="resnet50", vit_backbone="vit_small_patch16_224",
                 embed_dim=256, num_heads=8, num_classes=2, pretrained=True, dropout=0.2,
                 use_cnn=True, use_vit=True):
        super(HybridCNNViTModel, self).__init__()
        
        self.cnn_backbone_name = cnn_backbone
        self.vit_backbone_name = vit_backbone
        self.embed_dim = embed_dim
        self.use_cnn = use_cnn
        self.use_vit = use_vit
        
        # 1. Instantiate the branches based on config
        if self.use_cnn:
            self.cnn_branch = CNNBranch(cnn_backbone, pretrained=pretrained)
        if self.use_vit:
            self.vit_branch = ViTBranch(vit_backbone, pretrained=pretrained)
        
        # 2. Instantiate Fusion or Projection Module
        if self.use_cnn and self.use_vit:
            self.fusion = MultiScaleCrossAttentionFusion(
                cnn_stage2_channels=self.cnn_branch.stage2_channels,
                cnn_stage4_channels=self.cnn_branch.stage4_channels,
                vit_dim=self.vit_branch.num_features,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        elif self.use_cnn:
            # CNN-only projection and pooling
            self.proj_cnn = nn.Conv2d(self.cnn_branch.stage4_channels, embed_dim, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif self.use_vit:
            # ViT-only projection
            self.proj_vit = nn.Linear(self.vit_branch.num_features, embed_dim)
        else:
            raise ValueError("At least one of use_cnn or use_vit must be True.")
        
        # 3. Instantiate Classification Head
        self.head = MaizeClassifierHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x):
        if self.use_cnn and self.use_vit:
            # Extract CNN features
            feat_stage2, feat_stage4 = self.cnn_branch(x)
            # Extract ViT features
            feat_vit = self.vit_branch(x)
            # Fuse CNN and ViT representations
            fused = self.fusion(feat_stage2, feat_stage4, feat_vit)
        elif self.use_cnn:
            # CNN-only forward
            _, feat_stage4 = self.cnn_branch(x)
            fused = self.proj_cnn(feat_stage4)
            fused = self.pool(fused).squeeze(-1).squeeze(-1)
        else:
            # ViT-only forward
            feat_vit = self.vit_branch(x)
            fused = self.proj_vit(feat_vit)
            fused = torch.mean(fused, dim=1)
            
        # Classify
        logits = self.head(fused)
        return logits

    def swap_head(self, num_classes, dropout=0.2):
        """
        Replaces the classification head with a new one (e.g. switching from 3 classes to 2).
        """
        self.head = MaizeClassifierHead(
            embed_dim=self.embed_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def freeze_backbone(self):
        """
        Freezes all parameters in the CNN, ViT, and Fusion branches.
        Only the classifier head will train.
        """
        if self.use_cnn:
            for param in self.cnn_branch.parameters():
                param.requires_grad = False
        if self.use_vit:
            for param in self.vit_branch.parameters():
                param.requires_grad = False
        if hasattr(self, 'fusion'):
            for param in self.fusion.parameters():
                param.requires_grad = False
        if hasattr(self, 'proj_cnn'):
            for param in self.proj_cnn.parameters():
                param.requires_grad = False
        if hasattr(self, 'proj_vit'):
            for param in self.proj_vit.parameters():
                param.requires_grad = False
            
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """
        Unfreezes all parameters in the network.
        """
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, cnn_unfreeze=True, vit_unfreeze=True, fusion_unfreeze=True):
        """
        Selectively unfreezes components for fine-tuning.
        """
        if self.use_cnn and cnn_unfreeze:
            # Unfreeze only the final stages of the CNN branch (Stage 4 features + CBAM)
            for name, param in self.cnn_branch.backbone.named_parameters():
                if "layer4" in name or "stages.4" in name:
                    param.requires_grad = True
            for param in self.cnn_branch.cbam_stage4.parameters():
                param.requires_grad = True
            if hasattr(self, 'proj_cnn'):
                for param in self.proj_cnn.parameters():
                    param.requires_grad = True
                
        if self.use_vit and vit_unfreeze:
            # Unfreeze the last few blocks of the ViT transformer
            if hasattr(self.vit_branch.backbone, 'blocks'):
                # Standard ViT
                num_blocks = len(self.vit_branch.backbone.blocks)
                # Unfreeze last 2 blocks
                for idx in range(max(0, num_blocks - 2), num_blocks):
                    for param in self.vit_branch.backbone.blocks[idx].parameters():
                        param.requires_grad = True
            elif hasattr(self.vit_branch.backbone, 'layers'):
                # Swin
                num_layers = len(self.vit_branch.backbone.layers)
                for idx in range(max(0, num_layers - 1), num_layers):
                    for param in self.vit_branch.backbone.layers[idx].parameters():
                        param.requires_grad = True
            if hasattr(self, 'proj_vit'):
                for param in self.proj_vit.parameters():
                    param.requires_grad = True
                        
        if hasattr(self, 'fusion') and fusion_unfreeze:
            # Keep the fusion parameters trainable
            for param in self.fusion.parameters():
                param.requires_grad = True
