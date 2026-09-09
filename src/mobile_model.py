import torch.nn as nn

from src.cnn_branch import CNNBranch
from src.heads import MaizeClassifierHead


class MobileCNNModel(nn.Module):
    """CNN-only quality classifier used for TFLite mobile deployment."""

    def __init__(self, cnn_backbone="resnet50", embed_dim=256, num_classes=2, pretrained=False, dropout=0.2):
        super().__init__()
        self.cnn_branch = CNNBranch(cnn_backbone, pretrained=pretrained)
        self.proj = nn.Conv2d(self.cnn_branch.stage4_channels, embed_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = MaizeClassifierHead(embed_dim, num_classes, dropout)

    def forward(self, x):
        _, features = self.cnn_branch(x)
        features = self.pool(self.proj(features)).flatten(1)
        return self.head(features)