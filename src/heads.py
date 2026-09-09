import torch.nn as nn

class MaizeClassifierHead(nn.Module):
    """
    Swappable classification head for variety (3 classes) or quality (2 classes) tasks.
    """
    def __init__(self, embed_dim=256, num_classes=2, dropout=0.2):
        super(MaizeClassifierHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: fused representation tensor [B, embed_dim]
        Returns:
            logits: [B, num_classes]
        """
        return self.classifier(x)
