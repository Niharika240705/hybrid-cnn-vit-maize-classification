import os
import pytest
import torch
import yaml
from src.dataset import MaizeDataset, get_transforms
from src.attention import CBAM, CrossAttention
from src.cnn_branch import CNNBranch
from src.vit_branch import ViTBranch
from src.fusion import MultiScaleCrossAttentionFusion
from src.model import HybridCNNViTModel

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

def test_dataset_a_loader():
    """
    Test loading Dataset A (Variety) and verifying shapes and class mapping.
    """
    train_transform, _ = get_transforms(image_size=config['image_size'])
    dataset = MaizeDataset(
        root_dir=config['dataset_a_dir'],
        label_mode="variety",
        split="train",
        seed=42,
        transform=train_transform
    )
    
    assert len(dataset) > 0
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert label in [0, 1, 2]
    print(f"\nDataset A sample successfully loaded. Image shape: {img.shape}, Label: {label}")

def test_dataset_b_loader():
    """
    Test loading Dataset B (Quality) and verifying shapes and class mapping.
    """
    _, val_transform = get_transforms(image_size=config['image_size'])
    dataset = MaizeDataset(
        root_dir=config['dataset_b_dir'],
        label_mode="quality",
        split="val",
        seed=42,
        transform=val_transform
    )
    
    assert len(dataset) > 0
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert label in [0, 1]
    print(f"Dataset B sample successfully loaded. Image shape: {img.shape}, Label: {label}")

def test_attention_blocks():
    """
    Test CBAM and CrossAttention with random tensors.
    """
    # 1. Test CBAM
    cbam = CBAM(gate_channels=64)
    x_cnn = torch.randn(2, 64, 28, 28)
    out_cbam = cbam(x_cnn)
    assert out_cbam.shape == x_cnn.shape
    
    # 2. Test CrossAttention
    cross_attn = CrossAttention(embed_dim=256, num_heads=4)
    x_q = torch.randn(2, 197, 256)
    x_kv = torch.randn(2, 833, 256)
    out_attn = cross_attn(x_q, x_kv)
    assert out_attn.shape == (2, 197, 256)
    print("\nAttention modules verified successfully.")

def test_cnn_branch():
    """
    Test intermediate feature extraction in CNNBranch.
    """
    cnn = CNNBranch(backbone_name=config['cnn_backbone'], pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    f2, f4 = cnn(x)
    
    # Verify expected channel dimensions (from resnet50 stages 2 and 4)
    assert f2.shape[0] == 2
    assert f2.shape[1] == cnn.stage2_channels
    assert f2.shape[2:] == (28, 28)
    
    assert f4.shape[0] == 2
    assert f4.shape[1] == cnn.stage4_channels
    assert f4.shape[2:] == (7, 7)
    print(f"\nCNN branch verified. Stage 2 channels: {f2.shape[1]}, Stage 4 channels: {f4.shape[1]}")

def test_vit_branch():
    """
    Test sequence feature extraction in ViTBranch.
    """
    vit = ViTBranch(backbone_name=config['vit_backbone'], pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    feat = vit(x)
    
    assert len(feat.shape) == 3
    assert feat.shape[0] == 2
    assert feat.shape[2] == vit.num_features
    print(f"\nViT branch verified. Sequence shape: {list(feat.shape)}")

def test_fusion_module():
    """
    Test multi-scale cross-attention fusion.
    """
    cnn = CNNBranch(backbone_name=config['cnn_backbone'], pretrained=False)
    vit = ViTBranch(backbone_name=config['vit_backbone'], pretrained=False)
    
    fusion = MultiScaleCrossAttentionFusion(
        cnn_stage2_channels=cnn.stage2_channels,
        cnn_stage4_channels=cnn.stage4_channels,
        vit_dim=vit.num_features,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads']
    )
    
    f2 = torch.randn(2, cnn.stage2_channels, 28, 28)
    f4 = torch.randn(2, cnn.stage4_channels, 7, 7)
    f_vit = torch.randn(2, 197, vit.num_features)
    
    fused = fusion(f2, f4, f_vit)
    assert fused.shape == (2, config['embed_dim'])
    print(f"\nFusion module verified. Output shape: {fused.shape}")

def test_full_model_and_freezing():
    """
    Test complete Hybrid model, classification logits, head swapping, and freezing.
    """
    model = HybridCNNViTModel(
        cnn_backbone=config['cnn_backbone'],
        vit_backbone=config['vit_backbone'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_classes=3,
        pretrained=False
    )
    
    x = torch.randn(2, 3, 224, 224)
    
    # 1. Check Phase A Variety Classifier (3 classes)
    out_a = model(x)
    assert out_a.shape == (2, 3)
    
    # 2. Check head swap to Phase B (2 classes)
    model.swap_head(num_classes=2)
    out_b = model(x)
    assert out_b.shape == (2, 2)
    
    # 3. Check freezing logic
    model.freeze_backbone()
    
    # Verify that backbone weights do not require grads
    for name, param in model.named_parameters():
        if not name.startswith("head."):
            assert param.requires_grad == False, f"Expected {name} to be frozen"
        else:
            assert param.requires_grad == True, f"Expected {name} to be trainable"
            
    # 4. Check selective unfreezing
    model.unfreeze_last_n_blocks(cnn_unfreeze=True, vit_unfreeze=True, fusion_unfreeze=True)
    
    # Ensure fusion parameters and classifier parameters are trainable
    for param in model.fusion.parameters():
        assert param.requires_grad == True
    for param in model.head.parameters():
        assert param.requires_grad == True
        
    print("\nFull model and freezing/unfreezing logic verified successfully.")
