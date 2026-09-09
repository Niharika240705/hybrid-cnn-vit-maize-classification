import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MaizeDataset, get_transforms
from src.model import HybridCNNViTModel

def calculate_metrics(preds, targets):
    """
    Computes binary classification metrics (Class 1 = Good, Class 0 = Bad).
    """
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    
    tp = torch.sum((preds == 1) & (targets == 1)).item()
    fp = torch.sum((preds == 1) & (targets == 0)).item()
    tn = torch.sum((preds == 0) & (targets == 0)).item()
    fn = torch.sum((preds == 0) & (targets == 1)).item()
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

def train_phase2(mode="variety_pretrained", fold=None, resume=False, start_epoch=1):
    # 1. Load Configurations
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("mps" if torch.backends.mps.is_available() and config['device'] == "mps" else "cpu")
    print(f"\n--- Running Phase 2 | Mode: {mode} | Fold: {fold} | Device: {device} ---")
    
    # Establish save path
    run_name = f"{mode}_fold_{fold}" if fold is not None else f"{mode}_split_stratified"
    save_dir = os.path.join(config['phase2']['checkpoint_dir'], run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # TensorBoard setup
    tb_writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    # 2. Set up Ablation Flags
    use_cnn = True
    use_vit = True
    if mode == "cnn_only":
        use_vit = False
    elif mode == "vit_only":
        use_cnn = False

    # 3. Load Datasets & DataLoaders
    train_transform, val_transform = get_transforms(image_size=config['image_size'])
    
    train_dataset = MaizeDataset(
        root_dir=config['dataset_b_dir'],
        label_mode="quality",
        split="train",
        seed=42,
        fold=fold,
        num_folds=config['phase2']['cv_folds'],
        transform=train_transform
    )
    
    val_dataset = MaizeDataset(
        root_dir=config['dataset_b_dir'],
        label_mode="quality",
        split="val",
        seed=42,
        fold=fold,
        num_folds=config['phase2']['cv_folds'],
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['phase2']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['phase2']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Dataset B: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # 4. Instantiate Model
    # Determine pretraining weights option
    pretrained_imagenet = True
    if mode == "from_scratch":
        pretrained_imagenet = False
        
    model = HybridCNNViTModel(
        cnn_backbone=config['cnn_backbone'],
        vit_backbone=config['vit_backbone'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_classes=2,  # Binary Classification
        pretrained=pretrained_imagenet,
        use_cnn=use_cnn,
        use_vit=use_vit
    )

    if resume:
        resume_path = os.path.join(save_dir, "model_best.pth")
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Cannot resume: checkpoint not found at {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"Resuming from checkpoint: {resume_path} at epoch {start_epoch}")
    
    # 5. Load Variety Pretrained Weights if requested
    if mode in ["variety_pretrained", "cnn_only", "vit_only"]:
        backbone_weights_path = os.path.join(config['phase1']['checkpoint_dir'], "backbone_best.pth")
        if os.path.exists(backbone_weights_path):
            print(f"Loading maize variety pretrained weights from {backbone_weights_path}...")
            # Load state dict
            state_dict = torch.load(backbone_weights_path, map_location=device)
            # Filter state dict keys based on ablation
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if not use_cnn and k.startswith("cnn_branch."):
                    continue
                if not use_vit and k.startswith("vit_branch."):
                    continue
                if not use_cnn and k.startswith("fusion.proj_cnn"):
                    continue
                if not use_vit and k.startswith("fusion.proj_vit"):
                    continue
                filtered_state_dict[k] = v
                
            msg = model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Load status: {msg}")
        else:
            print(f"WARNING: Pretrained weights not found at {backbone_weights_path}. Training from ImageNet/scratch weights.")
            
    model.to(device)

    # 6. Set up Phased Optimization
    # Phase A: Freeze backbone, train only the head
    model.freeze_backbone()
    
    criterion = nn.CrossEntropyLoss()
    
    # Set up optimizer with classifier parameters only
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=config['phase2']['lr_head'], 
        weight_decay=config['phase2']['weight_decay']
    )
    
    epochs = config['phase2']['epochs']
    frozen_epochs = config['phase2']['frozen_epochs']
    best_val_f1 = 0.0
    backbone_unfrozen = False

    # 7. Training Loop
    for epoch in range(start_epoch, epochs + 1):
        # Unfreeze backbone after frozen_epochs
        if epoch > frozen_epochs and not backbone_unfrozen:
            print("Unfreezing backbone layers for full fine-tuning...")
            model.unfreeze_backbone()
            # Recreate optimizer with all parameters at a lower learning rate
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=config['phase2']['lr_finetune'], 
                weight_decay=config['phase2']['weight_decay']
            )
            backbone_unfrozen = True

        # --- Training ---
        model.train()
        running_loss = 0.0
        train_preds = []
        train_targets = []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
        train_metrics = calculate_metrics(train_preds, train_targets)
        train_loss = running_loss / len(train_dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
        val_metrics = calculate_metrics(val_preds, val_targets)
        epoch_val_loss = val_loss / len(val_dataset)
        
        # Log to Console
        print(f"Epoch [{epoch}/{epochs}] "
              f"| Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.2f}% F1: {train_metrics['f1']:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} Acc: {val_metrics['accuracy']:.2f}% F1: {val_metrics['f1']:.2f}%")
        
        # Log to TensorBoard
        tb_writer.add_scalar("Loss/Train", train_loss, epoch)
        tb_writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
        tb_writer.add_scalar("Accuracy/Train", train_metrics['accuracy'], epoch)
        tb_writer.add_scalar("Accuracy/Val", val_metrics['accuracy'], epoch)
        tb_writer.add_scalar("F1/Train", train_metrics['f1'], epoch)
        tb_writer.add_scalar("F1/Val", val_metrics['f1'], epoch)
        
        # Save best model checkpoints based on validation F1 score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            print(f"--> Saved best model checkpoint with Val F1: {best_val_f1:.2f}%")
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pth"))
            
    tb_writer.close()
    return best_val_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 2 Quality Finetuning")
    parser.add_option = parser.add_argument
    parser.add_argument("--mode", type=str, default="variety_pretrained", 
                        choices=["variety_pretrained", "imagenet_only", "from_scratch", "cnn_only", "vit_only"],
                        help="Pretraining mode and branch architectures")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index for cross-validation. Set to None to use 70/15/15 stratified split.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the saved best checkpoint instead of starting from ImageNet weights.")
    parser.add_argument("--start-epoch", type=int, default=1,
                        help="First epoch to run when resuming, inclusive.")
    
    args = parser.parse_args()
    train_phase2(mode=args.mode, fold=args.fold, resume=args.resume, start_epoch=args.start_epoch)
