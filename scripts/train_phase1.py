import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MaizeDataset, get_transforms
from src.model import HybridCNNViTModel

def train_phase1():
    # 1. Load Configurations
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("mps" if torch.backends.mps.is_available() and config['device'] == "mps" else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directories
    os.makedirs(config['phase1']['checkpoint_dir'], exist_ok=True)
    
    # Set up TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(config['phase1']['checkpoint_dir'], "logs"))

    # 2. Load Datasets & DataLoaders
    train_transform, val_transform = get_transforms(image_size=config['image_size'])
    
    train_dataset = MaizeDataset(
        root_dir=config['dataset_a_dir'],
        label_mode="variety",
        split="train",
        seed=42,
        transform=train_transform
    )
    
    val_dataset = MaizeDataset(
        root_dir=config['dataset_a_dir'],
        label_mode="variety",
        split="val",
        seed=42,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['phase1']['batch_size'],
        shuffle=True,
        num_workers=0,  # 0 is safer for MPS to avoid multiprocessing fork issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['phase1']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Dataset A: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # 3. Instantiate Model
    model = HybridCNNViTModel(
        cnn_backbone=config['cnn_backbone'],
        vit_backbone=config['vit_backbone'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_classes=3,  # 3 variety classes
        pretrained=True
    )
    model.to(device)

    # 4. Optimizer, Loss & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['phase1']['lr'], 
        weight_decay=config['phase1']['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['phase1']['epochs']
    )

    # 5. Training Loop
    epochs = config['phase1']['epochs']
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = (correct / total) * 100
        
        # Step the scheduler
        scheduler.step()
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        
        print(f"Epoch [{epoch}/{epochs}] "
              f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        
        # Log to TensorBoard
        tb_writer.add_scalar("Loss/Train", train_loss, epoch)
        tb_writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
        tb_writer.add_scalar("Accuracy/Train", train_acc, epoch)
        tb_writer.add_scalar("Accuracy/Val", epoch_val_acc, epoch)
        
        # Save checkpoints
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f"--> Saved best variety model checkpoint with Accuracy: {best_val_acc:.2f}%")
            
            # Save full model
            full_model_path = os.path.join(config['phase1']['checkpoint_dir'], "model_best.pth")
            torch.save(model.state_dict(), full_model_path)
            
            # Save backbone separately for Phase 2 fine-tuning
            backbone_state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("head.")}
            backbone_path = os.path.join(config['phase1']['checkpoint_dir'], "backbone_best.pth")
            torch.save(backbone_state_dict, backbone_path)
            
    tb_writer.close()
    print("Pretraining Phase 1 finished successfully!")

if __name__ == "__main__":
    train_phase1()
