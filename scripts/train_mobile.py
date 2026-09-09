import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MaizeDataset, get_transforms
from src.mobile_model import MobileCNNModel


def main():
    with (PROJECT_ROOT / "src" / "config.yaml").open() as config_file:
        config = yaml.safe_load(config_file)
    device = torch.device("mps" if config["device"] == "mps" and torch.backends.mps.is_available() else "cpu")
    output_dir = PROJECT_ROOT / "checkpoints" / "mobile"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_transform, val_transform = get_transforms(config["image_size"])
    train_set = MaizeDataset(config["dataset_b_dir"], "quality", "train", transform=train_transform)
    val_set = MaizeDataset(config["dataset_b_dir"], "quality", "val", transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    model = MobileCNNModel(config["cnn_backbone"], config["embed_dim"], pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_f1 = -1.0

    for epoch in range(1, 6):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        true_positive = false_positive = false_negative = correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                predictions = model(images.to(device)).argmax(1).cpu()
                true_positive += int(((predictions == 1) & (labels == 1)).sum())
                false_positive += int(((predictions == 1) & (labels == 0)).sum())
                false_negative += int(((predictions == 0) & (labels == 1)).sum())
                correct += int((predictions == labels).sum())
                total += len(labels)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        print(f"Epoch {epoch}/5 | Accuracy {100 * correct / total:.2f}% | F1 {100 * f1:.2f}%")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), output_dir / "model_best.pth")


if __name__ == "__main__":
    main()