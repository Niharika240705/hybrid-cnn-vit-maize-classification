import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MaizeDataset, get_transforms
from src.model import HybridCNNViTModel
from scripts.train_phase2 import calculate_metrics

def evaluate_model(mode="variety_pretrained", fold=None, config=None, device=None):
    """
    Evaluates a single model configuration and returns its metrics.
    """
    run_name = f"{mode}_fold_{fold}" if fold is not None else f"{mode}_split_stratified"
    model_path = os.path.join(config['phase2']['checkpoint_dir'], run_name, "model_best.pth")
    
    if not os.path.exists(model_path):
        print(f"Skipping evaluation for {run_name} (No checkpoint found at {model_path})")
        return None
        
    # Set up ablation flags
    use_cnn = True
    use_vit = True
    if mode == "cnn_only":
        use_vit = False
    elif mode == "vit_only":
        use_cnn = False
        
    # Load dataset
    _, val_transform = get_transforms(image_size=config['image_size'])
    test_dataset = MaizeDataset(
        root_dir=config['dataset_b_dir'],
        label_mode="quality",
        split="test",
        seed=42,
        fold=fold,
        num_folds=config['phase2']['cv_folds'],
        transform=val_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['phase2']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Load Model
    model = HybridCNNViTModel(
        cnn_backbone=config['cnn_backbone'],
        vit_backbone=config['vit_backbone'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_classes=2,
        pretrained=False,
        use_cnn=use_cnn,
        use_vit=use_vit
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
            
    metrics = calculate_metrics(val_preds, val_targets)
    return metrics

def run_all_evaluations():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("mps" if torch.backends.mps.is_available() and config['device'] == "mps" else "cpu")
    
    modes = ["variety_pretrained", "imagenet_only", "from_scratch", "cnn_only", "vit_only"]
    results = {}
    
    print("\n=======================================================")
    print("Evaluating all trained baselines and ablations...")
    print("=======================================================")
    
    for mode in modes:
        # Check if we ran with cross-validation or single split
        cv_dir = os.path.join(config['phase2']['checkpoint_dir'], f"{mode}_fold_0")
        if os.path.exists(cv_dir):
            # Calculate CV average across folds
            fold_accuracies = []
            fold_precisions = []
            fold_recalls = []
            fold_f1s = []
            
            for fold in range(config['phase2']['cv_folds']):
                m = evaluate_model(mode=mode, fold=fold, config=config, device=device)
                if m is not None:
                    fold_accuracies.append(m['accuracy'])
                    fold_precisions.append(m['precision'])
                    fold_recalls.append(m['recall'])
                    fold_f1s.append(m['f1'])
            
            if fold_accuracies:
                import numpy as np
                results[mode] = {
                    "accuracy": f"{np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}",
                    "precision": f"{np.mean(fold_precisions):.2f}%",
                    "recall": f"{np.mean(fold_recalls):.2f}%",
                    "f1": f"{np.mean(fold_f1s):.2f}%"
                }
        else:
            # Single stratified split evaluation
            m = evaluate_model(mode=mode, fold=None, config=config, device=device)
            if m is not None:
                results[mode] = {
                    "accuracy": f"{m['accuracy']:.2f}%",
                    "precision": f"{m['precision']:.2f}%",
                    "recall": f"{m['recall']:.2f}%",
                    "f1": f"{m['f1']:.2f}%"
                }
                
    # Print results markdown table
    print("\n### Final Evaluation Report & Headline Results")
    print("| Model Pipeline | Accuracy | Precision | Recall | F1-Score |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    mode_names = {
        "variety_pretrained": "Variety-Pretrained (Our Hybrid CNN-ViT)",
        "imagenet_only": "ImageNet-Only (No Variety Pretraining)",
        "from_scratch": "From Scratch (No Pretraining)",
        "cnn_only": "CNN-Only Ablation",
        "vit_only": "ViT-Only Ablation"
    }
    
    for mode in modes:
        if mode in results:
            name = mode_names[mode]
            r = results[mode]
            print(f"| {name} | {r['accuracy']} | {r['precision']} | {r['recall']} | {r['f1']} |")
            
    print("\n=======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Maize Seed Classifier Baselines")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["variety_pretrained", "imagenet_only", "from_scratch", "cnn_only", "vit_only"],
                        help="Specific mode to evaluate. If None, evaluates all modes and prints summary table.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index for specific cross validation fold evaluation.")
    
    args = parser.parse_args()
    
    if args.mode is None:
        run_all_evaluations()
    else:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        device = torch.device("mps" if torch.backends.mps.is_available() and config['device'] == "mps" else "cpu")
        m = evaluate_model(mode=args.mode, fold=args.fold, config=config, device=device)
        if m is not None:
            print(f"\nEvaluation Results for {args.mode} (Fold: {args.fold}):")
            print(f"  Accuracy:  {m['accuracy']:.2f}%")
            print(f"  Precision: {m['precision']:.2f}%")
            print(f"  Recall:    {m['recall']:.2f}%")
            print(f"  F1-Score:  {m['f1']:.2f}%")
            print(f"  Confusion Matrix: TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}")
