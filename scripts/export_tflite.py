import argparse
import os
import subprocess
import sys
from pathlib import Path

import onnx
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import HybridCNNViTModel
from src.mobile_model import MobileCNNModel


def load_config(config_path):
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def select_device(config):
    if config.get("device") == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config, checkpoint_path, device, mobile=False):
    if mobile:
        model = MobileCNNModel(config["cnn_backbone"], config["embed_dim"], num_classes=2, pretrained=False)
    else:
        model = HybridCNNViTModel(
            cnn_backbone=config["cnn_backbone"],
            vit_backbone=config["vit_backbone"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_classes=2,
            pretrained=False,
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def export_onnx(model, output_path, image_size, device, opset, dynamic_batch=True):
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_options = {
        "input_names": ["image"],
        "output_names": ["logits"],
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamo": False,
    }
    if dynamic_batch:
        export_options["dynamic_axes"] = {
            "image": {0: "batch"},
            "logits": {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            **export_options,
        )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)


def validate_onnx(output_path, image_size):
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: np.zeros((1, 3, image_size, image_size), dtype=np.float32)})
    if output[0].shape != (1, 2):
        raise RuntimeError(f"Unexpected ONNX output shape: {output[0].shape}")


def convert_to_tflite(onnx_path, output_dir, quantize_int8=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(output_dir)]
    if quantize_int8:
        command.append("-oiqt")
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Export the fine-tuned maize model.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "src" / "config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "phase2" / "variety_pretrained_split_stratified" / "model_best.pth",
    )
    parser.add_argument("--onnx", type=Path, default=PROJECT_ROOT / "exports" / "maize_classifier.onnx")
    parser.add_argument("--tflite-dir", type=Path, default=PROJECT_ROOT / "exports" / "tflite")
    parser.add_argument("--convert-tflite", action="store_true")
    parser.add_argument("--int8", action="store_true", help="Request onnx2tf integer quantization.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--static-batch", action="store_true", help="Export a fixed batch-1 graph for mobile conversion.")
    parser.add_argument("--mobile", action="store_true", help="Export the TFLite-compatible CNN-only mobile checkpoint.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train Phase 2 first or pass --checkpoint."
        )

    config = load_config(args.config)
    device = select_device(config)
    model = build_model(config, args.checkpoint, device, mobile=args.mobile)
    export_onnx(model, args.onnx, config["image_size"], device, args.opset, dynamic_batch=not args.static_batch)
    validate_onnx(args.onnx, config["image_size"])
    print(f"ONNX export verified: {args.onnx} ({args.onnx.stat().st_size} bytes)")

    if args.convert_tflite:
        convert_to_tflite(args.onnx, args.tflite_dir, args.int8)
        print(f"TFLite conversion completed in: {args.tflite_dir}")


if __name__ == "__main__":
    main()