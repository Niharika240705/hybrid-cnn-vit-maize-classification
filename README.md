# A Hybrid CNN–Vision Transformer Architecture for Fine-Grained Maize Seed Defect Detection and Quality Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Flutter](https://img.shields.io/badge/Flutter-3.0+-02569B.svg)](https://flutter.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end research prototype and mobile deployment system combining Convolutional Neural Networks (ResNet / EfficientNetV2) and Vision Transformers (ViT / Swin-Tiny) via a Multi-Scale Cross-Attention Fusion mechanism for fine-grained maize seed variety and quality classification.

---

## 🌽 Key Highlights & Architecture

- **Dual-Branch Hybrid Backbone**:
  - **CNN Branch**: Captures high-resolution local spatial features, micro-textures, and fine seed edge patterns (enhanced with CBAM attention).
  - **Vision Transformer Branch**: Captures long-range spatial dependencies and global morphological context across seed structures.
  - **Multi-Scale Cross-Attention Fusion**: Interactively aligns and weights tokens between CNN feature maps and Transformer sequence outputs using query-key attention.
- **Two-Phase Progressive Transfer Learning**:
  - **Phase 1 (Variety Classification)**: Pre-trained on multi-cultivar seed datasets to learn fine-grained agronomic representation spaces.
  - **Phase 2 (Quality & Defect Classification)**: Fine-tuned with frozen backbone warm-up and stratified 5-fold cross-validation on target quality datasets.
- **Edge & Mobile Deployment**:
  - Distilled mobile backbone (`MobileCNNBranch` / ResNet / MobileNetV3) with streamlined classification heads.
  - PyTorch $\to$ ONNX $\to$ TensorFlow Lite quantization pipeline (`scripts/export_tflite.py`).
  - Cross-platform Flutter mobile application (`mobile/`) supporting real-time camera capture, gallery picking, and offline on-device TFLite inference.
- **Serving & Demo Suite**:
  - RESTful inference API built with **FastAPI** (`app/api.py`).
  - Interactive web application built with **Streamlit** (`app/demo.py`).

---

## 📁 Repository Structure

```plaintext
.
├── app/
│   ├── api.py                    # FastAPI REST service for cloud / server inference
│   └── demo.py                   # Streamlit interactive web demonstration
├── mobile/                       # Flutter offline mobile application
│   ├── lib/
│   │   ├── classifier.dart       # TFLite interpreter wrapper & preprocessing
│   │   ├── home_screen.dart      # Material UI, camera & gallery interaction
│   │   └── main.dart             # Flutter entry point
│   ├── assets/models/            # Bundled TFLite model for offline inference
│   └── pubspec.yaml              # Flutter dependencies and asset registrations
├── scripts/
│   ├── evaluate.py               # Evaluation script with multi-metric reports
│   ├── export_tflite.py          # PyTorch -> ONNX -> TFLite model conversion
│   ├── train_mobile.py           # Mobile-optimized backbone training
│   ├── train_phase1.py           # Phase 1: Pretraining on variety classification
│   └── train_phase2.py           # Phase 2: Fine-tuning on quality classification (5-fold CV)
├── src/
│   ├── attention.py              # CBAM and Multi-Head Cross-Attention modules
│   ├── cnn_branch.py             # CNN feature extraction with attention
│   ├── config.yaml               # Centralized hyperparameters & paths
│   ├── dataset.py                # Albumentations augmentations & PyTorch Datasets
│   ├── fusion.py                 # Multi-scale cross-attention fusion layer
│   ├── heads.py                  # Classification heads
│   ├── mobile_model.py           # Mobile-friendly lightweight network
│   ├── model.py                  # Full Hybrid CNN-ViT Architecture
│   └── vit_branch.py             # Vision Transformer branch
├── tests/
│   └── test_components.py        # PyTest suite validating tensor dimensions & forward passes
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 or higher
- PyTorch 2.0+ (with CUDA or Apple Silicon MPS support)
- Flutter 3.0+ (optional, for mobile app development)

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verification

Run the test suite to verify component tensor dimensions, attention layers, and forward passes:

```bash
pytest tests/test_components.py -v
```

---

## 🏋️ Training & Evaluation Pipeline

### Phase 1: Variety Classification Pretraining
Trains the hybrid CNN-ViT backbone on multi-class variety datasets:
```bash
python scripts/train_phase1.py
```

### Phase 2: Quality Classification Fine-Tuning
Fine-tunes the pretrained backbone using 5-fold cross-validation with an initial frozen warmup:
```bash
python scripts/train_phase2.py
```

### Evaluation
Evaluates the model on test partitions and generates classification metrics:
```bash
python scripts/evaluate.py --checkpoint checkpoints/phase2/best_model.pt
```

---

## 📲 Mobile Deployment (Flutter & TFLite)

### Export to TFLite
Convert the PyTorch model checkpoint into an optimized, offline TensorFlow Lite model:
```bash
python scripts/export_tflite.py --checkpoint checkpoints/phase2/best_model.pt --output mobile/assets/models/maize_classifier.tflite
```

### Run the Mobile Application
```bash
cd mobile
flutter pub get
flutter run
```

---

## 🌐 Web API & Demo

### Launch FastAPI Server
```bash
uvicorn app.api:app --reload --port 8000
```
API Documentation will be available at `http://127.0.0.1:8000/docs`.

### Launch Streamlit Demo
```bash
streamlit run app/demo.py
```

---

## 🔬 Citation & Acknowledgments

If you find this codebase helpful in your research or application, please cite:

```bibtex
@misc{maize_hybrid_cnn_vit_2026,
  title={A Hybrid CNN--Vision Transformer Architecture for Fine-Grained Maize Seed Defect Detection and Quality Classification},
  author={Singh, Niharika},
  year={2026}
}
```
