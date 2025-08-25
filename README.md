# Fine-Tuning a Faster R-CNN for litter detection 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive object detection pipeline for waste classification using Faster R-CNN with ResNet-50 backbone. This project fine-tunes pre-trained models on the TACO dataset to detect 60 different types of recyclable waste objects (bottles, papers, cans, etc.) in images.

## Features

- **Faster R-CNN Model**: State-of-the-art object detection with ResNet-50 backbone
- **TACO Dataset Support**: Optimized for the TACO waste dataset with 60 categories
- **Data Augmentation**: Comprehensive augmentation pipeline with bounding box transformation
- **Multiple Export Formats**: Support for PyTorch, ONNX, and OpenVINO models
- **Training Pipeline**: Complete training workflow with validation and evaluation
- **Inference Tools**: Ready-to-use inference scripts for different model formats

## Table of Contents

- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Model Export](#model-export)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/fine-tune-litter-detector.git
cd fine-tune litter detector
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import torch, torchvision, cv2, albumentations; print('All packages installed successfully!')"
```

## Dataset Setup

### TACO Dataset Structure

The pipeline handles the TACO dataset's unique batch organization:
- **15 batch directories** (batch_1 to batch_15)
- **COCO format annotations** with 60 waste categories
- **Automatic image discovery** across batch directories
- **Smart data splitting** (train/val/test)

### Required Directory Structure

```
data/
├── annotations_train.json
├── annotations_val.json
├── annotations.json
├── all_image_urls.csv
├── annotations_unofficial.json
├── batch_1/
├── batch_2/
└── ... (batch_15)
```

### Data Augmentation

Comprehensive augmentation strategies:
- **Geometric**: Horizontal/vertical flips, rotations, scaling
- **Photometric**: Brightness, contrast, gamma adjustments
- **Spatial**: Shift, scale, rotate transformations

The augmentation pipeline creates a folder containing both original and augmented datasets, along with an `annotations_augmented.json` file that includes all images with transformed bounding boxes.

```
augmented_datasets/
├── batch_1/
│   ├── 000000_original.jpg
│   ├── 000000_aug_1.jpg      ← Has transformed bbox in JSON
│   ├── 000000_aug_2.jpg      ← Has transformed bbox in JSON
│   └── 000001_original.jpg
├── batch_2/
│   └── ...
└── annotations_augmented.json  ← COMPLETE annotation file
```

## Usage

### Data Augmentation

Run the augmentation script to create enhanced datasets:

```bash
python augment_all_datasets.py
```

This will process all images in batch folders and generate augmented versions with proper annotation files.

### Training

Start the training process with custom parameters:

```bash
python train/train.py \
    --dataset data \
    --output trained_model \
    --epochs 50 \
    --batch-size-train 2 \
    --batch-size-val 1 \
    --lr 0.005 \
    --device auto
```

**Training Arguments:**
- `--dataset`: Path to dataset folder
- `--output`: Output folder for models
- `--epochs`: Number of training epochs
- `--batch-size-train`: Training batch size
- `--batch-size-val`: Validation batch size
- `--lr`: Learning rate
- `--device`: Device (cpu/gpu/auto)

### Inference

#### PyTorch Model Inference

```bash
python inference.py \
    --ckpt trained_model/best_model.pth \
    --input path/to/image.jpg \
    --out outputs \
    --thresh 0.7 \
    --labels labels.txt
```

#### ONNX Model Inference

```bash
python onnx/onnx_inference.py \
    --onnx Fast_RCNN_detection_model.onnx \
    --input path/to/image.jpg \
    --out outputs/onnx_result.jpg \
    --thresh 0.7 \
    --labels labels.txt
```

#### OpenVINO Model Inference

```bash
python openvino/infer_openvino_frcnn.py \
    --xml Fast_RCNN_detection_model.xml \
    --input path/to/image.jpg \
    --labels labels.txt \
    --out result.jpg \
    --thresh 0.7 \
    --device AUTO
```

## Evaluation Metrics

The training pipeline provides comprehensive evaluation metrics:

- **mAP (mean Average Precision)**: Primary detection accuracy metric
- **IoU-based Detection**: Intersection over Union threshold evaluation
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of correctly detected positive instances
- **AR (Average Recall)**: Average recall across different IoU thresholds

## Model Export

### ONNX Export

Export your trained model to ONNX format:

```bash
python onnx/onnx_export.py
```

### OpenVINO Export

Convert to OpenVINO format for optimized inference:

```bash
python openvino/openvino_export.py
```

## Project Structure

```
fine-tune litter detector/
├── train/                      # Training scripts
│   ├── train.py               # Main training script
│   ├── engine.py              # Training engine
│   ├── transforms.py          # Data transformations
│   ├── utils.py               # Utility functions
│   ├── coco_eval.py          # COCO evaluation
│   └── coco_utils.py         # COCO utilities
├── onnx/                      # ONNX export and inference
│   ├── onnx_export.py        # ONNX model export
│   └── onnx_inference.py     # ONNX model inference
├── openvino/                  # OpenVINO optimization
│   ├── openvino_export.py    # OpenVINO export
│   └── infer_openvino_frcnn.py # OpenVINO inference
├── augment_all_datasets.py    # Data augmentation script
├── inference.py               # PyTorch model inference
├── requirements.txt           # Python dependencies
├── labels.txt                 # Class labels
└── README.md                  # This file
```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **TACO Dataset**: [Pedro Proença et al.](https://github.com/pedropro/TACO)
- **PyTorch**: [Facebook Research](https://pytorch.org/)
- **TorchVision**: [Facebook Research](https://pytorch.org/vision/)
- **OpenVINO**: [Intel](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)

## Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/fine-tune-litter-detector/issues) page
2. Create a new issue with detailed information
3. Include your environment details and error messages

---




