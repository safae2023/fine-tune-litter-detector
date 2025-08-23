# -*- coding: utf-8 -*-
"""
FINAL PROJECT - Object Detection with Faster R-CNN
==================================================

This project implements an object detection model based on Faster R-CNN
for waste detection in images.

"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================
import matplotlib.pyplot as plt
import cv2
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import argparse
import sys

# =============================================================================
# COMMAND LINE ARGUMENTS CONFIGURATION
# =============================================================================
def parse_arguments():
    """
    Parse command line arguments to configure training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Training a Faster R-CNN model for waste detection"
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Path to the dataset folder (with subfolders data/, annotations_train.json, annotations_val.json)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Output folder to save checkpoints and models'
    )
    
    # Optional arguments with default values
    parser.add_argument(
        '--device', 
        type=str, 
        choices=['cpu', 'gpu', 'auto'], 
        default='auto',
        help='Computing device: cpu, gpu, or auto (automatic detection) [default: auto]'
    )
    
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.005,
        help='Learning rate for optimizer [default: 0.005]'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='Number of training epochs [default: 50]'
    )
    
    parser.add_argument(
        '--batch-size-train', 
        type=int, 
        default=2,
        help='Batch size for training [default: 2]'
    )
    
    parser.add_argument(
        '--batch-size-val', 
        type=int, 
        default=1,
        help='Batch size for validation [default: 1]'
    )
    
    parser.add_argument(
        '--momentum', 
        type=float, 
        default=0.9,
        help='Momentum for SGD optimizer [default: 0.9]'
    )
    
    parser.add_argument(
        '--weight-decay', 
        type=float, 
        default=0.0005,
        help='Weight decay (L2 regularization) [default: 0.0005]'
    )
    
    parser.add_argument(
        '--print-freq', 
        type=int, 
        default=25,
        help='Frequency of displaying metrics during training [default: 25]'
    )
    
    parser.add_argument(
        '--save-freq', 
        type=int, 
        default=1,
        help='Frequency of saving models (every N epochs) [default: 1]'
    )
    
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Path to a checkpoint to resume training [default: None]'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Verbose mode with more debug information'
    )
    
    return parser.parse_args()

# =============================================================================
# ARGUMENT VALIDATION AND CONFIGURATION
# =============================================================================
def validate_and_setup_config(args):
    """
    Validates arguments and configures the training environment.
    
    Args:
        args: Parsed arguments
        
    Returns:
        dict: Validated configuration
    """
    config = {}
    
    # Dataset folder validation
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset folder '{args.dataset}' does not exist!")
        sys.exit(1)
    
    # Dataset structure verification
    required_files = [
        os.path.join(args.dataset, "annotations_train.json"),
        os.path.join(args.dataset, "annotations_val.json")
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required file missing: {file_path}")
            sys.exit(1)
    
    # Output folder creation
    os.makedirs(args.output, exist_ok=True)
    print(f"Output folder created/verified: {args.output}")
    
    # Device configuration
    if args.device == 'auto':
        if torch.cuda.is_available():
            config['device'] = torch.device("cuda")
            print("CUDA GPU detected and automatically activated")
        else:
            config['device'] = torch.device("cpu")
            print("GPU not available, using CPU")
    elif args.device == 'gpu':
        if torch.cuda.is_available():
            config['device'] = torch.device("cuda")
            print("CUDA GPU activated")
        else:
            print("ERROR: GPU requested but not available!")
            sys.exit(1)
    else:  # cpu
        config['device'] = torch.device("cpu")
        print("CPU activated")
    
    # Training parameters validation
    if args.lr <= 0:
        print("ERROR: Learning rate must be positive!")
        sys.exit(1)
    
    if args.epochs <= 0:
        print("ERROR: Number of epochs must be positive!")
        sys.exit(1)
    
    if args.batch_size_train <= 0 or args.batch_size_val <= 0:
        print("ERROR: Batch sizes must be positive!")
        sys.exit(1)
    
    # Configuration storage
    config.update({
        'dataset_path': args.dataset,
        'output_path': args.output,
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'batch_size_train': args.batch_size_train,
        'batch_size_val': args.batch_size_val,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'print_freq': args.print_freq,
        'save_freq': args.save_freq,
        'resume_path': args.resume,
        'verbose': args.verbose
    })
    
    # Configuration display
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Dataset: {config['dataset_path']}")
    print(f"Output: {config['output_path']}")
    print(f"Device: {config['device']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch Size (train/val): {config['batch_size_train']}/{config['batch_size_val']}")
    print(f"Momentum: {config['momentum']}")
    print(f"Weight Decay: {config['weight_decay']}")
    print(f"Display frequency: {config['print_freq']}")
    print(f"Save frequency: {config['save_freq']}")
    if config['resume_path']:
        print(f"Resume from: {config['resume_path']}")
    print("="*60)
    
    return config

# =============================================================================
# DEVICE CONFIGURATION (GPU/CPU)
# =============================================================================
# Note: Device configuration is now handled in validate_and_setup_config()

# =============================================================================
# CUSTOM COCO DATASET CLASS
# =============================================================================
class CocoDetectionDataset(Dataset):
    """
    Custom dataset for loading COCO data with annotations.
    
    This class inherits from torch.utils.data.Dataset and loads images
    with their corresponding annotations in COCO format.
    """
    
    def __init__(self, image_dir, annotation_path, transforms=None):
        """
        Initializes the dataset.
        
        Args:
            image_dir (str): Path to the folder containing images
            annotation_path (str): Path to the JSON annotation file
            transforms: Transformations to apply to images
        """
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)  # Loads COCO annotations
        self.image_ids = list(self.coco.imgs.keys())  # List of image IDs
        self.transforms = transforms

        print(f"Dataset loaded: {len(self.image_ids)} images, {len(self.coco.anns)} annotations")

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Loads an image and its annotations.
        
        Args:
            idx (int): Index of the image to load
            
        Returns:
            tuple: (image, target) where target contains the annotations
        """
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]

        # Builds the complete path to the image
        relative_path = image_info['file_name']
        image_path = os.path.join(self.image_dir, relative_path)

        # Loads and converts the image to RGB
        image = Image.open(image_path).convert("RGB")

        # Loads all annotations for this image
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # Prepares annotation fields
        boxes = []      # Bounding boxes
        labels = []     # Object classes
        areas = []      # Box areas
        iscrowds = []   # Crowd indicators

        for ann in annotations:
            # COCO format: bbox = [x, y, width, height]
            # Conversion to [x1, y1, x2, y2] format
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowds.append(ann.get('iscrowd', 0))

        # Handles the case where there are no annotations
        if len(boxes) == 0:
            # Creates empty tensors for images without objects
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            area = torch.zeros(0, dtype=torch.float32)
            iscrowd = torch.zeros(0, dtype=torch.int64)
        else:
            # Converts to PyTorch tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowds, dtype=torch.int64)

        # Target structure for training
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,  # Keeps as integer, not as tensor
            'area': area,
            'iscrowd': iscrowd
        }

        # Applies transformations (if specified)
        if self.transforms:
            image = self.transforms(image)

        return image, target

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_transform():
    """
    Defines the transformations to apply to images.
    
    Returns:
        ToTensor: PIL image -> PyTorch tensor transformation
    """
    return ToTensor()

def collate_fn(batch):
    """
    Custom collation function to handle variable length targets.
    
    Args:
        batch: Data batch
        
    Returns:
        tuple: Reorganized data
    """
    return tuple(zip(*batch))

# =============================================================================
# DATA PREPARATION
# =============================================================================
def prepare_datasets(config):
    """
    Prepares training and validation datasets.
    
    Args:
        config (dict): Training configuration
        
    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset)
    """
    print("Preparing datasets...")

    # Loads training dataset
    train_dataset = CocoDetectionDataset(
        image_dir=config['dataset_path'],
        annotation_path=os.path.join(config['dataset_path'], "annotations_train.json"),
        transforms=get_transform()
    )

    # Loads validation dataset
    val_dataset = CocoDetectionDataset(
        image_dir=config['dataset_path'],
        annotation_path=os.path.join(config['dataset_path'], "annotations_val.json"),
        transforms=get_transform()
    )

    # Creates DataLoaders with custom collation function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size_train'],
        shuffle=True,       # Shuffles the data
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size_val'],
        shuffle=False,      # No shuffling for validation
        collate_fn=collate_fn
    )

    print(f"Datasets prepared:")
    print(f"   - Training: {len(train_dataset)} images")
    print(f"   - Validation: {len(val_dataset)} images")
    print(f"   - Training batch size: {config['batch_size_train']}")
    print(f"   - Validation batch size: {config['batch_size_val']}")
    
    return train_loader, val_loader, train_dataset, val_dataset

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
def setup_model(config, train_dataset):
    """
    Configures the Faster R-CNN model.
    
    Args:
        config (dict): Training configuration
        train_dataset: Training dataset to get the number of classes
        
    Returns:
        torch.nn.Module: Configured model
    """
    print("\nConfiguring Faster R-CNN model...")

    # Loads a pre-trained Faster R-CNN model with ResNet50 backbone and FPN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    # Number of classes in the dataset (including background class)
    # +1 for background class
    num_classes = len(train_dataset.coco.getCatIds()) + 1
    print(f"Number of classes detected: {num_classes}")

    # Number of input features for the classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print(f"Input features: {in_features}")

    # Replaces the prediction head to adapt to the number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(f"Prediction head adapted for {num_classes} classes")

    # Moves the model to the configured device
    model.to(config['device'])
    print(f"Model moved to {config['device']}")
    
    return model

# =============================================================================
# OPTIMIZER CONFIGURATION
# =============================================================================
def setup_optimizer(model, config):
    """
    Configures the optimizer for training.
    
    Args:
        model: Model to train
        config (dict): Training configuration
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    print("\nConfiguring optimizer...")

    # Gets parameters that require gradients (trainable parameters)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(params)}")

    # Defines SGD optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(
        params, 
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    print(f"SGD optimizer configured (lr={config['learning_rate']}, momentum={config['momentum']}, weight_decay={config['weight_decay']})")
    
    return optimizer

# =============================================================================
# DATASET VALIDATION BEFORE TRAINING
# =============================================================================
def validate_datasets(train_loader, val_loader):
    """
    Validates datasets before training.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
    """
    print("\nValidating dataset before training...")
    try:
        # Tests a few samples from both datasets
        print("Testing training dataset...")
        for i in range(min(3, len(train_loader.dataset))):
            sample_img, sample_target = train_loader.dataset[i]
            print(f"   Training sample {i}: image shape {sample_img.size()}, target keys {sample_target.keys()}")

        print("Testing validation dataset...")
        for i in range(min(3, len(val_loader.dataset))):
            sample_img, sample_target = val_loader.dataset[i]
            print(f"   Validation sample {i}: image shape {sample_img.size()}, target keys {sample_target.keys()}")

        print("Dataset validation successful!")
    except Exception as e:
        print(f"Dataset validation failed: {str(e)}")
        raise e

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def train_model(model, optimizer, train_loader, val_loader, config):
    """
    Trains the model with the specified configuration.
    
    Args:
        model: Model to train
        optimizer: Configured optimizer
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config (dict): Training configuration
        
    Returns:
        dict: Training results
    """
    print("\nStarting training...")
    print("=" * 80)

    # Tracks the best mAP for model saving
    best_map = 0.0
    best_epoch = 0
    training_history = []

    # Main training loop
    for epoch in range(config['num_epochs']):
        print(f"\nEPOCH {epoch + 1}/{config['num_epochs']}")
        print("=" * 60)

        try:
            # =============================================================
            # TRAINING PHASE
            # =============================================================
            print("Training phase...")
            train_metrics = train_one_epoch(
                model, optimizer, train_loader, config['device'], 
                epoch, config['print_freq']
            )
            
            # Displays training metrics
            print(f"\nTraining metrics for epoch {epoch + 1}:")
            print(f"   Total loss: {train_metrics.meters['loss'].global_avg:.4f}")
            
            # Displays detailed losses if available
            if 'loss_classifier' in train_metrics.meters:
                print(f"   Classifier loss: {train_metrics.meters['loss_classifier'].global_avg:.4f}")
            if 'loss_box_reg' in train_metrics.meters:
                print(f"   Box regression loss: {train_metrics.meters['loss_box_reg'].global_avg:.4f}")
            if 'loss_objectness' in train_metrics.meters:
                print(f"   Objectness loss: {train_metrics.meters['loss_objectness'].global_avg:.4f}")
            if 'loss_rpn_box_reg' in train_metrics.meters:
                print(f"   RPN regression loss: {train_metrics.meters['loss_rpn_box_reg'].global_avg:.4f}")

            # =============================================================
            # EVALUATION PHASE AND MAP CALCULATION
            # =============================================================
            print("\nEvaluation phase...")
            coco_evaluator = evaluate(model, val_loader, device=config['device'])
            
            # Extracts mAP metrics from COCO evaluator
            print(f"\nValidation mAP metrics for epoch {epoch + 1}:")
            
            # Gets bbox evaluation (main metric for object detection)
            bbox_eval = coco_evaluator.coco_eval['bbox']
            
            # Displays detailed mAP metrics
            print(f"   mAP @ IoU=0.50:0.95: {bbox_eval.stats[0]:.4f}")
            print(f"   mAP @ IoU=0.50: {bbox_eval.stats[1]:.4f}")
            print(f"   mAP @ IoU=0.75: {bbox_eval.stats[2]:.4f}")
            print(f"   mAP @ IoU=0.50:0.95 (small): {bbox_eval.stats[3]:.4f}")
            print(f"   mAP @ IoU=0.50:0.95 (medium): {bbox_eval.stats[4]:.4f}")
            print(f"   mAP @ IoU=0.50:0.95 (large): {bbox_eval.stats[5]:.4f}")
            print(f"   AR @ IoU=0.50:0.95: {bbox_eval.stats[6]:.4f}")
            print(f"   AR @ IoU=0.50: {bbox_eval.stats[7]:.4f}")
            print(f"   AR @ IoU=0.75: {bbox_eval.stats[8]:.4f}")
            print(f"   AR @ IoU=0.50:0.95 (small): {bbox_eval.stats[9]:.4f}")
            print(f"   AR @ IoU=0.50:0.95 (medium): {bbox_eval.stats[10]:.4f}")
            print(f"   AR @ IoU=0.50:0.95 (large): {bbox_eval.stats[11]:.4f}")
            
            # Gets the main mAP metric (IoU=0.50:0.95)
            current_map = bbox_eval.stats[0]
            
            # Stores training history
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_metrics.meters['loss'].global_avg,
                'val_map': current_map,
                'val_map_50': bbox_eval.stats[1],
                'val_map_75': bbox_eval.stats[2]
            }
            training_history.append(epoch_results)
            
            # =============================================================
            # MODEL SAVING
            # =============================================================
            # Saves the model according to the configured frequency
            if (epoch + 1) % config['save_freq'] == 0:
                model_path = os.path.join(config['output_path'], f"model_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"\nModel saved in {model_path}")
            
            # Checks if this is the best model so far
            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch + 1
                best_model_path = os.path.join(
                    config['output_path'], 
                    f"best_model_epoch_{epoch + 1}_map_{current_map:.4f}.pth"
                )
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model! mAP: {current_map:.4f}")
                print(f"Best model saved in {best_model_path}")
            
            print(f"\nCurrent best mAP: {best_map:.4f} (epoch {best_epoch})")
            print("=" * 60)

        except Exception as e:
            print(f"Error during epoch {epoch + 1}: {str(e)}")
            print("Moving to next epoch...")
            continue

    # Saves training history
    history_path = os.path.join(config['output_path'], 'training_history.json')
    import json
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"\nTraining history saved in {history_path}")

    return {
        'best_map': best_map,
        'best_epoch': best_epoch,
        'training_history': training_history
    }

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """
    Main function of the training script.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Validates and configures the environment
    config = validate_and_setup_config(args)
    
    # Prepares datasets
    train_loader, val_loader, train_dataset, val_dataset = prepare_datasets(config)
    
    # Configures the model
    model = setup_model(config, train_dataset)
    
    # Configures the optimizer
    optimizer = setup_optimizer(model, config)
    
    # Validates datasets
    validate_datasets(train_loader, val_loader)
    
    # Trains the model
    results = train_model(model, optimizer, train_loader, val_loader, config)
    
    # Displays final summary
    print(f"\nTRAINING COMPLETED!")
    print(f"Best mAP achieved: {results['best_map']:.4f} at epoch {results['best_epoch']}")
    print(f"Best model saved in folder: {config['output_path']}")
    print("=" * 80)

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()