
#!/usr/bin/env python3
"""
Simple ONNX Export Script
=========================
Converts trained Faster R-CNN model to ONNX format.
"""

import torch
import torchvision
import onnx
import os

def export_model():
    """Export trained model to ONNX"""
    
    # 1. Load trained model
    print("Loading your trained model...")
    model_path = "model/best.pth"  # Change this to the path of the best model.ckpt
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please update the model_path variable with your best model file")
        return
    
    # 2. Create model architecture 
    print("Creating model architecture...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        num_classes=61  # 60 waste categories + 1 background
    )
    
    # 3. Load the trained weights
    print("Loading trained weights...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    
    # 4. Export to ONNX
    print("Exporting to ONNX...")
    output_path = "Fast_RCNN_detection_model.onnx"
    
    # Create dummy input (any size will work since the model doesn't resize)
    dummy_input = torch.randn(1, 3, 800, 800)
    
    # Export

    torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["boxes", "labels", "scores"],
    dynamic_axes={
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'boxes': {0: 'batch'},
        'labels': {0: 'batch'},
        'scores': {0: 'batch'}
    }
)

    print(f" Model exported successfully to: {output_path}")
    # 5. Validate the ONNX model
    print("Validating ONNX model...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
    except Exception as e:
        print(f" ONNX validation failed: {e}")
        return
    
    # 6. Create simple config file
    config = {
        "num_classes": 61,
        "use_normalization": False,
        "use_resize": False,
        "confidence_threshold": 0.5
    }
    
    import json
    with open("model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(" Model config saved to: modele_config.json")
    print(" Export completed! You can now use your ONNX model.")

if __name__ == "__main__":
    export_model()
