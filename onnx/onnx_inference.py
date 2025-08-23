import onnx
import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser(description="ONNX Inference Script")
parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model file')
parser.add_argument('--input', type=str, required=True, help='Path to input image')
parser.add_argument('--out', type=str, default="onnx_inference_result.jpg", help='Path to save output image')
parser.add_argument('--thresh', type=float, default=0.5, help='Detection threshold')
parser.add_argument('--labels', type=str, default="labels.txt", help='Path to labels file')
args = parser.parse_args()

with open(args.labels, "r", encoding="utf-8") as f:
    label_list = [line.strip() for line in f if line.strip()]
num_classes = len(label_list)
print(f"Number of classes: {num_classes}")

# --- Load ONNX model ---
onnx_path = args.onnx
if not os.path.exists(onnx_path):
    raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")
print(f"Loading ONNX model: {onnx_path}")
session = ort.InferenceSession(onnx_path)
print("ONNX model loaded")

# --- Optionally validate the model ---
onnx_model = onnx.load(onnx_path)
try:
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated")
except Exception as e:
    print("ONNX model validation failed:", e)

# --- Load and preprocess image ---
img_path = args.input
if not os.path.isfile(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

image_bgr = cv2.imread(img_path)
if image_bgr is None:
    raise ValueError(f"Failed to read image: {img_path}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)

transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image_pil).unsqueeze(0)
image_np = image_tensor.numpy()
print(f"Input tensor shape: {image_tensor.shape}")

# --- Run ONNX inference ---
print("Running inference...")
input_name = session.get_inputs()[0].name
# Assuming export defined outputs as ["boxes", "labels", "scores"]
outputs = session.run(None, {input_name: image_np})
boxes, labels, scores = outputs  # unpacking outputs

# --- Filter and visualize detections ---
threshold = args.thresh
for box, label_idx, score in zip(boxes, labels.astype(int), scores):
    if score < threshold:
        continue
    x1, y1, x2, y2 = map(int, box)
    label = label_list[label_idx] if label_idx < len(label_list) else f"Unknown_{label_idx}"
    text = f"{label}: {score:.2f}"
    cv2.putText(image_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

# --- Display result ---
image_rgb_display = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb_display)
plt.title(f"Detections above threshold {threshold}")
plt.axis('off')
plt.show()

save_path = args.out
cv2.imwrite(save_path, image_bgr)
print(f"Detection result saved to {save_path}")
print("Boxes:", boxes)
print("Labels:", labels)
print("Scores:", scores)
