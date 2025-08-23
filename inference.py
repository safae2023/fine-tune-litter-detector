import os
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import argparse


# ------------------------
# Argument parsing
parser = argparse.ArgumentParser(description="Run object detection inference.")
parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
parser.add_argument('--input', type=str, required=True, help='Path to input image')
parser.add_argument('--out', type=str, default=None, help='Output folder (default: same as input)')
parser.add_argument('--thresh', type=float, default=0.7, help='Detection threshold')
parser.add_argument('--labels', type=str, default="labels.txt", help='Path to labels file')
args = parser.parse_args()

# ------------------------
# Labels (index 0 = background)
with open(args.labels, "r", encoding="utf-8") as f:
    label_list = [line.strip() for line in f if line.strip()]
num_classes = len(label_list) +1 # 6

# ------------------------
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ------------------------
# Build the model (must match the checkpoint's num_classes)
model = models.detection.fasterrcnn_resnet50_fpn(
    weights=None,              
    weights_backbone=None,
    num_classes=num_classes
)

ckpt_path = args.ckpt
state = torch.load(ckpt_path, map_location=device)

# Some training scripts save {"model": state_dict, ...}
if isinstance(state, dict) and "roi_heads.box_predictor.cls_score.weight" not in state:
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]

model.load_state_dict(state, strict=True)
model.to(device).eval()

# ------------------------
img_path = args.input
image_bgr = cv2.imread(img_path)
if image_bgr is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)

transform = transforms.Compose([transforms.ToTensor()])  # -> CxHxW, float[0..1]
image_tensor = transform(image_pil).to(device)

# ------------------------
# Inference (note the list input)
with torch.no_grad():
    preds = model([image_tensor])  # list in, list out
pred = preds[0]

boxes  = pred["boxes"]
labels = pred["labels"]
scores = pred["scores"]

# ------------------------
threshold = args.thresh
vis = image_bgr.copy()
H, W = vis.shape[:2]

for i in range(len(boxes)):
    s = float(scores[i].item())
    if s < threshold:
        continue

    x1, y1, x2, y2 = boxes[i].to("cpu").numpy().astype(int).tolist()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    cls_id = int(labels[i].item())
    name = label_list[cls_id] if 0 <= cls_id < len(label_list) else f"id_{cls_id}"

    text = f"{name}: {s:.2f}"
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    y_text = y1 - 8 if y1 - 8 > 10 else y1 + 20
    cv2.putText(vis, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

# ------------------------
# Show (and optionally save)
plt.figure(figsize=(14, 10))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detections")
plt.show()

# Save result
if args.out:
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, os.path.splitext(os.path.basename(img_path))[0] + "_pred.jpg")
else:
    out_path = os.path.splitext(img_path)[0] + "_pred.jpg"
cv2.imwrite(out_path, vis)
print(f"Saved: {out_path}")
