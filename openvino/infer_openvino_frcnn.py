#!/usr/bin/env python3
"""
OpenVINO inference for Faster R-CNN exported from pth -> ONNX -> IR.

"""

import argparse
from pathlib import Path
import sys

import numpy as np
import cv2
from PIL import Image
import openvino as ov


def parse_args():
    ap = argparse.ArgumentParser(description="OpenVINO Faster R-CNN Inference")
    ap.add_argument("--xml", required=True, help="Path to OpenVINO IR .xml file")
    ap.add_argument("--input", required=True, help="Path to input image")
    ap.add_argument("--labels", default="labels.txt", help="Path to labels file")
    ap.add_argument("--out", default="ov_inference_result.jpg", help="Output image path")
    ap.add_argument("--thresh", type=float, default=0.5, help="Score threshold")
    ap.add_argument("--device", default="AUTO", help="Device (e.g. AUTO, CPU, GPU)")
    return ap.parse_args()


def load_labels(path):
    p = Path(path)
    if not p.exists():
        print(f"[WARN] labels file not found: {p}. Proceeding with numeric labels.")
        return None
    names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(names) == 0 or names[0] != "__background__":
        names = ["__background__"] + names
    return names


def preprocess(image_path):
    """Read image -> RGB -> float32 [0,1] -> NCHW."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # ToTensor() equivalent
    arr = np.asarray(Image.fromarray(rgb)).astype(np.float32) / 255.0  # HWC, [0,1]
    nchw = np.transpose(arr, (0, 1, 2))  # still HWC, but explicit
    nchw = np.transpose(nchw, (2, 0, 1))  # CHW
    nchw = np.expand_dims(nchw, 0)       # NCHW
    return bgr, nchw


def squeeze01(x):
    """Squeeze leading batch dim if present (1, N, ...)."""
    x = np.array(x)
    if x.ndim >= 2 and x.shape[0] == 1:
        return np.squeeze(x, axis=0)
    return x


def main():
    args = parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        print(f"IR model not found: {xml_path}")
        sys.exit(1)

    # ---- OpenVINO: load + compile ----
    core = ov.Core()
    # You can either pass the path directly to compile_model, or call read_model first.
    compiled = ov.compile_model(str(xml_path), device_name=args.device)  # AUTO by default

    # Input name (robust to renaming)
    input_port = compiled.inputs[0]  # single input
    input_name = input_port.any_name

    # Output ports and their .any_name (we’ll match by the names we set during ONNX export)
    out_ports = compiled.outputs
    out_names = {p.any_name: p for p in out_ports}

    # ---- Preprocess ----
    image_bgr, input_blob = preprocess(args.input)

    # If IR was built for a fixed size, and the image is a different size,
    # OpenVINO will usually reshape dynamically if the model allows it. If not,
    # resize image here to the model's expected HxW.
    # (TorchVision FRCNN typically handles variable sizes, but ONNX export often
    # requires fixed H/W; adjust if needed.)

    # ---- Inference ----
    results = compiled({input_name: input_blob})

    # fetch by names first (named outputs ["boxes","labels","scores"])
    boxes = results.get(out_names.get("boxes"), None)
    labels = results.get(out_names.get("labels"), None)
    scores = results.get(out_names.get("scores"), None)

    # Fallback: if names didn’t survive conversion, assume order [boxes, labels, scores]
    if boxes is None or labels is None or scores is None:
        ordered = [results[p] for p in out_ports]
        if len(ordered) != 3:
            raise RuntimeError(f"Expected 3 outputs, got {len(ordered)}: {[p.any_name for p in out_ports]}")
        # Identify by simple heuristics
        cands = sorted(
            enumerate(ordered),
            key=lambda kv: (
                -1 if (ordered[kv[0]].ndim >= 2 and ordered[kv[0]].shape[-1] == 4) else 0,
                -ordered[kv[0]].size
            ),
        )
        boxes = ordered[cands[0][0]]
        remaining = [ordered[i] for i in range(3) if i != cands[0][0]]
        # labels are integer-like; scores are float
        if np.issubdtype(remaining[0].dtype, np.integer):
            labels, scores = remaining[0], remaining[1]
        else:
            scores, labels = remaining[0], remaining[1]

    boxes = squeeze01(boxes)
    labels = squeeze01(labels).astype(np.int64)
    scores = squeeze01(scores).astype(np.float32)

    # ---- Draw detections ----
    h, w = image_bgr.shape[:2]
    label_names = load_labels(args.labels)
    thr = float(args.thresh)

    keep = scores >= thr
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    for box, lab, sc in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        name = str(lab)
        if label_names is not None and 0 <= lab < len(label_names):
            name = label_names[lab]

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{name}: {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 255), -1)
        cv2.putText(image_bgr, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = Path(args.out)
    cv2.imwrite(str(out_path), image_bgr)
    print(f" Saved: {out_path}")
    print(f"Detections kept (score ≥ {thr:.2f}): {len(boxes)}")


if __name__ == "__main__":
    main()
