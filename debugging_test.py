from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def test_image(image_path, conf_threshold=0.001):
    """Test image detection with direct model access"""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert to RGB (YOLO expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load model
    model_path = "app/models/best.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = YOLO(model_path)

    # Run prediction with extremely low confidence
    results = model.predict(
        source=img_rgb,
        conf=conf_threshold,
        iou=0.45,
        max_det=100,  # Detect many objects
        verbose=True,
        save=True,  # Save results
        save_txt=True  # Save text results
    )

    # Process results
    print(f"\nDetection Results (confidence >= {conf_threshold}):")
    print("-" * 50)

    if not results or len(results) == 0:
        print("No results returned!")
        return

    result = results[0]
    if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
        print("No objects detected!")
        return

    # Print all detections
    boxes = result.boxes
    if len(boxes) == 0:
        print("No boxes found in results")
        return

    print(f"Found {len(boxes)} potential detections:")

    # Sort by confidence (highest first)
    confidences = [float(box.conf[0]) for box in boxes]
    sorted_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)

    # Print all detections
    for i, idx in enumerate(sorted_indices):
        box = boxes[idx]
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        cls_name = model.names.get(cls, f"Unknown-{cls}")

        if hasattr(box, 'xyxy'):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bbox = f"({x1}, {y1}), ({x2}, {y2})"
        else:
            bbox = "Unknown"

        print(f"  {i+1}. Class: {cls_name}, Confidence: {conf:.6f}, BBox: {bbox}")

    # The results should already be saved, but let's also save an annotated version
    annotated_img = result.plot()
    cv2.imwrite(f"debug_result_{os.path.basename(image_path)}", cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    print(f"\nAnnotated image saved as 'debug_result_{os.path.basename(image_path)}'")

    return results

def main():
    """Main function to test multiple images"""
    print("=== Direct Model Testing ===")

    # Test first few images in Dataset directory
    dataset_dir = Path("Dataset")
    image_files = list(dataset_dir.glob("*.jpeg"))[:5]  # Test first 5 images

    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Testing {img_path}")
        test_image(str(img_path), conf_threshold=0.0001)  # Use extremely low threshold

    print("\nTesting complete.")

if __name__ == "__main__":
    main()