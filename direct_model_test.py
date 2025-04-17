#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import logging
import time
from pathlib import Path
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path=None):
    """
    Load YOLOv8 model directly, bypassing any custom preprocessing
    """
    if model_path is None:
        # Default model path
        model_path = os.path.join("app", "models", "best.pt")

    model_path = os.path.abspath(model_path)
    logger.info(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None

    try:
        # Log model file size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model file size: {model_size_mb:.2f} MB")

        # Load model with very low confidence threshold
        model = YOLO(model_path)
        logger.info(f"Model loaded successfully: {type(model).__name__}")

        # Check model classes if available
        if hasattr(model, 'names'):
            logger.info("Model classes:")
            for idx, name in model.names.items():
                logger.info(f"  Class {idx}: {name}")

        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_direct_model(image_path, conf_threshold=0.01):
    """
    Test direct model inference with original image, no preprocessing except resize
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # Log original image info
        logger.info(f"Image: {image_path}, shape: {image.shape}")

        # Convert BGR to RGB for YOLO
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load model
        model = load_model()
        if model is None:
            logger.error("Failed to load model")
            return None

        # Run inference with very low confidence threshold
        logger.info(f"Running direct inference with confidence threshold: {conf_threshold}")
        start_time = time.time()

        # Feed image directly to model
        results = model.predict(
            source=rgb_image,
            conf=conf_threshold,
            verbose=False
        )

        inference_time = time.time() - start_time
        logger.info(f"Inference time: {inference_time:.2f} seconds")

        # Process all detections without filtering
        all_detections = []

        if not results or len(results) == 0:
            logger.info("No detections found")
            return {
                "image_path": image_path,
                "inference_time": inference_time,
                "detections": []
            }

        result = results[0]

        # Check if there are any boxes
        if not hasattr(result, 'boxes') or len(result.boxes) == 0:
            logger.info("No boxes found")
            return {
                "image_path": image_path,
                "inference_time": inference_time,
                "detections": []
            }

        # Process all boxes without filtering
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        # Log raw detection count
        logger.info(f"Found {len(boxes)} raw detections")

        # Save image with detections
        output_img = rgb_image.copy()

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            # Get coordinates
            x1, y1, x2, y2 = map(int, box)

            # Calculate metrics for debugging
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0

            img_area = image.shape[0] * image.shape[1]
            box_area = width * height
            area_percentage = (box_area / img_area) * 100

            # Log detection info
            logger.info(f"Detection #{i+1}: conf={conf:.4f}, coords=[{x1},{y1},{x2},{y2}], "
                       f"aspect={aspect_ratio:.2f}, area={area_percentage:.2f}%")

            # Add detection to list
            detection = {
                "confidence": float(conf),
                "box": [int(x) for x in [x1, y1, x2, y2]],
                "aspect_ratio": float(aspect_ratio),
                "area_percentage": float(area_percentage),
                "would_pass_normal_filters": {
                    "aspect_ratio": bool(1.0 <= aspect_ratio <= 8.0),
                    "area_percentage": bool(0.1 <= area_percentage <= 60.0),
                    "confidence": bool(conf >= 0.2)
                }
            }
            all_detections.append(detection)

            # Draw box on image (green if would pass filters, red if not)
            if all(detection["would_pass_normal_filters"].values()):
                color = (0, 255, 0)  # Green
                thickness = 2
            else:
                color = (255, 0, 0)  # Red
                thickness = 1

            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness)

            # Add confidence text
            cv2.putText(
                output_img,
                f"{i+1}: {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        # Sort detections by confidence
        all_detections.sort(key=lambda x: x["confidence"], reverse=True)

        # Save annotated image
        output_filename = f"direct_{Path(image_path).stem}.jpg"
        cv2.imwrite(output_filename, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved annotated image to {output_filename}")

        # Check if there's a high confidence detection (>=0.5)
        high_conf_detections = [d for d in all_detections if d["confidence"] >= 0.5]
        if high_conf_detections:
            logger.info(f"Found {len(high_conf_detections)} high confidence detection(s)")
        else:
            logger.info("No high confidence detections")

        # Count detections that would pass normal filters
        would_pass = sum(1 for d in all_detections if all(d["would_pass_normal_filters"].values()))
        logger.info(f"Detections that would pass normal filters: {would_pass}/{len(all_detections)}")

        return {
            "image_path": image_path,
            "inference_time": inference_time,
            "detections": all_detections,
            "annotated_image": output_filename,
            "high_confidence_count": len(high_conf_detections),
            "normal_filter_pass_count": would_pass
        }

    except Exception as e:
        logger.error(f"Error in direct model test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_multiple_images(image_paths=None):
    """
    Test multiple images with direct model inference
    """
    if not image_paths:
        # Use default images from the Dataset folder
        dataset_dir = Path("Dataset")
        image_paths = [str(p) for p in list(dataset_dir.glob("*.jpeg"))[:10]]

    if not image_paths:
        logger.error("No images to process")
        return

    results = []
    success_count = 0

    for path in image_paths:
        logger.info(f"\n=== Testing image: {path} ===")
        result = test_direct_model(path)

        if result:
            results.append(result)
            if result.get("high_confidence_count", 0) > 0:
                success_count += 1
                logger.info(f"High confidence detection found")

    # Print summary
    total = len(image_paths)
    success_rate = success_count / total * 100 if total > 0 else 0
    logger.info(f"\n=== Summary ===")
    logger.info(f"Tested {total} images")
    logger.info(f"High confidence detections: {success_count}/{total} ({success_rate:.1f}%)")

    # Save results to JSON file
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open("direct_model_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Results saved to direct_model_results.json")

    return results

def main():
    """
    Main entry point
    """
    # Check if image paths are provided
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        # Use default paths
        image_paths = None

    # Run tests
    test_multiple_images(image_paths)

if __name__ == "__main__":
    main()