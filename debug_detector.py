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

def minimal_preprocess(image, target_size=1024):
    """
    Minimal preprocessing - just resize while preserving aspect ratio
    """
    # Store original dimensions
    original_height, original_width = image.shape[:2]
    logger.info(f"Original image shape: {image.shape}")

    # Convert BGR to RGB (YOLOv8 expects RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image  # Already RGB or grayscale

    # Calculate scale to maintain aspect ratio
    scale = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize image
    resized = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a square canvas with black padding
    square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Center the resized image on the square canvas
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    square_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

    logger.info(f"Preprocessed image shape: {square_img.shape}")

    # Create transform parameters dictionary for coordinate mapping
    transform_params = {
        "scale": scale,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "original_size": (original_height, original_width)
    }

    return square_img, transform_params

def map_coordinates_to_original(box, transform_params):
    """
    Map coordinates from processed image space back to original image space
    """
    try:
        if not transform_params:
            logger.error("Missing transform parameters for coordinate mapping")
            return box

        scale = transform_params["scale"]
        x_offset = transform_params["x_offset"]
        y_offset = transform_params["y_offset"]

        # Unpack box coordinates
        x1, y1, x2, y2 = box

        # Remove offset (account for padding in the square canvas)
        x1 = (x1 - x_offset) if x1 >= x_offset else 0
        y1 = (y1 - y_offset) if y1 >= y_offset else 0
        x2 = (x2 - x_offset) if x2 >= x_offset else 0
        y2 = (y2 - y_offset) if y2 >= y_offset else 0

        # Reverse the scaling
        if scale > 0:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

        # Ensure coordinates are within original image bounds
        original_height, original_width = transform_params["original_size"]
        x1 = max(0, min(x1, original_width - 1))
        y1 = max(0, min(y1, original_height - 1))
        x2 = max(0, min(x2, original_width - 1))
        y2 = max(0, min(y2, original_height - 1))

        return [x1, y1, x2, y2]

    except Exception as e:
        logger.error(f"Error mapping coordinates: {str(e)}")
        return box

def detect_with_debug(image_path, conf_threshold=0.01):
    """
    Debug detection function with minimal preprocessing and filtering
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # Only perform minimal preprocessing
        processed_img, transform_params = minimal_preprocess(image)

        # Load model
        model = load_model()
        if model is None:
            logger.error("Failed to load model")
            return None

        logger.info(f"Running inference with confidence threshold: {conf_threshold}")

        # Run inference with very low confidence threshold
        start_time = time.time()
        results = model.predict(
            source=processed_img,
            conf=conf_threshold,
            verbose=False
        )
        inference_time = time.time() - start_time

        # Process results without filtering
        all_detections = []

        if not results or len(results) == 0:
            logger.info("No detections found")
            return {
                "image_path": image_path,
                "inference_time": inference_time,
                "detections": all_detections
            }

        result = results[0]

        # Check if there are any boxes
        if not hasattr(result, 'boxes') or len(result.boxes) == 0:
            logger.info("No boxes found")
            return {
                "image_path": image_path,
                "inference_time": inference_time,
                "detections": all_detections
            }

        # Process all boxes without filtering
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        logger.info(f"Found {len(boxes)} raw detections")

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            # Get coordinates
            x1, y1, x2, y2 = map(int, box)

            # Map to original image space
            orig_coords = map_coordinates_to_original([x1, y1, x2, y2], transform_params)
            ox1, oy1, ox2, oy2 = orig_coords

            # Calculate metrics for debugging
            width = ox2 - ox1
            height = oy2 - oy1
            aspect_ratio = width / height if height > 0 else 0

            img_area = image.shape[0] * image.shape[1]
            box_area = width * height
            area_percentage = (box_area / img_area) * 100

            # Log all detection info for debugging
            logger.info(f"Detection #{i+1}: conf={conf:.4f}, coords=[{ox1},{oy1},{ox2},{oy2}], "
                      f"aspect={aspect_ratio:.2f}, area={area_percentage:.2f}%")

            # Add to detections list without any filtering
            detection = {
                "confidence": float(conf),
                "box": [int(coord) for coord in [ox1, oy1, ox2, oy2]],
                "aspect_ratio": float(aspect_ratio),
                "area_percentage": float(area_percentage),
                "would_pass_normal_filters": {
                    "aspect_ratio": bool(1.0 <= aspect_ratio <= 8.0),
                    "area_percentage": bool(0.1 <= area_percentage <= 60.0),
                    "confidence": bool(conf >= 0.2)
                }
            }
            all_detections.append(detection)

        # Sort detections by confidence
        all_detections.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "image_path": image_path,
            "inference_time": inference_time,
            "detections": all_detections
        }

    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def visualize_results(image_path, detection_results):
    """
    Visualize detection results with different colors based on filter status
    """
    if not detection_results or not detection_results.get("detections"):
        logger.info(f"No detections to visualize for {image_path}")
        return

    # Load image for visualization
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image for visualization: {image_path}")
        return

    # Convert to RGB for matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))

    # Display the image
    ax.imshow(rgb_image)

    # Draw all detections with different colors
    for i, detection in enumerate(detection_results["detections"]):
        box = detection["box"]
        conf = detection["confidence"]
        filters = detection["would_pass_normal_filters"]

        # Determine box color
        if all(filters.values()):
            color = 'g'  # Green if would pass all normal filters
            linestyle = '-'
        else:
            color = 'r'  # Red if would be filtered out normally
            linestyle = '--'

        # Create rectangle
        rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor=color, facecolor='none', linestyle=linestyle
        )

        # Add rectangle to plot
        ax.add_patch(rect)

        # Add text annotation
        ax.text(
            box[0], box[1]-10,
            f"{i+1}: conf={conf:.2f}, ar={detection['aspect_ratio']:.1f}, area={detection['area_percentage']:.1f}%",
            color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.5)
        )

    # Set title and remove axes
    ax.set_title(f"Detection Results - {os.path.basename(image_path)}")
    ax.axis('off')

    # Save visualization
    output_filename = f"debug_{os.path.basename(image_path)}_vis.jpg"
    plt.savefig(output_filename, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_filename}")

    # Close the figure to free memory
    plt.close(fig)

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy types
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    """
    Main entry point
    """
    # Check if image path is provided
    if len(sys.argv) > 1:
        image_paths = [sys.argv[1]]
    else:
        # Use some default images from the Dataset folder
        dataset_dir = Path("Dataset")
        image_paths = [str(p) for p in list(dataset_dir.glob("*.jpeg"))[:5]]

    if not image_paths:
        logger.error("No images to process")
        return

    results = []

    for path in image_paths:
        logger.info(f"=== Processing image: {path} ===")
        detection_result = detect_with_debug(path)

        if detection_result:
            # Log summary
            num_detections = len(detection_result["detections"])
            logger.info(f"Found {num_detections} detections in {detection_result['inference_time']:.2f} seconds")

            if num_detections > 0:
                best_conf = detection_result["detections"][0]["confidence"]
                logger.info(f"Best detection confidence: {best_conf:.4f}")

                # Count detections that would pass normal filters
                would_pass = sum(1 for d in detection_result["detections"]
                               if all(d["would_pass_normal_filters"].values()))
                logger.info(f"Detections that would pass normal filters: {would_pass}/{num_detections}")

            # Visualize results
            visualize_results(path, detection_result)

            # Add to results list
            results.append(detection_result)

    # Save results to JSON file using custom encoder for NumPy types
    with open("debug_detection_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Results saved to debug_detection_results.json")

if __name__ == "__main__":
    main()