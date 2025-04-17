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
import itertools
from tqdm import tqdm

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

        # Load model
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

def preprocess_image(image, target_size=1024):
    """
    Basic preprocessing - resize while preserving aspect ratio
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image

    # Store original dimensions
    original_height, original_width = image.shape[:2]

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

def detect_license_plate(image_path, model, base_conf_threshold=0.01, filter_params=None):
    """
    Detect license plates with flexible filtering parameters
    """
    if filter_params is None:
        filter_params = {
            'min_aspect_ratio': 1.0,
            'max_aspect_ratio': 8.0,
            'min_area_percentage': 0.1,
            'max_area_percentage': 60.0,
            'conf_threshold': 0.2
        }

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None

    # Preprocess image
    processed_img, transform_params = preprocess_image(image)

    # Run inference
    results = model.predict(
        source=processed_img,
        conf=base_conf_threshold,  # Use very low threshold for initial detection
        verbose=False
    )

    # Process results
    all_detections = []
    valid_detections = []

    if not results or len(results) == 0:
        return {"image_path": image_path, "all_detections": [], "valid_detections": []}

    result = results[0]

    # Check if there are any boxes
    if not hasattr(result, 'boxes') or len(result.boxes) == 0:
        return {"image_path": image_path, "all_detections": [], "valid_detections": []}

    # Process all boxes
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for i, (box, conf) in enumerate(zip(boxes, confs)):
        # Get coordinates
        x1, y1, x2, y2 = map(int, box)

        # Map to original image space
        orig_coords = map_coordinates_to_original([x1, y1, x2, y2], transform_params)
        ox1, oy1, ox2, oy2 = orig_coords

        # Calculate metrics
        width = ox2 - ox1
        height = oy2 - oy1
        aspect_ratio = width / height if height > 0 else 0

        img_area = image.shape[0] * image.shape[1]
        box_area = width * height
        area_percentage = (box_area / img_area) * 100

        # Create detection dict
        detection = {
            "confidence": float(conf),
            "box": [int(coord) for coord in [ox1, oy1, ox2, oy2]],
            "aspect_ratio": float(aspect_ratio),
            "area_percentage": float(area_percentage)
        }
        all_detections.append(detection)

        # Apply custom filters
        if (filter_params['min_aspect_ratio'] <= aspect_ratio <= filter_params['max_aspect_ratio'] and
            filter_params['min_area_percentage'] <= area_percentage <= filter_params['max_area_percentage'] and
            conf >= filter_params['conf_threshold']):
            valid_detections.append(detection)

    # Sort detections by confidence
    all_detections.sort(key=lambda x: x["confidence"], reverse=True)
    valid_detections.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "image_path": image_path,
        "all_detections": all_detections,
        "valid_detections": valid_detections
    }

def optimize_filter_parameters(image_paths, ground_truth=None):
    """
    Find optimal filter parameters by testing different combinations
    """
    # Load model once
    model = load_model()
    if model is None:
        logger.error("Failed to load model")
        return

    # Define parameter ranges to test
    param_ranges = {
        'min_aspect_ratio': [0.8, 1.0, 1.2],
        'max_aspect_ratio': [5.0, 6.0, 8.0],
        'min_area_percentage': [0.05, 0.1, 0.2],
        'max_area_percentage': [40.0, 60.0, 80.0],
        'conf_threshold': [0.05, 0.1, 0.2, 0.3]
    }

    # Create all combinations of parameters
    keys = list(param_ranges.keys())
    param_combinations = list(itertools.product(
        param_ranges['min_aspect_ratio'],
        param_ranges['max_aspect_ratio'],
        param_ranges['min_area_percentage'],
        param_ranges['max_area_percentage'],
        param_ranges['conf_threshold']
    ))

    logger.info(f"Testing {len(param_combinations)} parameter combinations on {len(image_paths)} images")

    best_params = None
    best_detection_count = 0
    best_avg_confidence = 0

    results = []

    # Test each parameter combination
    for combo in tqdm(param_combinations):
        filter_params = {
            'min_aspect_ratio': combo[0],
            'max_aspect_ratio': combo[1],
            'min_area_percentage': combo[2],
            'max_area_percentage': combo[3],
            'conf_threshold': combo[4]
        }

        # Skip invalid combinations
        if filter_params['min_aspect_ratio'] >= filter_params['max_aspect_ratio'] or \
           filter_params['min_area_percentage'] >= filter_params['max_area_percentage']:
            continue

        image_results = []
        total_confidence = 0
        detection_count = 0

        for path in image_paths:
            detection_result = detect_license_plate(path, model, base_conf_threshold=0.01, filter_params=filter_params)

            if detection_result:
                valid_detections = detection_result.get("valid_detections", [])
                detection_found = len(valid_detections) > 0
                confidence = valid_detections[0]["confidence"] if detection_found else 0

                image_result = {
                    "image": path,
                    "detection_found": detection_found,
                    "confidence": confidence
                }

                image_results.append(image_result)

                if detection_found:
                    detection_count += 1
                    total_confidence += confidence

        # Calculate average confidence
        avg_confidence = total_confidence / detection_count if detection_count > 0 else 0

        # Record results for this parameter set
        param_result = {
            "filter_params": filter_params,
            "detection_count": detection_count,
            "total_confidence": total_confidence,
            "avg_confidence": avg_confidence,
            "image_results": image_results
        }

        results.append(param_result)

        # Update best parameters if better
        if detection_count > best_detection_count or \
           (detection_count == best_detection_count and avg_confidence > best_avg_confidence):
            best_params = filter_params
            best_detection_count = detection_count
            best_avg_confidence = avg_confidence

            logger.info(f"New best parameters: {best_params}")
            logger.info(f"Detection count: {best_detection_count}/{len(image_paths)}, Avg confidence: {best_avg_confidence:.4f}")

    # Save all results to JSON
    with open("filter_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to filter_optimization_results.json")

    # Return the best parameters
    return best_params

def main():
    """
    Main entry point
    """
    # Get image files
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        # Use default paths - sample a subset of images
        dataset_dir = Path("Dataset")
        all_images = list(dataset_dir.glob("*.jpeg"))
        # Use a random sample of images to speed up testing
        import random
        random.seed(42)  # For reproducibility
        sample_size = min(15, len(all_images))
        image_paths = [str(p) for p in random.sample(all_images, sample_size)]

    # Run optimization
    best_params = optimize_filter_parameters(image_paths)

    if best_params:
        logger.info("\n=== Best Filter Parameters ===")
        for param, value in best_params.items():
            logger.info(f"{param}: {value}")

if __name__ == "__main__":
    main()