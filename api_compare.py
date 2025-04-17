#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import logging
import time
from pathlib import Path
import json
import requests
from ultralytics import YOLO
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

def detect_with_direct_model(image_path, model, conf_threshold=0.01, filter_params=None):
    """
    Detect license plates directly using the model
    """
    if filter_params is None:
        filter_params = {
            'min_aspect_ratio': 1.0,
            'max_aspect_ratio': 8.0,
            'min_area_percentage': 0.1,
            'max_area_percentage': 60.0,
            'conf_threshold': 0.2
        }

    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # Log original image info
        logger.info(f"Image: {image_path}, shape: {image.shape}")

        # Preprocess image
        processed_img, transform_params = preprocess_image(image)

        # Run inference with low confidence threshold
        start_time = time.time()
        results = model.predict(
            source=processed_img,
            conf=conf_threshold,
            verbose=False
        )
        inference_time = time.time() - start_time

        # Process results
        all_detections = []
        best_detection = None

        if not results or len(results) == 0 or not hasattr(results[0], 'boxes') or len(results[0].boxes) == 0:
            return {
                "image_path": image_path,
                "direct_model": {
                    "detection_found": False,
                    "inference_time": inference_time,
                    "detections": []
                }
            }

        # Process boxes
        result = results[0]
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
                "area_percentage": float(area_percentage),
                "passes_filters": (
                    filter_params['min_aspect_ratio'] <= aspect_ratio <= filter_params['max_aspect_ratio'] and
                    filter_params['min_area_percentage'] <= area_percentage <= filter_params['max_area_percentage'] and
                    conf >= filter_params['conf_threshold']
                )
            }
            all_detections.append(detection)

            # Update best detection if this is the first valid one or has higher confidence
            if detection["passes_filters"] and (best_detection is None or detection["confidence"] > best_detection["confidence"]):
                best_detection = detection

        # Sort detections by confidence
        all_detections.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "image_path": image_path,
            "direct_model": {
                "detection_found": best_detection is not None,
                "inference_time": inference_time,
                "best_detection": best_detection,
                "all_detections": all_detections,
                "detection_count": len(all_detections),
                "valid_detection_count": sum(1 for d in all_detections if d["passes_filters"])
            }
        }

    except Exception as e:
        logger.error(f"Error in direct model detection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def detect_with_api(image_path, api_url="http://localhost:8000/api/v1/detect", conf_threshold=0.2, skip_enhancement=True):
    """
    Detect license plates using the API
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None

        # Read the image file
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # First try a ping to see if the API is up
        try:
            ping_response = requests.get("http://localhost:8000/api/v1/ping", timeout=3)
            if ping_response.status_code != 200:
                logger.error(f"API not responding properly: {ping_response.status_code}")
                return {
                    "image_path": image_path,
                    "api": {
                        "error": f"API ping failed with status {ping_response.status_code}",
                        "detection_found": False
                    }
                }
        except requests.exceptions.RequestException as e:
            logger.error(f"API server not running: {str(e)}")
            return {
                "image_path": image_path,
                "api": {
                    "error": "API server not running",
                    "detection_found": False
                }
            }

        # Call the API
        files = {'car_image': (os.path.basename(image_path), image_data, 'image/jpeg')}
        data = {
            'return_type': 'json',
            'conf_threshold': conf_threshold,
            'skip_enhancement': skip_enhancement
        }

        start_time = time.time()
        response = requests.post(
            api_url,
            files=files,
            data=data,
            timeout=30
        )
        api_time = time.time() - start_time

        if response.status_code != 200:
            logger.error(f"API error: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return {
                "image_path": image_path,
                "api": {
                    "error": f"API returned status {response.status_code}",
                    "response_text": response.text,
                    "detection_found": False,
                    "response_time": api_time
                }
            }

        result = response.json()

        return {
            "image_path": image_path,
            "api": {
                "detection_found": result.get("detection_found", False),
                "confidence": result.get("confidence"),
                "bounding_box": result.get("bounding_box"),
                "response_time": api_time,
                "api_response": result
            }
        }

    except Exception as e:
        logger.error(f"Error in API detection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "image_path": image_path,
            "api": {
                "error": str(e),
                "detection_found": False
            }
        }

def compare_detections(image_path, model=None, api_url="http://localhost:8000/api/v1/detect", filter_params=None):
    """
    Compare direct model detection vs API detection
    """
    if model is None:
        model = load_model()
        if model is None:
            logger.error("Failed to load model")
            return None

    # Run both detection methods
    direct_result = detect_with_direct_model(image_path, model, filter_params=filter_params)
    api_result = detect_with_api(image_path, api_url)

    if direct_result is None:
        logger.error(f"Direct model detection failed for {image_path}")
        return None

    # Combine results
    combined_result = {
        "image_path": image_path,
        "direct_model": direct_result.get("direct_model", {}),
        "api": api_result.get("api", {})
    }

    # Check for discrepancies
    direct_detected = combined_result["direct_model"].get("detection_found", False)
    api_detected = combined_result["api"].get("detection_found", False)

    combined_result["discrepancy"] = direct_detected != api_detected

    if direct_detected and api_detected:
        # Calculate IoU between detections
        direct_box = combined_result["direct_model"]["best_detection"]["box"]
        api_box = combined_result["api"]["bounding_box"]

        if direct_box and api_box:
            # Calculate intersection
            x1 = max(direct_box[0], api_box[0])
            y1 = max(direct_box[1], api_box[1])
            x2 = min(direct_box[2], api_box[2])
            y2 = min(direct_box[3], api_box[3])

            if x2 >= x1 and y2 >= y1:
                intersection_area = (x2 - x1) * (y2 - y1)
                direct_area = (direct_box[2] - direct_box[0]) * (direct_box[3] - direct_box[1])
                api_area = (api_box[2] - api_box[0]) * (api_box[3] - api_box[1])
                union_area = direct_area + api_area - intersection_area

                iou = intersection_area / union_area if union_area > 0 else 0
                combined_result["iou"] = iou
                combined_result["similar_detection"] = iou > 0.5
            else:
                combined_result["iou"] = 0
                combined_result["similar_detection"] = False

    return combined_result

def analyze_results(results):
    """
    Analyze comparison results
    """
    if not results:
        logger.error("No results to analyze")
        return

    # Count metrics
    total_images = len(results)
    direct_detections = sum(1 for r in results if r.get("direct_model", {}).get("detection_found", False))
    api_detections = sum(1 for r in results if r.get("api", {}).get("detection_found", False))
    discrepancies = sum(1 for r in results if r.get("discrepancy", False))

    # Calculate statistics
    direct_percentage = (direct_detections / total_images) * 100 if total_images > 0 else 0
    api_percentage = (api_detections / total_images) * 100 if total_images > 0 else 0
    discrepancy_percentage = (discrepancies / total_images) * 100 if total_images > 0 else 0

    # Calculate average response times
    direct_times = [r.get("direct_model", {}).get("inference_time", 0) for r in results if "direct_model" in r]
    api_times = [r.get("api", {}).get("response_time", 0) for r in results if "api" in r and "response_time" in r.get("api", {})]

    avg_direct_time = sum(direct_times) / len(direct_times) if direct_times else 0
    avg_api_time = sum(api_times) / len(api_times) if api_times else 0

    # Create summary
    summary = {
        "total_images": total_images,
        "direct_model_detections": direct_detections,
        "direct_model_detection_percentage": direct_percentage,
        "api_detections": api_detections,
        "api_detection_percentage": api_percentage,
        "discrepancies": discrepancies,
        "discrepancy_percentage": discrepancy_percentage,
        "avg_direct_model_time": avg_direct_time,
        "avg_api_time": avg_api_time
    }

    # List images with discrepancies
    discrepant_images = [r["image_path"] for r in results if r.get("discrepancy", False)]
    summary["discrepant_images"] = discrepant_images

    # Group by issue type
    direct_only = [r["image_path"] for r in results
                  if r.get("direct_model", {}).get("detection_found", False) and not r.get("api", {}).get("detection_found", False)]
    api_only = [r["image_path"] for r in results
               if not r.get("direct_model", {}).get("detection_found", False) and r.get("api", {}).get("detection_found", False)]

    summary["direct_model_only_detections"] = direct_only
    summary["api_only_detections"] = api_only

    return summary

def main():
    """
    Main entry point
    """
    # Get image files
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        # Use default paths - sample images
        dataset_dir = Path("Dataset")
        all_images = list(dataset_dir.glob("*.jpeg"))
        # Use a representative sample
        import random
        random.seed(42)  # For reproducibility
        sample_size = min(15, len(all_images))
        image_paths = [str(p) for p in random.sample(all_images, sample_size)]

    # Load model once
    model = load_model()
    if model is None:
        logger.error("Failed to load model")
        return

    # Run comparisons
    results = []

    for path in tqdm(image_paths):
        logger.info(f"Processing {path}")
        result = compare_detections(path, model)
        if result:
            results.append(result)

            # Log result
            direct_detected = result["direct_model"].get("detection_found", False)
            api_detected = result["api"].get("detection_found", False)
            discrepancy = result.get("discrepancy", False)

            status = "MATCH" if not discrepancy else "DISCREPANCY"
            direct_conf = result["direct_model"].get("best_detection", {}).get("confidence", 0) if direct_detected else 0
            api_conf = result["api"].get("confidence", 0) if api_detected else 0

            logger.info(f"{status}: Direct Model: {direct_detected} (conf: {direct_conf:.4f}), API: {api_detected} (conf: {api_conf:.4f})")

    # Analyze results
    summary = analyze_results(results)

    # Print summary
    logger.info("\n=== Detection Comparison Summary ===")
    logger.info(f"Total images: {summary['total_images']}")
    logger.info(f"Direct model detections: {summary['direct_model_detections']} ({summary['direct_model_detection_percentage']:.1f}%)")
    logger.info(f"API detections: {summary['api_detections']} ({summary['api_detection_percentage']:.1f}%)")
    logger.info(f"Discrepancies: {summary['discrepancies']} ({summary['discrepancy_percentage']:.1f}%)")
    logger.info(f"Average direct model time: {summary['avg_direct_model_time']:.3f}s")
    logger.info(f"Average API time: {summary['avg_api_time']:.3f}s")

    # Save results and summary to JSON
    full_results = {
        "summary": summary,
        "detailed_results": results
    }

    with open("api_comparison_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info("Results saved to api_comparison_results.json")

if __name__ == "__main__":
    main()